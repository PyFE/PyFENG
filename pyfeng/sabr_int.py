import abc
import numpy as np
from . import sabr
import scipy.special as spsp
import scipy.stats as spst
from . import opt_smile_abc as smile


class SabrUncorrChoiWu2021(sabr.SabrABC, smile.MassZeroABC):
    """
    The uncorrelated SABR (rho=0) model pricing by approximating the integrated variance with
    a log-normal distribution.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> param = {"sigma": 0.4, "vov": 0.6, "rho": 0, "beta": 0.3, 'n_quad': 9}
        >>> fwd, texp = 0.05, 1
        >>> strike = np.array([0.4, 0.8, 1, 1.2, 1.6, 2.0]) * fwd
        >>> m = pf.SabrUncorrChoiWu2021(**param)
        >>> m.mass_zero(fwd, texp)
        0.7623543217183134
        >>> m.price(strike, fwd, texp)
        array([0.04533777, 0.04095806, 0.03889591, 0.03692339, 0.03324944,
               0.02992918])

    References:
        - Choi, J., & Wu, L. (2021). A note on the option price and `Mass at zero in the uncorrelated SABR model and implied volatility asymptotics’. Quantitative Finance (Forthcoming). https://doi.org/10.1080/14697688.2021.1876908
        - Gulisashvili, A., Horvath, B., & Jacquier, A. (2018). Mass at zero in the uncorrelated SABR model and implied volatility asymptotics. Quantitative Finance, 18(10), 1753–1765. https://doi.org/10.1080/14697688.2018.1432883
    """

    n_quad = 10

    @staticmethod
    def avgvar_lndist(vovn):
        """
        Lognormal distribution parameters of integrated integrated variance:
        sigma^2 * texp * m1 * exp(sig*Z - 0.5*sig^2)

        Args:
            vovn: vov * sqrt(texp)

        Returns:
            (m1, sig)
            True distribution should be multiplied by sigma^2*t
        """
        v2 = vovn ** 2
        w = np.exp(v2)
        m1 = np.where(v2 > 1e-6, (w - 1) / v2, 1 + v2 / 2 * (1 + v2 / 3))
        m2m1ratio = (5 + w * (4 + w * (3 + w * (2 + w)))) / 15
        sig = np.sqrt(np.where(v2 > 1e-8, np.log(m2m1ratio), 4 / 3 * v2))
        return m1, sig

    def price(self, strike, spot, texp, cp=1):
        assert np.isclose(self.rho, 0.0)
        assert self._base_beta is None
        m1, fac = self.avgvar_lndist(self.vov * np.sqrt(texp))

        zz, ww = spsp.roots_hermitenorm(self.n_quad)
        ww /= np.sqrt(2 * np.pi)

        vol = self.sigma * np.sqrt(m1) * np.exp(0.5 * (zz - 0.5 * fac) * fac)

        p_grid = self.base_model(vol[:, None]).price(strike, spot, texp, cp=cp)
        p = np.sum(p_grid * ww[:, None], axis=0)
        return p

    def mass_zero(self, spot, texp, log=False, mu=0):
        assert np.isclose(self.rho, 0.0)
        m1, fac = self.avgvar_lndist(self.vov * np.sqrt(texp))

        zz, ww = spsp.roots_hermitenorm(self.n_quad)
        ww /= np.sqrt(2 * np.pi)

        log_rn_deriv = 0.0 if mu == 0 else -mu * (zz + 0.5 * mu)
        zz += mu
        vol = self.sigma * np.sqrt(m1) * np.exp(0.5 * (zz - 0.5 * fac) * fac)

        if log:
            log_mass = (
                np.log(ww)
                + log_rn_deriv
                + self.base_model(vol).mass_zero(spot, texp, log=True)
            )
            log_max = np.amax(log_mass)
            log_mass -= log_max
            log_mass = log_max + np.log(np.sum(np.exp(log_mass)))
            return log_mass
        else:
            mass = self.base_model(vol).mass_zero(spot, texp, log=False)
            mass = np.sum(mass * ww * np.exp(log_rn_deriv))
            return mass


class SabrCondDistABC(sabr.SabrABC, abc.ABC):
    correct_fwd = False

    @abc.abstractmethod
    def cond_spot_sigma(self, fwd, texp):
        # return (fwd, vol, weight) each 1d array
        return NotImplementedError

    def price(self, strike, spot, texp, cp=1):
        fwd = spot * (1.0 if self.is_fwd else np.exp(texp * (self.intr - self.divr)))

        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)
        #if self.beta == 0:
        #    kk = strike - fwd + 1.0
        #    fwd = 1.0
        #else:
        kk = strike / fwd

        fwd_eff, vol_eff, ww = self.cond_spot_sigma(fwd, texp)
        # print(f'E(F) = {np.sum(fwd_eff*ww)}')
        if self.correct_fwd:
            fwd_eff /= np.sum(fwd_eff*ww)
        assert np.isclose(np.sum(ww), 1)

        # apply if beta > 0
        if self.beta > 0:
            ind = (fwd_eff*ww > 1e-16)
        else:
            ind = (fwd_eff*ww > -999)

        fwd_eff = np.expand_dims(fwd_eff[ind], -1)
        vol_eff = np.expand_dims(vol_eff[ind], -1)
        ww = np.expand_dims(ww[ind], -1)

        base_model = self.base_model(alpha * vol_eff)
        price_vec = base_model.price(kk, fwd_eff, texp, cp=cp)
        price = fwd * np.sum(price_vec * ww, axis=0)
        return price


class SabrCondQuad(SabrCondDistABC):
    n_quad = None
    dist = 'ln'

    def n_quad_vovn(self, vovn):
        return self.n_quad or np.floor(3 + 4*vovn)

    @staticmethod
    def condvar_m1(z, vovn):
        """
        Calculate the conditional mean of the normalized integrated variance of SABR model
        E{ int_0^1 exp{2 vov sqrt(T) Z_s - vov^2 T s^2} ds | Z_1 = z }
        int_0^T exp{vov Z_t - vov^2/2 t^2} dt =
        """
        m1 = (spst.norm.cdf(z + vovn) - spst.norm.cdf(z - vovn))/(2*vovn*spst.norm.pdf(z))\
             *np.exp(0.5*vovn**2)
        return m1 #*np.exp(vovn*z)

    @staticmethod
    def condvar_m2(z, vovn):
        """
        Calculate the 2nd moment of the normalized integrated variance of SABR model
        E{ int_0^1 exp{2 vov sqrt(T) Z_s - vov^2 T s^2} ds | Z_1 = z }
        int_0^T exp{vov Z_t - vov^2/2 t^2} dt =
        """
        m2 = (SabrCondQuad.condvar_m1(z, 2*vovn)
              - SabrCondQuad.condvar_m1(z, vovn)*np.cosh(z*vovn))/vovn**2
        return m2 #*np.exp(2*vovn*z)

    def zhat_weight(self, vovn):
        """
        The points and weights for the terminal volatility

        Args:
            vovn: vov * sqrt(texp)

        Returns:
            points and weights in column vector
        """

        npt = self.n_quad_vovn(vovn)
        zhat, ww = spsp.roots_hermitenorm(npt)
        ww /= np.sqrt(2*np.pi)
        zhat = zhat[:, None] - 0.5*vovn
        ww = ww[:, None]
        return zhat, ww

    def cond_int_var(self, vovn, zhat):

        m1 = self.condvar_m1(zhat, vovn)
        m2 = self.condvar_m2(zhat, vovn)
        m1m2_ratio = m2 / m1**2
        m1 *= np.exp(zhat * vovn)

        w2 = np.ones_like(zhat)

        if self.dist.lower() == 'm1':
            r_var = m1
            r_vol = np.sqrt(r_var)
        elif self.dist.lower() == 'ln':
            r_var = m1 / np.sqrt(np.sqrt(m1m2_ratio))
            r_vol = np.sqrt(r_var)
        elif self.dist.lower() == 'ig':  # inverse Gaussian
            lam = m1 / (m1m2_ratio - 1.0)
            r_var = 1 - 1 / (8 * lam) * (1 - 9 / (2 * 8 * lam) * (1 - 25 / (6 * 8 * lam)))
            r_var[lam < 100] = spsp.kv(0, lam[lam < 100]) / spsp.kv(-0.5, lam[lam < 100])
            r_var = m1 * r_var ** 2
            r_vol = np.sqrt(r_var)
        else:
            pass

        assert r_var.shape == w2.shape
        return r_var, r_vol, w2

    def cond_spot_sigma(self, fwd, texp):
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)
        rho_alpha = self.rho * alpha

        zhat, w0 = self.zhat_weight(vovn)  # column vectors
        r_var, r_vol, w123 = self.cond_int_var(vovn, zhat)
        w0123 = w0 * w123

        r_vol *= rhoc  # matrix
        exp_plus = np.exp(0.5*vovn*zhat)
        exp_plus2 = exp_plus**2

        if self.beta == 0:
            fwd_ratio = 1 + (rho_alpha/self.vov) * (exp_plus2 - 1)
            #fwd_ratio = fwd_ratio * np.ones(self.n_quad[1])
        elif self.beta > 0:
            fwd_ratio = rho_alpha * ((exp_plus2 - 1)/self.vov - 0.5*rho_alpha*texp*r_var)
            fwd_ratio = np.exp(fwd_ratio)
        else:
            fwd_ratio = 1.0

        return fwd_ratio.flatten(), r_vol.flatten(), w0123.flatten()
