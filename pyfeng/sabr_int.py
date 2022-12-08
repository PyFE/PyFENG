import abc
import numpy as np
from . import sabr
import scipy.special as spsp
from . import opt_smile_abc as smile


class SabrMixtureABC(sabr.SabrABC, smile.MassZeroABC, abc.ABC):

    correct_fwd = False

    @staticmethod
    def avgvar_lndist(vovn):
        """
        Lognormal distribution parameters (mean, sigma) of the normalized average variance:
        (1/T) \int_0^T e^{2*vov Z_t - vov^2 t} dt = \int_0^1 e^{2 vovn Z_s - vovn^2 s} ds
        where vovn = vov*sqrt(T). See p.2 in Choi & Wu (2021).

        Args:
            vovn: vov * sqrt(texp)

        Returns:
            (m1, sig)
            True distribution should be multiplied by sigma^2 * texp

        References
            - Choi J, Wu L (2021) A note on the option price and ‘Mass at zero in the uncorrelated SABR model and implied volatility asymptotics.’ Quantitative Finance 21:1083–1086. https://doi.org/10.1080/14697688.2021.1876908
        """
        vovn2 = vovn**2
        ww = np.exp(vovn2)
        m1 = np.where(vovn2 > 1e-6, (ww - 1) / vovn2, 1 + vovn2 / 2 * (1 + vovn2 / 3))
        var_m1sq_ratio = (10 + ww*(6 + ww*(3 + ww))) / 15 * m1 * vovn2
        sig = np.sqrt(np.where(vovn2 > 1e-8, np.log(1.0 + var_m1sq_ratio), 4/3 * vovn2))
        ### Equivalently ....
        #m2_m1sq_ratio = (5 + ww * (4 + ww * (3 + ww * (2 + ww)))) / 15
        #sig = np.sqrt(np.where(vovn2 > 1e-8, np.log(m2_m1sq_ratio), 4/3 * vovn2))

        return m1, sig

    @abc.abstractmethod
    def cond_spot_sigma(self, texp, fwd):
        # return (fwd, vol, weight) each 1d array
        return NotImplementedError

    def price(self, strike, spot, texp, cp=1):
        fwd = self.forward(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)
        #if self.beta == 0:
        #    kk = strike - fwd + 1.0
        #    fwd = 1.0
        #else:
        kk = strike / fwd

        fwd_ratio, vol_ratio, ww = self.cond_spot_sigma(texp, fwd)
        # print(f'E(F) = {np.sum(fwd_ratio * ww)}')

        if self.correct_fwd:
            fwd_ratio /= np.sum(fwd_ratio*ww)
        assert np.isclose(np.sum(ww), 1)

        # apply if beta > 0
        if self.beta > 0:
            ind = (fwd_ratio*ww > 1e-16)
        else:
            ind = (fwd_ratio*ww > -999)

        fwd_ratio = np.expand_dims(fwd_ratio[ind], -1)
        vol_ratio = np.expand_dims(vol_ratio[ind], -1)
        ww = np.expand_dims(ww[ind], -1)

        base_model = self.base_model(alpha * vol_ratio)
        base_model.is_fwd = True
        price_vec = base_model.price(kk, fwd_ratio, texp, cp=cp)
        price = fwd * np.sum(price_vec * ww, axis=0)
        return price

    def mass_zero(self, spot, texp, log=False, mu=0):

        fwd = self.forward(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)
        fwd_ratio, vol_ratio, ww = self.cond_spot_sigma(texp, fwd)

        if self.correct_fwd:
            fwd_ratio /= np.sum(fwd_ratio*ww)
        assert np.isclose(np.sum(ww), 1)

        base_model = self.base_model(alpha * vol_ratio)
        base_model.is_fwd = True

        if log:
            log_mass = np.log(ww) + base_model.mass_zero(fwd_ratio, texp, log=True)
            log_max = np.amax(log_mass)
            log_mass -= log_max
            log_mass = log_max + np.log(np.sum(np.exp(log_mass)))
            return log_mass
        else:
            mass = base_model.mass_zero(fwd_ratio, texp, log=False)
            mass = np.sum(mass * ww)
            return mass


class SabrUncorrChoiWu2021(SabrMixtureABC):
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

    def cond_spot_sigma(self, texp, _):

        assert np.isclose(self.rho, 0.0)

        m1, fac = self.avgvar_lndist(self.vov * np.sqrt(texp))

        zz, ww = spsp.roots_hermitenorm(self.n_quad)
        ww /= np.sqrt(2 * np.pi)

        vol_ratio = np.sqrt(m1) * np.exp(0.5 * (zz - 0.5 * fac) * fac)

        return np.full(self.n_quad, 1.0), vol_ratio, ww


class SabrMixture(SabrMixtureABC):
    n_quad = None
    dist = 'ln'

    def n_quad_vovn(self, vovn):
        return self.n_quad or np.floor(3 + 4*vovn)

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

    def cond_avgvar(self, vovn, zhat):

        m1, m2 = self.cond_avgvar_mnc2(vovn, zhat)
        m1m2_ratio = m2 / m1**2

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
            r_var = m1 * r_var**2
            r_vol = np.sqrt(r_var)
        else:
            pass

        assert r_var.shape == w2.shape
        return r_var, r_vol, w2

    def cond_spot_sigma(self, texp, fwd):
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)
        rho_alpha = self.rho * alpha

        zhat, w0 = self.zhat_weight(vovn)  # column vectors
        r_var, r_vol, w123 = self.cond_avgvar(vovn, zhat)
        w0123 = w0 * w123

        r_vol *= rhoc  # matrix
        exp_plus2 = np.exp(vovn*zhat)

        if np.isclose(self.beta, 0):
            fwd_ratio = 1 + (rho_alpha/self.vov) * (exp_plus2 - 1)
        elif self.beta > 0:
            fwd_ratio = rho_alpha * ((exp_plus2 - 1)/self.vov - 0.5*rho_alpha*texp*r_var)
            np.exp(fwd_ratio, out=fwd_ratio)
        else:
            fwd_ratio = 1.0

        return fwd_ratio.flatten(), r_vol.flatten(), w0123.flatten()
