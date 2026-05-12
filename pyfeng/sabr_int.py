import abc
import warnings
import numpy as np
from . import sabr
import scipy.special as spsp
import scipy.stats as spst
import scipy.integrate as spint
from .opt_abc import MassZeroABC
from .util import MathFuncs
from .disthelper import DistLognormal


class SabrMixtureABC(sabr.SabrABC, MassZeroABC):

    correct_fwd = False

    @abc.abstractmethod
    def cond_spot_sigma(self, texp, fwd):
        # return (fwd, vol, weight) each 1d array
        raise NotImplementedError

    def price(self, strike, spot, texp, cp=1):
        fwd = self.forward(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)
        #if self.beta == 0:
        #    kk = strike - fwd + 1.0
        #    fwd = 1.0
        #else:
        kk = strike / fwd

        fwd_ratio, vol_ratio, ww = self.cond_spot_sigma(texp, fwd)


        if self.correct_fwd:
            fwd_ratio /= np.sum(fwd_ratio*ww)
        assert np.isclose(np.sum(ww), 1)

        # apply if beta > 0
        #if self.beta > 0:
        #    ind = (fwd_ratio*ww > 1e-16)
        #else:
        #    ind = (fwd_ratio*ww > -999)

        n_dim = ww.ndim
        fwd_ratio = np.expand_dims(fwd_ratio, -1)
        vol_ratio = np.expand_dims(vol_ratio, -1)
        ww = np.expand_dims(ww, -1)

        base_model = self.base_model(alpha * vol_ratio)
        base_model.is_fwd = True
        price_vec = base_model.price(kk, fwd_ratio, texp, cp=cp)
        price = fwd * np.sum(price_vec * ww, axis=tuple(range(n_dim)))
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
        - Choi J, Wu L (2021) A note on the option price and ‘Mass at zero in the uncorrelated SABR model and implied volatility asymptotics.’ Quantitative Finance 21:1083–1086. https://doi.org/10.1080/14697688.2021.1876908
        - Gulisashvili A, Horvath B, Jacquier A (2018) Mass at zero in the uncorrelated SABR model and implied volatility asymptotics. Quantitative Finance 18(10):1753–1765. https://doi.org/10.1080/14697688.2018.1432883
    """

    n_quad = 10

    def cond_spot_sigma(self, texp, _):

        assert np.isclose(self.rho, 0.0)

        vovn = self.vov * np.sqrt(texp)
        m1, var_scaled, *_ = self.avgvar_mvsk(vovn)
        avgvar, ww = DistLognormal.from_mv(m1, var_scaled).quad(self.n_quad)

        return np.full(self.n_quad, 1.0), np.sqrt(avgvar), ww


class SabrMixtureChoi(SabrMixtureABC):
    """
    SABR model pricing via 2-D Gaussian quadrature over the joint distribution of
    the terminal volatility ratio (σ_T/σ_0) and the normalised integrated variance
    (I_0^T = (1/T)∫₀ᵀ (σₜ/σ₀)² dt).

    The method is the quadrature analogue of the Monte-Carlo algorithm in
    Choi, Hu & Kwok (2026); the quadrature adaptation itself is unpublished.
    Conditional on σ_T/σ_0 (outer quadrature, n_quad[0] nodes), the terminal
    forward F̄_T is approximated by the frozen-coefficient CEV formula
    (single time-step h = T applied to the full expiry).  Conditional on I_0^T
    (inner quadrature, n_quad[1] nodes, approximated by a shifted log-normal with
    λ = 5/6), the residual orthogonal component is priced by the base CEV model.

    **Accuracy note**: because a single step h = T is used, the frozen-coefficient
    approximation degrades for large ``vov * sqrt(texp)``.  Accuracy is reasonable
    for ``vov * sqrt(texp)`` ≲ 0.5–0.6 but deteriorates noticeably beyond that.

    References:
        - Choi J, Hu L, Kwok YK (2026) Efficient and accurate simulation of the
          stochastic-alpha-beta-rho model. European Journal of Operational Research
          329:166–179. https://doi.org/10.1016/j.ejor.2025.09.027
    """

    n_quad = (7, 5)
    sln_lam = 5 / 6

    def cond_spot_sigma(self, texp, fwd):
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)
        rho_alpha = self.rho * alpha

        sig_t, w0, zhat = DistLognormal(sig=vovn).quad(self.n_quad[0], return_zhat=True)  # sig_t / sig_0
        sig_t = sig_t[:, None]
        w0 = w0[:, None]
        zhat = zhat[:, None]

        m1, var_scaled, *_ = self.cond_avgvar_mvsk(vovn, zhat)
        avgvar, w2 = DistLognormal.from_mv(m1, var_scaled, lam=self.sln_lam).quad(self.n_quad[1])

        r_vol = rhoc * np.sqrt(avgvar)
        w0123 = w0 * w2

        if np.isclose(self.beta, 0):
            fwd_ratio = 1 + (rho_alpha/self.vov) * (sig_t - 1)
        elif self.beta > 0:
            fwd_ratio = rho_alpha * ((sig_t - 1)/self.vov - 0.5*rho_alpha*texp*avgvar)
            np.exp(fwd_ratio, out=fwd_ratio)
            np.fmax(fwd_ratio, 1e-20, out=fwd_ratio)  # prevent CEV blow-up at fwd=0 (tiny^betac would underflow)
        else:
            ValueError(f'Unsupported beta={self.beta}')

        return fwd_ratio, r_vol, w0123


class SabrNormAnalytic(sabr.SabrABC):
    """
    Approximated analytic 1-d integral of Antonov et al. (2019, S 3.4.5) with Elliptic integral of 2nd kind (E).

    `price` method implements the 2nd order approximation and optional correction with Gaussian quadrature (when `quad_correction` is True).
    `price_quad` method impelements the generic integral with `numpy.integrate.quad` function.

    References:
        - Antonov A, Konikov M, Spector M (2019) Modern SABR Analytics. Springer International Publishing, Cham
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
    """
    quad_correction = False
    n_quad = 9  # only used when quad_correction is True

    def __init__(self, sigma, vov=0.1, rho=0.0, beta=None, intr=0.0, divr=0.0, is_fwd=False, is_atmvol=False):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            beta: elasticity parameter. should be 0 or None.
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
            is_atmvol: If True, use `sigma` as the ATM normal vol
        """
        # Make sure beta = 0
        if beta is not None and not np.isclose(beta, 0.0):
            warnings.warn(f"Ignoring beta = {beta}.")
        super().__init__(sigma, vov, rho, beta=0, intr=intr, divr=divr, is_fwd=is_fwd)

    def price(self, strike, spot, texp, cp=1):

        fwd, df, _ = self._fwd_factor(spot, texp)

        rhoc2 = 1.0 - self.rho**2
        xi = self.vov * np.sqrt(texp)/2
        k = self.vov/self.sigma*(strike - fwd) + self.rho
        V = np.sqrt(k**2 + rhoc2)
        u0 = np.arcsinh(np.abs((self.rho*V - k)/rhoc2)) / (2*xi)

        tmp1 = np.abs(self.rho - k/V)
        tmp2 = 1. - self.rho*k/V
        A = 0.75*tmp1
        B = 0.75*(tmp2 - 5/16*tmp1**2)

        R = MathFuncs.mills_ratio(np.array([u0 - xi, u0, u0 + xi]))
        opt_val = ((R[0] - R[2]) + A*(R[0] + R[2] - 2*R[1]) + 2*B*(R[0] - R[2] - 2*xi*(1. - u0*R[1]))) / xi
        # At the end of above, n(u0)/xi should be multiplied. Only /xi is multiplied and n(u0) will be applied later.

        if self.quad_correction:
            v_value, v_weight = spsp.roots_genlaguerre(self.n_quad, 1.5)
            axis = len(np.broadcast_shapes(np.shape(strike), np.shape(spot), np.shape(texp)))
            v_value = np.expand_dims(v_value, list(range(1, axis+1)))
            v_weight = np.expand_dims(v_weight, list(range(1, axis+1)))

            uu = np.sqrt(u0**2 + 2*v_value)
            v_weight = v_weight / (v_value*np.sqrt(v_value)) / uu

            ch = np.cosh(2*xi*uu)
            diff = np.sqrt(np.fmax(rhoc2*(ch**2. - 1.0 - (k - self.rho*ch)**2), 0.0))

            v_p = self.rho*k + rhoc2*ch + diff
            V = np.sqrt(k**2 + rhoc2)
            fn = (2/np.pi)*np.sqrt(v_p/V)*spsp.ellipe(np.fmin(2*diff/v_p, 1.0)) - 1. - xi*(uu - u0)*(A + B*xi*(uu - u0))
            base = MathFuncs.mills_ratio(uu + xi) + MathFuncs.mills_ratio(uu - xi)
            opt_val += np.sum(fn*base*v_weight, axis=0)

        opt_val *= 0.5*self.sigma*np.sqrt(texp*V)*np.exp(-0.5*xi**2) * spst.norm._pdf(u0)
        opt_val += np.fmax(cp*(fwd - strike), 0.0)
        opt_val *= df
        return opt_val


class SabrNormEllipeInt(sabr.SabrABC):
    """
    Approximated analytic 1-d integral of Antonov et al. (2019, S 3.4.5) with Elliptic integral of 2nd kind (E).

    `price` method implements the 2nd order approximation and optional correction with Gaussian quadrature (when `quad_correction` is True).
    `price_quad` method impelements the generic integral with `numpy.integrate.quad` function.

    References:
        - Antonov A, Konikov M, Spector M (2019) Modern SABR Analytics. Springer International Publishing, Cham
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
    """
    quad_eps = 1e-6  # used for price_quad

    def __init__(self, sigma, vov=0.1, rho=0.0, beta=None, intr=0.0, divr=0.0, is_fwd=False, is_atmvol=False):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            beta: elasticity parameter. should be 0 or None.
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
            is_atmvol: If True, use `sigma` as the ATM normal vol
        """
        # Make sure beta = 0
        if beta is not None and not np.isclose(beta, 0.0):
            warnings.warn(f"Ignoring beta = {beta}.")
        super().__init__(sigma, vov, rho, beta=0, intr=intr, divr=divr, is_fwd=is_fwd)

    @staticmethod
    def hh_xi(uu, k, xi, rho):
        """
        H_xi (u) function. It's used in `price_quad` method

        Args:
            uu: argument >= u_0. uu = s/(2 xi)
            k: vov/sigma0*(strike - fwd) + rho
            xi: vov*sqrt(T)/2
            rho: correlation

        Returns:

        """

        rhoc2 = 1.0 - rho**2
        ch = np.cosh(2*xi*uu)
        diff = np.sqrt(np.fmax(rhoc2*(ch**2. - 1.0 - (k - rho*ch)**2), 0.0))
        v_p = np.fmax(rho*k + rhoc2*ch + diff, 0.0)
        return np.sqrt(v_p) * spsp.ellipe(np.fmin(2*diff/v_p, 1.0))

    @staticmethod
    def hh_xi_approx(uu, k, xi, rho):  # u = s/(2*xi)
        """
        H_xi (u) function approximated to the 2nd order.
        It's NOT used for pricing, but implemented for plot

        Args:
            uu: argument >= u_0. uu = s/(2 xi)
            k: vov/sigma0*(strike - fwd) + rho
            xi: vov*sqrt(T)/2
            rho: correlation

        Returns:

        """

        rhoc2 = 1.0 - rho**2
        # u = s/(2*xi)
        V = np.sqrt(k**2 + rhoc2)
        u0 = np.arcsinh(np.abs((rho*V - k)/rhoc2)) / (2*xi)
        tmp1 = np.abs(rho - k/V)
        tmp2 = 1. - rho*k/V
        A = 0.75*tmp1
        B = 0.75*(tmp2 - 5/16*tmp1**2)

        return np.sqrt(V)*np.pi/2 * (1. + xi*(uu - u0)*(A + B*xi*(uu - u0)))

    def price(self, strike, spot, texp, cp=1):

        fwd, df, _ = self._fwd_factor(spot, texp)

        rhoc2 = 1.0 - self.rho**2
        xi = 0.5 * self.vov * np.sqrt(texp)
        k = self.vov/self.sigma*(strike - fwd) + self.rho
        V = np.sqrt(k**2 + rhoc2)
        u0 = np.arcsinh(np.abs((self.rho*V - k)/rhoc2)) / (2*xi)

        def integrand(uu_, xi_, k_):
            rv = self.hh_xi(uu_, k_, xi_, self.rho)
            rv *= spst.norm._pdf(uu_)*(MathFuncs.mills_ratio(uu_ + xi_) + MathFuncs.mills_ratio(uu_ - xi_))
            return rv

        def integral(u0_, xi_, k_):
            return spint.quad(integrand, u0_, np.sqrt(u0_**2 + 73), (xi_, k_), epsabs=self.quad_eps, epsrel=self.quad_eps)

        opt_val, est_error = np.vectorize(integral)(u0, xi, k)
        opt_val *= self.sigma*np.sqrt(texp)/np.pi*np.exp(-0.5*xi**2)
        opt_val += np.fmax(cp*(fwd - strike), 0.0)
        opt_val *= df

        return opt_val
