import numpy as np
from . import sabr
import scipy.special as spsp
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
        Choi, J., & Wu, L. (2021). A note on the option price and `Mass at zero in the uncorrelated
        SABR model and implied volatility asymptotics’. Quantitative Finance (Forthcoming).
        https://doi.org/10.1080/14697688.2021.1876908

        Gulisashvili, A., Horvath, B., & Jacquier, A. (2018). Mass at zero
        in the uncorrelated SABR model and implied volatility asymptotics.
        Quantitative Finance, 18(10), 1753–1765.
        https://doi.org/10.1080/14697688.2018.1432883
    """

    _base_beta = None
    n_quad = 9

    def __init__(self, sigma, vov=0.0, rho=0.0, beta=1.0, intr=0.0, divr=0.0, is_fwd=False, n_quad=9):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility. Should be 0 in this model.
            beta: elasticity parameter. 0.5 by default
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
            n_quad: number of quadrature points
        """
        assert abs(rho) < 1e-8
        self.n_quad = n_quad
        super().__init__(sigma, vov, 0.0, beta, intr=intr, divr=divr, is_fwd=is_fwd)

    @staticmethod
    def int_var_lndist(vovn):
        """
        Lognormal distribution parameters of integrated integrated variance:
        sigma^2 * texp * m1 * exp(sig*Z - 0.5*sig^2)

        Args:
            vovn: vov * sqrt(texp)

        Returns:
            (m1, sig)
            True distribution should be multiplied by sigma^2*t
        """
        v2 = vovn**2
        w = np.exp(v2)
        m1 = np.where(v2 > 1e-6, (w-1)/v2, 1+v2/2*(1+v2/3))
        m2m1ratio = (5 + w*(4 + w*(3 + w*(2 + w))))/15
        sig = np.sqrt(np.where(v2 > 1e-8, np.log(m2m1ratio), 4/3*v2))
        return m1, sig

    def price(self, strike, spot, texp, cp=1):
        assert (self._base_beta is None)
        m1, fac = self.int_var_lndist(self.vov * np.sqrt(texp))

        zz, ww = spsp.roots_hermitenorm(self.n_quad)
        ww /= np.sqrt(2 * np.pi)

        vol = self.sigma * np.sqrt(m1) * np.exp(0.5 * (zz - 0.5 * fac) * fac)

        p_grid = self._m_base(vol[:, None]).price(strike, spot, texp, cp=cp)
        p = np.sum(p_grid * ww[:, None], axis=0)
        return p

    def mass_zero(self, spot, texp, log=False, mu=0):
        m1, fac = self.int_var_lndist(self.vov * np.sqrt(texp))

        zz, ww = spsp.roots_hermitenorm(self.n_quad)
        ww /= np.sqrt(2 * np.pi)

        log_rn_deriv = 0.0 if mu == 0 else -mu * (zz + 0.5 * mu)
        zz += mu
        vol = self.sigma * np.sqrt(m1) * np.exp(0.5 * (zz - 0.5 * fac) * fac)

        if log:
            log_mass = np.log(ww) + log_rn_deriv + self._m_base(vol).mass_zero(spot, texp, log=True)
            log_max = np.amax(log_mass)
            log_mass -= log_max
            log_mass = log_max + np.log(np.sum(np.exp(log_mass)))
            return log_mass
        else:
            mass = self._m_base(vol).mass_zero(spot, texp, log=False)
            mass = np.sum(mass * ww * np.exp(log_rn_deriv))
            return mass
