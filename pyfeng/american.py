import numpy as np
import scipy.stats as spst
import scipy.optimize as spopt
from .bsm import Bsm
from .opt_abc import OptABC


class AmerLi2010QdPlus(OptABC):
    """
    Implementation of "initial guess" and QD+ of Li (2010)

    References:
        - Li M (2010) Analytical approximations for the critical stock prices of American options: a performance comparison. Rev Deriv Res 13:75–99. https://doi.org/10.1007/s11147-009-9044-3
        - Andersen L, Lake M, Offengenden D (2016) High-performance American option pricing. JCF 39–87. https://doi.org/10.21314/JCF.2016.312
    """

    bsm_model = None

    def __init__(self, sigma, *args, **kwargs):
        super().__init__(sigma, *args, **kwargs)
        self.bsm_model = Bsm(sigma, *args, **kwargs)

    def exer_bdd_ig(self, strike, texp, cp=-1):
        """
        "Initial Guess" (IG) in p.80, Eqs (7)-(8) of Li (2010)

        Args:
            strike: strike price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            Exercise boundary (critical stock price)
        """
        s0 = strike * np.fmin(1, self.intr/np.fmax(self.divr, np.finfo(float).tiny))
        mm = 2*self.intr / self.sigma**2
        nn_m1 = 2*(self.intr - self.divr) / self.sigma**2 - 1
        q_inf = -0.5 * (nn_m1 + np.sqrt(nn_m1**2 + 4*mm))
        s_inf = strike / (1 - 1/q_inf)
        theta = strike/(s_inf - strike) * ((self.intr - self.divr)*texp + 2*self.sigma*np.sqrt(texp))
        bdd = s0 + (s_inf - s0)*(1 - np.exp(-theta))
        return bdd

    def exer_bdd(self, strike, texp, cp=-1):
        """
        QD+ method in p.85 of Li (2010) or Appendix A of Andersen et al. (2016)

        Args:
            strike: strike price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            Exercise boundary (critical stock price)
        """
        root = spopt.root(self.zero_func, x0=strike*np.ones_like(texp), args=(strike, texp, cp))
        rv = root.x[0] if np.isscalar(texp) else root.x
        return rv

    def zero_func(self, spot_bdd, strike, texp, cp=-1):
        """
        Function to solve for QD+ method.
        Eq. (34) of Li (2010) with c replaced with c0 or Eq. (66) of Andersen et al. (2016)

        Args:
            spot_bdd: boundary
            strike: strike price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            function value
        """

        fwd, df, divf = self._fwd_factor(spot_bdd, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).tiny)
        d1 = np.log(fwd/strike)/sigma_std
        d1 += 0.5*sigma_std

        nn_m1 = 2*(self.intr - self.divr) / self.sigma**2 - 1
        mm = 2*self.intr / self.sigma**2
        hh = 1.0 - df

        qqd_sqrt = np.sqrt(nn_m1**2 + 4*mm/hh)
        qqd_prime = mm / (hh**2 * qqd_sqrt)

        p_euro = self.bsm_model.price(strike, spot_bdd, texp, cp=cp)
        theta = self.bsm_model.theta(strike, spot_bdd, texp, cp=cp)

        qqd_c0 = mm/qqd_sqrt * (df / hh - theta / self.intr / (strike - spot_bdd - p_euro) - df * qqd_prime / qqd_sqrt)
        qqd_c0 -= 0.5*(nn_m1 + qqd_sqrt)

        zero = (1 - divf*spst.norm._cdf(-d1)) * spot_bdd + qqd_c0 * (strike - spot_bdd - p_euro)
        return zero

    def price(self, strike, spot, texp, cp=-1):
        """
        American call/put option price. Eq. (29) of Li (2010)

        Args:
            strike: strike price.
            spot: spot (or forward) price.
            texp: time to expiry.
            cp: 1/-1 for call/put option.

        Returns:
            option price
        """

        assert cp < 0
        fwd, df, divf = self._fwd_factor(spot, texp)
        spot_bdd = self.exer_bdd(strike, texp, cp=cp)

        if spot <= spot_bdd:
            return strike - spot

        p_euro = self.bsm_model.price(strike, spot_bdd, texp, cp=cp)
        theta = self.bsm_model.theta(strike, spot_bdd, texp, cp=cp)

        nn_m1 = 2*(self.intr - self.divr) / self.sigma**2 - 1
        mm = 2*self.intr / self.sigma**2
        hh = 1.0 - df

        qqd_sqrt = np.sqrt(nn_m1**2 + 4*mm/hh)
        qqd_prime = mm / (hh**2 * qqd_sqrt)
        qqd = -0.5*(nn_m1 + qqd_sqrt)
        c0 = mm/qqd_sqrt * (df/hh - theta/self.intr/(strike - spot_bdd - p_euro) - df*qqd_prime/qqd_sqrt)
        b = -df * mm * qqd_prime / (2*qqd_sqrt)

        log = np.log(spot/spot_bdd)
        p_amer = self.bsm_model.price(strike, spot, texp, cp=cp) \
                 + (strike - spot_bdd - p_euro) / (1 - log*(c0 + b*log)) * np.exp(qqd*log)

        return p_amer
