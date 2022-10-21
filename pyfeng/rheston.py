import abc
from . import sv_abc as sv


class RoughHestonABC(sv.SvABC, abc.ABC):
    """
    Rough Heston model Abstract class
    """

    model_type = "rHeston"
    var_process = True

    alpha = 0.62

    def __init__(self, sigma, vov=0.01, rho=0.0, mr=0.01, theta=None, alpha=0.62, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility or variance at t=0.
            vov: volatility of volatility
            rho: correlation between price and volatility
            mr: mean-reversion speed (kappa)
            theta: long-term mean of volatility or variance. If None, same as sigma
            alpha: vol roughness parameter. 0.62 by default.
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        self.alpha = alpha
        super().__init__(sigma, vov, rho, mr, theta, intr=intr, divr=divr, is_fwd=is_fwd)
