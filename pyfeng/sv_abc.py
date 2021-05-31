import numpy as np
import abc
from . import bsm
from . import opt_smile_abc as smile


class SvABC(smile.OptSmileABC, abc.ABC):

    vov, rho, mr, theta = 0.01, 0.0, 0.01, 1.0

    def __init__(self, sigma, vov=0.01, rho=0.0, mr=0.01, theta=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility at t=0. variance = sigma**2
            vov: volatility of volatility
            rho: correlation between price and volatility
            mr: mean-reversion speed (kappa)
            theta: long-term mean of volatility. For variance process, use theta**2. If None, same as sigma
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """

        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)

        self.vov = vov
        self.rho = rho
        self.mr = mr
        self.theta = sigma if theta is None else theta

    def params_kw(self):
        params1 = super().params_kw()
        params2 = {"vov": self.vov, "rho": self.rho, "mr": self.mr, "sig_inf": self.theta}
        return {**params1, **params2}


class CondMcBsmABC(smile.OptSmileABC, abc.ABC):
    """
    Abstract Class for conditional Monte-Carlo method for BSM-based stochastic volatility models
    """

    dt = 0.05
    n_path = 10000
    rn_seed = None
    rng = np.random.default_rng(None)
    antithetic = True

    
    def set_mc_params(self, n_path=10000, dt=0.05, rn_seed=None, antithetic=True):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step for Euler/Milstein steps
            rn_seed: random number seed
            antithetic: antithetic
        """
        self.n_path = int(n_path)
        self.dt = dt
        self.rn_seed = rn_seed
        self.antithetic = antithetic
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)

    def base_model(self, vol):
        return bsm.Bsm(vol, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)

    def tobs(self, texp):
        """
        Return array of observation time in even size. 0 is not included.

        Args:
            texp: time-to-expiry

        Returns:
            array of observation time
        """
        n_steps = (texp//(2*self.dt)+1)*2
        tobs = np.arange(1, n_steps + 0.1) / n_steps * texp
        return tobs

    def _bm_incr(self, tobs, cum=False, n_path=None):
        """
        Calculate incremental Brownian Motions

        Args:
            tobs: observation times (array). 0 is not included.
            cum: return cumulative values if True
            n_path: number of paths. If None (default), use the stored one.

        Returns:
            price path (time, path)
        """
        dt = np.diff(np.atleast_1d(tobs), prepend=0)
        n_dt = len(dt)

        n_path = n_path or self.n_path

        if self.antithetic:
            # generate random number in the order of path, time, asset and transposed
            # in this way, the same paths are generated when increasing n_path
            bm_incr = self.rng.normal(size=(int(n_path/2), n_dt)).T * np.sqrt(dt[:, None])
            bm_incr = np.stack([bm_incr, -bm_incr], axis=-1).reshape((-1, n_path))
        else:
            bm_incr = np.random.randn(n_path, n_dt).T * np.sqrt(dt[:, None])

        if cum:
            np.cumsum(bm_incr, axis=0, out=bm_incr)

        return bm_incr

    @abc.abstractmethod
    def vol_paths(self, tobs):
        """
        Volatility or variance paths at 0 and tobs.

        Args:
            tobs: observation time (array)

        Returns:
            2d array of (time, path)
        """

        return np.ones(size=(len(tobs), self.n_path))

    @abc.abstractmethod
    def cond_fwd_vol(self, texp):
        """
        Returns new forward and volatility conditional on volatility path (e.g., sigma_T, integrated variance)
        The forward and volatility are standardized in the sense that F_0 = 1 and sigma_0 = 1
        Therefore, they should be scaled by the original F_0 and sigma_0 values

        Args:
            texp: time-to-expiry

        Returns: (forward, volatility)
        """

        return np.ones(self.n_path), self.sigma * np.ones(self.n_path)

    def price(self, strike, spot, texp, cp=1):

        kk = strike / spot
        scalar_output = np.isscalar(kk)
        kk = np.atleast_1d(kk)

        fwd_cond, vol_cond = self.cond_fwd_vol(texp)

        base_model = self.base_model(self.sigma*vol_cond)
        price_grid = base_model.price(kk[:, None], fwd_cond, texp=texp, cp=cp)

        price = spot * np.mean(price_grid, axis=1)

        return price[0] if scalar_output else price
