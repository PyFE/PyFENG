import numpy as np
import abc
from . import bsm
from . import opt_smile_abc as smile


class SvABC(smile.OptSmileABC, abc.ABC):

    vov, rho, mr, sig_inf = 0.01, 0.0, 0.01, 1.0

    def __init__(self, sigma, vov=0.01, rho=0.0, mr=0.01, sig_inf=None, intr=0.0, divr=0.0, is_fwd=False):
        # Note:
        #    sigma^2: initial variance
        #    var_inf: long-term variance

        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)

        self.vov = vov
        self.rho = rho
        self.mr = mr
        self.sig_inf = sigma if sig_inf is None else sig_inf

    def params_kw(self):
        params1 = super().params_kw()
        params2 = {"vov": self.vov, "rho": self.rho, "mr": self.mr, "sig_inf": self.sig_inf}
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

    def set_mc_params(self, n_path, dt=0.1, rn_seed=None, antithetic=True):
        self.n_path = int(n_path)
        self.dt = dt
        self.rn_seed = rn_seed
        self.antithetic = antithetic
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)

    def base_model(self, vol):
        return bsm.Bsm(vol, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)

    def tobs(self, texp):
        n_steps = texp // self.dt + 1
        tobs = np.arange(n_steps + 1) / n_steps * texp
        return tobs

    @abc.abstractmethod
    def vol_paths(self, tobs):
        """
        Volatility or variance paths at tobs.
        The paths are standardized by sigma_0 = 1.0

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
