import numpy as np
import abc
import os
import pandas as pd
from . import bsm
from . import opt_smile_abc as smile


class SvABC(smile.OptSmileABC, abc.ABC):

    vov, rho, mr, theta = 0.01, 0.0, 0.01, 1.0

    def __init__(
        self,
        sigma,
        vov=0.01,
        rho=0.0,
        mr=0.01,
        theta=None,
        intr=0.0,
        divr=0.0,
        is_fwd=False,
    ):
        """
        Args:
            sigma: model volatility or variance at t=0.
            vov: volatility of volatility
            rho: correlation between price and volatility
            mr: mean-reversion speed (kappa)
            theta: long-term mean of volatility or variance. If None, same as sigma
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
        params2 = {
            "vov": self.vov,
            "rho": self.rho,
            "mr": self.mr,
            "sig_inf": self.theta,
        }
        return {**params1, **params2}

    @classmethod
    def init_benchmark(cls, set_no=None):
        """
        Initiate an SV model with stored benchmark parameter sets

        Args:
            set_no: set number

        Returns:
            Dataframe of all test cases if set_no = None
            (model, Dataframe of result, params) if set_no is specified

        References:
        """

        this_dir, _ = os.path.split(__file__)
        file = os.path.join(this_dir, "data/sv_benchmark.xlsx")
        df_param = pd.read_excel(file, sheet_name="Param", index_col="Sheet")

        if set_no is None:
            return df_param
        else:
            df_val = pd.read_excel(file, sheet_name=str(set_no))
            param = df_param.loc[set_no]
            args_model = {k: param[k] for k in ("sigma", "theta", "vov", "rho", "mr", "intr")}
            args_pricing = {k: param[k] for k in ("texp", "spot")}

            assert df_val.columns[0] == "k" or df_val.columns[0] == "K"
            args_pricing["strike"] = df_val.values[:, 0]
            if df_val.columns[0] == "k":
                args_pricing["strike"] *= param["spot"]

            val = df_val[param["col_name"]].values
            is_iv = param["col_name"].startswith("IV")

            m = cls(**args_model)

            param_dict = {
                "args_pricing": args_pricing,
                "ref": param["Reference"],
                "val": val,
                "is_iv": is_iv,
            }

            return m, df_val, param_dict


class CondMcBsmABC(smile.OptSmileABC, abc.ABC):
    """
    Abstract Class for conditional Monte-Carlo method for BSM-based stochastic volatility models
    """

    dt = 0.05
    n_path = 10000
    rn_seed = None
    rng = np.random.default_rng(None)
    antithetic = True

    var_process = True
    correct_fwd = True

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
        n_steps = (texp // (2 * self.dt) + 1) * 2
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
            # generate random number in the order of (path, time) first and transposed
            # in this way, the same paths are generated when increasing n_path
            bm_incr = self.rng.standard_normal((int(n_path//2), n_dt)).T * np.sqrt(
                dt[:, None]
            )
            bm_incr = np.stack([bm_incr, -bm_incr], axis=1).reshape((-1, n_path))
        else:
            bm_incr = self.rng.standard_normal(n_path, n_dt).T * np.sqrt(dt[:, None])

        if cum:
            np.cumsum(bm_incr, axis=0, out=bm_incr)

        return bm_incr

    @abc.abstractmethod
    def cond_spot_sigma(self, var_0, texp):
        """
        Returns new forward and volatility conditional on volatility path (e.g., sigma_T, integrated variance)
        The forward and volatility are standardized in the sense that F_0 = 1 and sigma_0 = 1
        Therefore, they should be scaled by the original F_0 and sigma_0 values.
        Volatility, not variance, is returned.

        Args:
            var_0: initial variance (or vol)
            texp: time-to-expiry

        Returns: (forward, volatility)
        """

        return np.ones(self.n_path), np.ones(self.n_path)

    def price(self, strike, spot, texp, cp=1):

        kk = strike / spot
        scalar_output = np.isscalar(kk)
        kk = np.atleast_1d(kk)

        fwd_cond, sigma_cond = self.cond_spot_sigma(self.sigma, texp)

        sigma = np.sqrt(self.sigma) if self.var_process else self.sigma
        base_model = self.base_model(sigma * sigma_cond)
        price_grid = base_model.price(kk[:, None], fwd_cond, texp=texp, cp=cp)

        price = spot * np.mean(price_grid, axis=1)

        return price[0] if scalar_output else price

    def price_paths(self, tobs):
        price = np.ones((len(tobs)+1, self.n_path))
        dt_arr = np.diff(np.atleast_1d(tobs), prepend=0)
        s_0 = np.full(self.n_path, self.sigma)

        for k, dt in enumerate(dt_arr):
            spot, sigma = self.cond_spot_sigma(s_0, dt)

            xx = np.random.standard_normal(int(self.n_path // 2))
            xx = np.array([xx, -xx]).flatten('F')

            price[k+1, :] = spot * np.exp(sigma*np.sqrt(dt) * xx)

        np.cumprod(price, axis=0, out=price)

        return price
