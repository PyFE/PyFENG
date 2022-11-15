import numpy as np
import abc
import os
import pandas as pd
from . import bsm
from . import opt_smile_abc as smile


class SvABC(smile.OptSmileABC, abc.ABC):

    model_type: str = NotImplementedError
    var_process: bool = NotImplementedError
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
            "theta": self.theta,
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
        file = os.path.join(this_dir, f"data/{cls.model_type.lower()}_benchmark.xlsx")
        df_param = pd.read_excel(file, sheet_name="Param", index_col="Sheet")

        if set_no is None:
            return df_param
        else:
            df_val = pd.read_excel(file, sheet_name=str(set_no))
            param = df_param.loc[set_no].to_dict()
            args_model = {k: param[k] for k in param.keys() & {"sigma", "theta", "vov", "rho", "mr", "intr", "divr"}}
            args_pricing = {k: param[k] for k in param.keys() & {"texp", "spot"}}

            assert df_val.columns[0] == "Strike"
            args_pricing["strike"] = df_val.values[:, 0]

            if "CP" in df_val.columns:
                args_pricing["cp"] = df_val["CP"].values

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
    var_process: bool = NotImplementedError

    dt = 0.05
    n_path = 10000
    rn_seed = None
    rng = np.random.default_rng(None)
    rng_spawn = []
    antithetic = True
    correct_fwd = False
    result = {}

    def set_num_params(self, n_path=10000, dt=0.25, rn_seed=None, antithetic=True):
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
        seed_seq = np.random.SeedSequence(rn_seed)
        self.rng_spawn = [np.random.default_rng(s) for s in seed_seq.spawn(6)]
        self.result = {}

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
        if self.dt is None or self.dt >= texp:
            return np.array([texp])
        else:
            n_dt = np.ceil(texp / self.dt)
            tobs = np.arange(1, n_dt + 1) / n_dt * texp
        return tobs

    def rv_normal(self, spawn=0):
        if self.antithetic:
            zz = self.rng_spawn[spawn].standard_normal(size=self.n_path // 2)
            zz = np.stack([zz, -zz], axis=1).flatten()
        else:
            zz = self.rng_spawn[spawn].standard_normal(size=self.n_path)
        return zz

    def rv_uniform(self, spawn=0):
        if self.antithetic:
            zz = self.rng_spawn[spawn].uniform(size=self.n_path // 2)
            zz = np.stack([zz, 1-zz], axis=1).flatten()
        else:
            zz = self.rng_spawn[spawn].uniform(size=self.n_path)
        return zz

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

        if n_path is None:
            n_path = self.n_path

        if self.antithetic:
            # generate random number in the order of (path, time) first and transposed
            # in this way, the same paths are generated when increasing n_path
            bm_incr = self.rng_spawn[0].standard_normal((int(n_path // 2), n_dt)).T * np.sqrt(dt[:, None])
            bm_incr = np.stack([bm_incr, -bm_incr], axis=1).reshape((-1, n_path))
        else:
            bm_incr = self.rng_spawn[0].standard_normal(n_path, n_dt).T * np.sqrt(dt[:, None])

        if cum:
            np.cumsum(bm_incr, axis=0, out=bm_incr)

        return bm_incr

    @abc.abstractmethod
    def cond_spot_sigma(self, texp, var_0):
        """
        Returns new forward and volatility conditional on volatility path (e.g., sigma_T, integrated variance)
        The forward and volatility are standardized in the sense that F_0 = 1 and sigma_0 = 1
        Therefore, they should be scaled by the original F_0 and sigma_0 values.
        Volatility, not variance, is returned.

        Args:
            texp: time-to-expiry
            var_0: initial variance (or vol)

        Returns: (forward, volatility)
        """
        return NotImplementedError


    def price(self, strike, spot, texp, cp=1):

        kk = strike / spot
        scalar_output = np.isscalar(kk)
        kk = np.atleast_1d(kk)

        fwd_cond, sigma_cond = self.cond_spot_sigma(texp, self.sigma)

        fwd_mean = fwd_cond.mean()
        self.result['spot error'] = fwd_mean - 1
        if self.correct_fwd:
            fwd_cond /= fwd_mean

        sigma = np.sqrt(self.sigma) if self.var_process else self.sigma
        base_model = self.base_model(sigma * sigma_cond)
        price_grid = base_model.price(kk[:, None], fwd_cond, texp=texp, cp=cp)

        price = spot * np.mean(price_grid, axis=1)

        return price[0] if scalar_output else price

    @abc.abstractmethod
    def return_var_realized(self, texp, cond):
        return NotImplementedError

    def price_var_opt(self, strike, texp, cp=1):
        """
        Variance option price

        Args:
            strike: strike price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            variance option price
        """
        scalar_output = np.isscalar(strike)
        strike = np.atleast_1d(strike)
        p = np.zeros_like(strike)
        var = self.return_var_realized(texp)
        for i, k in enumerate(strike):
            p[i] = np.mean(np.fmax(np.sign(cp)*(var - k), 0))
        if scalar_output:
            p = p[0]

        return p * np.exp(-self.intr * texp)

    def strike_var_swap(self, texp, cond=True):
        """
        Variance swap price (fair strike)

        Args:
            texp: time to expiry

        Returns:
            Variance swap fair strike
        """

        var = self.return_var_realized(texp, cond=cond)
        return np.mean(var)

    def price_paths(self, tobs):
        price = np.ones((len(tobs)+1, self.n_path))
        dt_arr = np.diff(np.atleast_1d(tobs), prepend=0)
        s_0 = np.full(self.n_path, self.sigma)

        for k, dt in enumerate(dt_arr):
            spot, sigma = self.cond_spot_sigma(dt, s_0)

            xx = np.random.standard_normal(int(self.n_path // 2))
            xx = np.array([xx, -xx]).flatten('F')

            price[k+1, :] = spot * np.exp(sigma*np.sqrt(dt) * xx)

        np.cumprod(price, axis=0, out=price)

        return price

class SvMixtureABC(smile.OptSmileABC, abc.ABC):
    """
    Abstract Class for BS-mixture model for the BSM-based stochastic volatility models
    """
    var_process: bool = NotImplementedError

    correct_fwd = False
    result = {}

    def base_model(self, vol):
        return bsm.Bsm(vol, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)

    @abc.abstractmethod
    def cond_spot_sigma(self, texp):
        """
        Returns new forward and volatility conditional on volatility path (e.g., sigma_T, integrated variance)
        The forward and volatility are standardized in the sense that F_0 = 1 and sigma_0 = 1
        Therefore, they should be scaled by the original F_0 and sigma_0 values.
        Volatility, not variance, is returned.

        Args:
            texp: time-to-expiry

        Returns: (spot, volatility, weight)
        """
        return NotImplementedError

    def price(self, strike, spot, texp, cp=1):

        kk = strike / spot
        scalar_output = np.isscalar(kk)
        kk = np.atleast_1d(kk)

        spot_cond, sigma_cond, ww = self.cond_spot_sigma(texp)

        spot_mean = np.sum(spot_cond * ww)
        self.result['spot error'] = spot_mean - 1
        if self.correct_fwd:
            spot_cond /= spot_mean
        assert np.isclose(np.sum(ww), 1)

        spot_cond = np.expand_dims(spot_cond, axis=-1)
        sigma_cond = np.expand_dims(sigma_cond, axis=-1)
        ww = np.expand_dims(ww, axis=-1)

        sigma = np.sqrt(self.sigma) if self.var_process else self.sigma
        base_model = self.base_model(sigma * sigma_cond)
        price_vec = base_model.price(kk, spot_cond, texp=texp, cp=cp)
        price = spot * np.sum(price_vec * ww, axis=0)

        return price[0] if scalar_output else price
