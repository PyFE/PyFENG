import abc
import numpy as np


class ParamABC(abc.ABC):
    """
    Abstract base class for the *model* (parameter) side of option pricing.

    Holds model parameters and the forward/discount-factor helpers that depend
    on them. Deliberately has no ``price()`` method — that lives in PricerABC.

    Concrete classes should inherit both ParamABC (or a subclass) and PricerABC
    (or a subclass), e.g.::

        class HestonFft(HestonABC, FftABC): ...
    """

    intr, divr, is_fwd = 0.0, 0.0, False

    def __init__(self):
        pass

    def params_kw(self):
        """
        Model parameters as a dictionary.
        """
        raise NotImplementedError

    def params_hash(self):
        dct = self.params_kw()
        return hash((frozenset(dct.keys()), frozenset(dct.values())))

    def forward(self, spot, texp):
        """
        Forward price.

        Args:
            spot: spot price
            texp: time to expiry

        Returns:
            forward price
        """
        if self.is_fwd:
            return np.array(spot)
        else:
            return np.array(spot) * np.exp((self.intr - self.divr) * np.array(texp))

    def _fwd_factor(self, spot, texp):
        """
        Forward price, discount factor, and dividend factor.

        Args:
            spot: spot (or forward) price
            texp: time to expiry

        Returns:
            (forward, discounting factor, dividend factor)
        """
        df = np.exp(-self.intr * np.array(texp))
        if self.is_fwd:
            divf = 1
            fwd = np.array(spot)
        else:
            divf = np.exp(-self.divr * np.array(texp))
            fwd = np.array(spot) * divf / df
        return fwd, df, divf


class SigmaParam(ParamABC):
    """
    Abstract base class for the *model* (parameter) side of option pricing.

    Holds model parameters and the forward/discount-factor helpers that depend
    on them. Deliberately has no ``price()`` method — that lives in PricerABC.

    Concrete classes should inherit both ParamABC (or a subclass) and PricerABC
    (or a subclass), e.g.::

        class HestonFft(HestonABC, FftABC): ...
    """

    sigma = None

    def __init__(self, sigma, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        self.sigma = sigma
        self.intr = intr
        self.divr = divr
        self.is_fwd = is_fwd

    def params_kw(self):
        """
        Model parameters as a dictionary.
        """
        return {
            "sigma": self.sigma,
            "intr": self.intr,
            "divr": self.divr,
            "is_fwd": self.is_fwd,
        }



class SvParam(SigmaParam):
    """
    Abstract base class for stochastic-volatility model parameters.

    Adds vov, rho, mr, theta on top of the base sigma/intr/divr/is_fwd.
    """

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


class CevParam(SigmaParam):
    """
    Abstract base class for CEV model parameters.

    Adds beta (elasticity) on top of the base sigma/intr/divr/is_fwd.
    """

    model_type = "Cev"
    beta = 0.5

    def __init__(self, sigma, beta=0.5, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility
            beta: elasticity parameter. 0.5 by default
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)
        self.beta = beta

    def params_kw(self):
        params = super().params_kw()
        extra = {"beta": self.beta}
        return {**params, **extra}


class SabrParam(SigmaParam):
    """
    Abstract base class for SABR model parameters.

    Adds vov, rho, beta on top of the base sigma/intr/divr/is_fwd.
    """

    vov, beta, rho = 0.0, 1.0, 0.0
    model_type = "SABR"
    var_process = False

    def __init__(self, sigma, vov=0.1, rho=0.0, beta=1.0, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            beta: elasticity parameter. 1.0 by default
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)
        self.vov = vov
        self.rho = rho
        self.beta = beta

    def params_kw(self):
        params = super().params_kw()
        extra = {"vov": self.vov, "beta": self.beta, "rho": self.rho}
        return {**params, **extra}


class NsvhParam(SigmaParam):

    beta = 0.0  ## should be fixed as 0
    vov, rho, lam = 0.0, 0.0, 0.0
    model_type = "Nsvh"
    var_process = False

    def __init__(self, sigma, vov=0.1, rho=0.0, lam=0.0, intr=0.0, divr=0.0, is_fwd=False, beta=None):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            lam: lambda. Normal SABR if 0, Johnson's SU if 1 (same as `Nsvh1`)
            beta: elasticity parameter. should be 0 or None.
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        # Make sure beta = 0
        if beta is not None and not np.isclose(beta, 0.0):
            print(f"Ignoring beta = {beta}...")
        self.lam = lam
        self.vov = vov
        self.rho = rho
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)

    def params_kw(self):
        params = super().params_kw()
        extra = {"vov": self.vov, "lam": self.lam, "rho": self.rho}
        return {**params, **extra}  # Py 3.9, params | extra

    @classmethod
    def init_benchmark(cls, set_no=None):
        """
        Initiate a SABR model with stored benchmark parameter sets

        Args:
            set_no: set number

        Returns:
            Dataframe of all test cases if set_no = None
            (model, Dataframe of result, params) if set_no is specified

        References:
            - Antonov, Alexander, Konikov, M., & Spector, M. (2013). SABR spreads its wings. Risk, 2013(Aug), 58–63.
            - Antonov, Alexandre, Konikov, M., & Spector, M. (2019). Modern SABR Analytics. Springer International Publishing. https://doi.org/10.1007/978-3-030-10656-0
            - Antonov, Alexandre, & Spector, M. (2012). Advanced analytics for the SABR model. Available at SSRN. https://ssrn.com/abstract=2026350
            - Cai, N., Song, Y., & Chen, N. (2017). Exact Simulation of the SABR Model. Operations Research, 65(4), 931–951. https://doi.org/10.1287/opre.2017.1617
            - Korn, R., & Tang, S. (2013). Exact analytical solution for the normal SABR model. Wilmott Magazine, 2013(7), 64–69. https://doi.org/10.1002/wilm.10235
            - Lewis, A. L. (2016). Option valuation under stochastic volatility II: With Mathematica code. Finance Press.
            - von Sydow, L., ..., Haentjens, T., & Waldén, J. (2018). BENCHOP - SLV: The BENCHmarking project in Option Pricing – Stochastic and Local Volatility problems. International Journal of Computer Mathematics, 1–14. https://doi.org/10.1080/00207160.2018.1544368
        """
        this_dir, _ = os.path.split(__file__)
        file = os.path.join(this_dir, "data/sabr_benchmark.xlsx")
        df_param = pd.read_excel(file, sheet_name="Param", index_col="Sheet")

        if set_no is None:
            return df_param
        else:
            df_val = pd.read_excel(file, sheet_name=str(set_no))
            param = df_param.loc[set_no]
            args_model = {k: param[k] for k in ("sigma", "vov", "rho", "beta")}
            args_pricing = {k: param[k] for k in ("texp", "spot")}

            assert df_val.columns[0] == "Strike"
            args_pricing["strike"] = df_val.values[:, 0]

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
