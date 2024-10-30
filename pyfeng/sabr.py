# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10
@author: jaehyuk

"""

import abc
import copy
import os
import pandas as pd
import numpy as np
import scipy.optimize as spop
from scipy import stats as spst
from numpy.polynomial.hermite_e import hermeval

from . import opt_smile_abc as smile
from . import bsm
from . import norm
from . import cev
from .util import MathFuncs, DistHelperLnShift

class SabrABC(smile.OptSmileABC, abc.ABC):
    vov, beta, rho = 0.0, 1.0, 0.0
    model_type = "SABR"
    var_process = False
    #### vol_beta: the beta for the volatility to choose _m_vol. If None (by default) vol_beta = beta
    _base_beta = None

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
        return {**params, **extra}  # Py 3.9, params | extra

    def _variables(self, fwd, texp):
        betac = 1.0 - self.beta
        alpha = self.sigma / np.power(fwd, betac)  # if self.beta > 0.0 else self.sigma
        rho2 = self.rho**2
        rhoc = np.sqrt(1.0 - rho2)
        vovn = self.vov * np.sqrt(np.maximum(texp, 1e-64))
        return alpha, betac, rhoc, rho2, vovn

    def base_model(self, vol):
        """
        Create base model based on _base_beta value: `Norm` for 0, Cev for (0,1), and `Bsm` for 1
        If `_base_beta` is None, use `base` instead.

        Args:
            vol: base model volatility

        Returns: model
        """
        base_beta = self.beta if self._base_beta is None else self._base_beta

        if np.isclose(base_beta, 1):
            return bsm.Bsm(vol, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)
        elif np.isclose(base_beta, 0):
            return norm.Norm(vol, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)
        else:
            return cev.Cev(vol, beta=base_beta, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)

    def vol_smile(self, strike, spot, texp, cp=None, model=None):
        if model is None:
            model = "norm" if np.isclose(self.beta, 0) else "bsm"

        return super().vol_smile(strike, spot, texp, cp=cp, model=model)

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

    @staticmethod
    def _vv(zz, rho):
        return np.sqrt(1 + zz * (zz + 2 * rho))

    @staticmethod
    def _hh(zz, rho):
        """
        H(z) in the paper

        Args:
            zz: array
            rho: scalar

        Returns: H(z)

        """
        rho2 = rho * rho

        with np.errstate(divide="ignore", invalid="ignore"):  # suppress error for zz=0
            vv = SabrVolApproxABC._vv(zz, rho)
            zz_xx = np.where(zz + rho > 0,
                             zz / np.log((vv + zz + rho) / (1 + rho)),
                             zz / np.log((1 - rho)/(vv - zz - rho))
                             )

        # expansion for small |zz|
        # Series[z/Log[(z + ρ + Sqrt[1 + z^2 + 2 z ρ])/(1 + ρ)], {z, 0, 6}]
        # 1 + (ρ z)/2 + (1/6 - ρ^2/4) z^2 + 1/24 ρ (6 ρ^2 - 5) z^3 + (-(5 ρ^4)/16 + ρ^2/3 - 17/360) z^4 + 1/480 ρ (210 ρ^4 - 275 ρ^2 + 74) z^5 + O(z^6)
        # (Taylor series)
        idx = np.abs(zz) < 1e-3
        if np.any(idx):
            zz_xx[idx] = 1 + zz[idx]*(rho/2 + zz[idx]*(1/6 - rho2/4 + rho*(6*rho2 - 5)/24 * zz[idx]))

        return zz_xx

    @staticmethod
    def cond_avgvar_mfn(z, vovn_k):
        assert vovn_k >= 0.0
        if np.abs(vovn_k) > 0.01:
            ncdf_diff = spst.norm.cdf(-np.abs(z) + vovn_k) - spst.norm.cdf(-np.abs(z) - vovn_k)
            m = np.sqrt(np.pi / 2) / vovn_k * ncdf_diff * np.exp((z**2 + vovn_k**2) / 2)
        else:
            ## Alternative evaluation usign Hermite polynomial when vv is very small
            ### coef = vv^(2n) / (n+1)!  for even n
            # coef = np.array([1, 0, vv**2/6, 0, vv**4/120, 0, vv**6/5040, 0, vv**8/362880])
            coef = vovn_k / np.arange(1, 8)  # from 1 to 7
            coef[0] = 1.0
            np.cumprod(coef, out=coef)
            coef[1::2] = 0.0  # zero out even terms
            m = hermeval(z, coef) * np.exp(vovn_k**2 / 2)

        return m

    @staticmethod
    def cond_avgvar_mvsk(vovn, zhat, mnc=False):
        """
        Mean, scaled variance, skewness, and ex-kurtosis of the conditional average variance.

        int_0^1 exp{2 vovn Z_s - vovn^2 s} ds | Z_1 - (vovn/2) = z
             conditional on z = log(sigma_T/sigma_0) / (vov*sqrt(T))

        See Appendix B in Choi et al. (2019), Eqs. (B5) and (B6) in Kennedy et al. (2012)

        Args:
            vovn: vov * sqrt(T)
            zhat: log(sigma_T/sigma_0) / (vov*sqrt(T)) = Z_1 - (vovn/2)
            mnc: if True, return (mean, mnc2, mnc3, mnc4)

        Returns:
            (mean, scaled var, skewness, ex-kurtosis) or (mean, mnc2, mnc3, mnc4)

        References:
            - Choi J, Liu C, Seo BK (2019) Hyperbolic normal stochastic volatility model. J Futures Mark 39:186–204. https://doi.org/10.1002/fut.21967
            - Kennedy JE, Mitra S, Pham D (2012) On the Approximation of the SABR Model: A Probabilistic Approach. Applied Mathematical Finance 19:553–586. https://doi.org/10.1080/1350486X.2011.646523
        """

        exp = np.exp(vovn * zhat)
        cosh = (exp + 1./exp)/2

        m1 = SabrABC.cond_avgvar_mfn(zhat, vovn)
        m2 = SabrABC.cond_avgvar_mfn(zhat, 2*vovn)
        m3 = SabrABC.cond_avgvar_mfn(zhat, 3*vovn)
        m4 = SabrABC.cond_avgvar_mfn(zhat, 4*vovn)

        #### non-central (raw) moments
        mnc2 = (m2 - m1*cosh)/vovn**2
        mnc3 = (3*m3 - 8*m2*cosh + m1*(4*cosh**2 + 1))/(8*vovn**4)
        mnc4 = (2*m4 - 9*m3*cosh + m2*(12*cosh**2 + 2) - m1*cosh*(4*cosh**2 + 3))/(24*vovn**6)

        if mnc:
            return m1*exp, mnc2*exp**2, mnc3*exp**3, mnc4*exp**4

        # Code from https://www.statsmodels.org/0.8.0/_modules/statsmodels/stats/moment_helpers.html#mnc2mvsk
        # mnc2mvsk in statsmodels behaves weidly when zhat is a column vector, so use the code here.
        mc2 = mnc2 - m1**2
        mc3 = mnc3 - (3 * m1 * mc2 + m1**3)  # 3rd central moment
        mc4 = mnc4 - (4 * m1 * mc3 + 6 * m1**2 * mc2 + m1**4)

        vovn2 = vovn**2
        if vovn2 < 0.01:
            # When vovn is small, calculation is unstable. so use expansions.
            var_scaled = vovn2*(1/3 - (6 - zhat**2)/45*vovn2)
            skew = np.sqrt(vovn2*(108/25 - (4/875)*(51*zhat**2 - 716)*vovn2))
            exkur = vovn2 * (276/35 - (8/175)*(9*zhat**2 - 158)*vovn2)
        else:
            var_scaled = np.divide(mc2, m1**2)
            skew = np.divide(mc3, mc2**1.5)
            exkur = np.divide(mc4, mc2**2) - 3.0

        return m1*exp, var_scaled, skew, exkur

    @staticmethod
    def cond_avgvar_lnshift_params(vovn, zhat, ratio=1.):
        """
        Find the prameters (mu, sigma, ratio) to fit moments

            Y ~ (1 - ratio) * mu + ratio * mu * exp(sigma * Z - sigma^2/2)

        Args:
            vovn: vov * sqrt(texp)
            zhat: Z - 0.5*vov
            ratio: ratio for the lognormal distribution. 1.0 by default

        Returns:
            (mean, sig, ratio)
        """

        # only ratio cares, so use remove_exp=True
        # but multiply exp(vovn * z) to m1 before returning
        m1, var_scaled, skew, exkur = SabrABC.cond_avgvar_mvsk(vovn, zhat)
        sln = DistHelperLnShift(lam=ratio)
        sln.fit(mvs=[m1, var_scaled, skew], lam=ratio)

        return m1, sln.sig, sln.lam


    @staticmethod
    def avgvar_mnc4(vovn):
        """
        First 4 non-central(raw) moments of the normalized average variance:
        (1/T) \int_0^T e^{2*vov Z_t - vov^2 Z_t} = \int_0^1 e^{2 vovn Z_s - vovn^2 s} ds
        where vovn = vov*sqrt(T). See p.2 in Choi & Wu (2021).

        The implementation is not stable for very small value of vovn. Use avgvar_mvsk instead.
        This is used for testing the validity of avgvar_mvsk method.

        Args:
            vovn: vov * sqrt(texp)

        Returns:
            np.array([m1, n2, m3, m4])

        References
            - McGhee WA (2021) An artificial neural network representation of the SABR stochastic volatility model. Journal of Computational Finance
        """
        vovn2 = vovn**2
        ww = np.exp(vovn2)
        m1 = np.where(vovn2 > 1e-6, (ww - 1)/vovn2, 1 + vovn2/2 * (1 + vovn2/3))
        m2 = (ww**6 - 6*ww + 5)/15/vovn2**2
        m3 = (ww**15 - 7*ww**6 + 27*ww - 21)/315/vovn2**3
        m4 = (5*ww**28 - 44*ww**15 + 182*ww**6 - 572*ww + 429)/45045/vovn2**4

        return np.array([m1, m2, m3, m4])


    @staticmethod
    def avgvar_mvsk(vovn):
        """
        mean and variance of the normalized average variance:
        (1/T) \int_0^T e^{2*vov Z_t - vov^2 Z_t} = \int_0^1 e^{2 vovn Z_s - vovn^2 s} ds
        where vovn = vov*sqrt(T). See p.2 in Choi & Wu (2021).

        Args:
            vovn: vov * sqrt(texp)

        Returns:
            (mean, variance, skewness, ex-kurtosis)

        References
            - McGhee WA (2021) An artificial neural network representation of the SABR stochastic volatility model. Journal of Computational Finance
        """

        vovn2 = vovn**2
        m = MathFuncs.avg_exp(vovn2)  # [exp(vov2)-1]/vov2
        ww = m * vovn2 + 1.  # = exp(vov2)

        m2 = (10 + ww*(6 + ww*(3 + ww))) / 15
        v = vovn2 * m2 * m**3    # * (ww-1)^3
        # If vov2 -> 0, v = (4/3) vovn^2

        coef = np.array([1, 5, 15, 35, 70, 126, 210, 330, 432, 456, 336])/315  # O(x^10)
        s = vovn * np.sqrt(m) * np.polyval(coef, ww) / np.power(m2, 1.5)  # skewness
        # If vov2 -> 0, s = (4/5)*3*sqrt(3)*vov = 2.4*sqrt(3) ~ 4.15692*vov

        # Ex-kurtosis calculation
        # Factor[-3(-1 + w) ^ 4 + (2(-1 + w) ^ 2(5 - 6 w + w ^ 6))/5 - (4(-1 + w)(-21 + 27 w - 7 w ^ 6 + w ^ 15))/315 + (
        #            429 - 572 w + 182 w ^ 6 - 44 w ^ 15 + 5 w ^ 28)/45045 - 3(-(-1 + w) ^ 2 + (5 - 6 w + w ^ 6)/15) ^ 2]
       #((w - 1) ^ 7(25
        # w ^ 21 + 175 w ^ 20 + 700 w ^ 19 + 2100 w ^ 18 + 5250 w ^ 17 + 11550 w ^ 16 + 23100 w ^ 15 + 42900 w ^ 14 + 75075 w ^ 13 + 125125 w ^ 12 + 200200 w ^ 11 + 309400 w ^ 10 + 461240 w ^ 9 + 660920 w ^ 8 + 907400 w ^ 7 + 1190280 w ^ 6 + 1483482 w ^ 5 + 1735734 w ^ 4 + 1857856 w ^ 3 + 1706848 w ^ 2 + 1246960 w + 583440))/225225

        coef = np.array([25, 175, 700, 2100, 5250, 11550, 23100, 42900, 75075, 125125, 200200,
                         309400, 461240, 660920, 907400, 1190280, 1483482, 1735734, 1857856,
                         1706848, 1246960, 583440]) / 225225  # O(x^22)
        k = vovn2 * m * np.polyval(coef, ww) / m2**2
        # If vov2 -> 0, k = 368/35*3 vovn^2 ~ 31.542857 vovn^2

        return m, v, s, k


class SabrVolApproxABC(SabrABC):
    """
    Abstract class for SABR models with volatility approximation (asymptotic expansion)
    """

    approx_order = 1  # order in texp: 0: leading order, 1: first order, -1: reserved for 2/(1+exp(-2*eps)) smoothing

    @abc.abstractmethod
    def vol_for_price(self, strike, spot, texp):
        """
        Equivalent volatility of the SABR model

        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry

        Returns:
            equivalent volatility
        """
        return NotImplementedError

    def price(self, strike, spot, texp, cp=1):
        vol = self.vol_for_price(strike, spot, texp)
        m_vol = self.base_model(vol)
        price = m_vol.price(strike, spot, texp, cp=cp)
        return price

    def impvol(self, price, strike, spot, texp, cp=1, setval=False):
        model = copy.copy(self)

        vol = self.base_model(None).impvol(price, strike, spot, texp, cp=cp)

        def iv_func(_sigma):
            model.sigma = _sigma
            return model.vol_for_price(strike, spot, texp) - vol

        sigma = spop.brentq(iv_func, 0, 10)

        if setval:
            self.sigma = sigma
        return sigma

    def vol_smile(self, strike, spot, texp, cp=None, model=None):
        if model is None:
            model = "norm" if np.isclose(self.beta, 0) else "bsm"

        vol_beta = self.beta if self._base_beta is None else self._base_beta

        if (model.lower() == "bsm" and np.isclose(vol_beta, 1)) \
                or (model.lower() == "norm" and np.isclose(vol_beta, 0)):
            vol = self.vol_for_price(strike, spot, texp)
        else:
            vol = super().vol_smile(strike, spot, texp, cp=cp, model=model)
        return vol


class SabrHagan2002(SabrVolApproxABC):
    """
    SABR model with Hagan's implied volatility approximation for 0<beta<=1.

    References:
        Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing Smile Risk.
        Wilmott, September, 84–108.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.SabrHagan2002(sigma=2, vov=0.2, rho=-0.3, beta=0.5)
        >>> m.vol_for_price(np.arange(80, 121, 10), 100, 1.2)
        array([0.21976016, 0.20922027, 0.200432  , 0.19311113, 0.18703486])
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([22.04862858, 14.56226187,  8.74170415,  4.72352155,  2.28891776])
    """

    _base_beta = 1.0  # should not be changed

    def vol_for_price(self, strike, spot, texp):
        # fwd, spot, sigma may be either scalar or np.array.
        # texp, vov, rho, beta should be scholar values

        if texp <= 0.0:
            return 0.0
        fwd, _, _ = self._fwd_factor(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(spot, texp)
        betac2 = betac**2

        log_kk = np.log(fwd / strike)
        log_kk2 = log_kk * log_kk
        pow_kk = np.power(strike / fwd, betac / 2)

        pre1 = pow_kk * (1 + betac2 / 24 * log_kk2 * (1 + betac2 / 80 * log_kk2))

        term02 = (2 - 3 * rho2) / 24 * self.vov**2
        term11 = alpha * self.vov * self.rho * self.beta / 4 / pow_kk
        term20 = (betac * alpha / pow_kk)**2 / 24

        if self.approx_order == 0:
            vol = 1.0
        else:
            vol = 1.0 + texp * (term02 + term11 + term20)

        zz = pow_kk * log_kk * self.vov / np.maximum(alpha, np.finfo(float).eps)
        hh = self._hh(-zz, self.rho)  # note we pass -zz becaues hh(zz) definition is different
        vol *= alpha * hh / pre1  # bsm vol
        return vol

    def calibrate3(self, price_or_vol3, strike3, spot, texp, cp=1, setval=False, is_vol=True):
        """
        Given option prices or implied vols at 3 strikes, compute the sigma, vov, rho to fit the data using `scipy.optimize.root`.
        If prices are given (is_vol=False) convert the prices to vol first.

        Args:
            price_or_vol3: 3 prices or 3 volatilities (depending on `is_vol`)
            strike3: 3 strike prices
            spot: spot price
            texp: time to expiry
            cp: cp
            setval: if True, set sigma, vov, rho values
            is_vol: if True, `price_or_vol3` are volatilities.

        Returns:
            Dictionary of `sigma`, `vov`, and `rho`.
        """
        model = copy.copy(self)

        if is_vol:
            vol3 = price_or_vol3
        else:
            vol3 = self.base_model(None).impvol(price_or_vol3, strike3, spot, texp, cp=cp)

        def iv_func(x):
            model.sigma = np.exp(x[0])
            model.vov = np.exp(x[1])
            model.rho = np.tanh(x[2])
            err = model.vol_for_price(strike3, spot, texp=texp) - vol3
            return err

        sol = spop.root(iv_func, np.array([np.log(vol3[1]), -1, 0.0]))
        params = {"sigma": np.exp(sol.x[0]), "vov": np.exp(sol.x[1]), "rho": np.tanh(sol.x[2])}

        if setval:
            self.sigma, self.vov, self.rho = params["sigma"], params["vov"], params["rho"]

        return params


class SabrNormVolApprox(SabrVolApproxABC):
    """
    Noram SABR model (beta=0) with normal volatility approximation.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.SabrNormVolApprox(sigma=20, vov=0.8, rho=-0.3)
        >>> m.vol_for_price(np.arange(80, 121, 10), 100, 1.2)
        array([24.97568842, 22.78062691, 21.1072    , 20.38569729, 20.78963436])
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([23.70791426, 15.74437409,  9.22425529,  4.78754361,  2.38004685])
    """

    _base_beta = 0  # should not be changed
    is_atmvol = False

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
            print(f"Ignoring beta = {beta}...")
        self.is_atmvol = is_atmvol
        super().__init__(sigma, vov, rho, beta=0, intr=intr, divr=divr, is_fwd=is_fwd)

    def price_vsk(self, texp=1):
        """
        Variance, skewness, and ex-kurtosis of forward price. Eq. (21) in Choi et al. (2019)
        It is a special case (lambda=0) of NsvhABC.price_vsk()

        Args:
            texp: time to expiry

        Returns:
            (variance, skewness, and ex-kurtosis)

        References:
            - Choi J, Liu C, Seo BK (2019) Hyperbolic normal stochastic volatility model. J Futures Mark 39:186–204. https://doi.org/10.1002/fut.21967
        """
        vovn = self.vov * np.sqrt(texp)
        m2 = np.expm1(vovn**2)
        ww = m2 + 1

        skew = self.rho * np.sqrt(m2) * (ww+2)
        exkurt = m2 * ((4*self.rho**2 + 1)/5 * (ww*(ww*(ww + 3) + 6) + 5) + 1)

        return m2*(self.sigma/self.vov)**2, skew, exkurt

    def vol_for_price(self, strike, spot, texp):
        fwd = self.forward(spot, texp)

        if self.approx_order == 0 or self.is_atmvol:
            vol = 1.0
        else:
            vol = 1.0 + texp * (2 - 3 * self.rho**2) / 24 * self.vov**2

        zz = self.vov / self.sigma * (strike - fwd)
        hh = self._hh(zz, self.rho)
        vol *= self.sigma * hh
        return vol


class SabrChoiWu2021H(SabrVolApproxABC, smile.MassZeroABC):
    """
    The CEV volatility approximation of the SABR model based on Theorem 1 of Choi & Wu (2019)

    References:
        - Choi, J., & Wu, L. (2019). The equivalent constant-elasticity-of-variance (CEV) volatility of the stochastic-alpha-beta-rho (SABR) model. ArXiv:1911.13123 [q-Fin]. https://arxiv.org/abs/1911.13123

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.SabrChoiWu2021H(sigma=2, vov=0.2, rho=-0.3, beta=0.5)
        >>> m.vol_for_price(np.arange(80, 121, 10), 100, 1.2)
        array([2.07833214, 2.03698255, 2.00332   , 1.97692259, 1.95735019])
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([22.04897988, 14.56240351,  8.74169054,  4.72340753,  2.28876105])
    """

    def vol_for_price(self, strike, spot, texp):
        # fwd, spot, sigma may be either scalar or np.array.
        # texp, vov, rho, beta should be scholar values

        vol_beta = self.beta if self._base_beta is None else self._base_beta
        vol_betac = 1.0 - vol_beta

        fwd, _, _ = self._fwd_factor(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)

        kk = strike / fwd  # standardized strike

        vov_over_alpha_safe = self.vov / np.maximum(alpha, np.finfo(float).eps)
        tmp = MathFuncs.avg_pow(kk - 1.0, -self.beta)
        qq_ratio = 1.0 if self._base_beta is None else MathFuncs.avg_pow(kk - 1.0, -vol_beta) / tmp

        qq = tmp * (kk - 1.0)
        zz = vov_over_alpha_safe * qq  # zeta = (vov/sigma0) q
        hh = self._hh(zz, self.rho)
        # term02: O(vov^2)
        term02 = (2 - 3 * rho2) / 24 * self.vov**2

        # term11: O(alpha*vov)
        # C(k)-C(1)/(k-1). Notice that 1/beta comes from int_inv_locvol
        term11 = self.rho * self.beta / 4 * self.vov * alpha * MathFuncs.avg_pow(kk - 1.0, -betac)

        # term20: O(alpha^2)
        if np.isclose(self.beta, vol_beta):
            term20 = 0.0
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ## Override ATM (qq=0)
                term20 = np.where(
                    np.fabs(qq) < 1e-6,
                    (betac**2 - vol_betac**2) / 24 * alpha**2,
                    (0.5*(self.beta - self._base_beta) * np.log(kk) - np.log(qq_ratio)) * alpha**2 / qq,
                )
        # else:
        #    raise ValueError('Cannot handle this vol_beta different from beta')

        ## Return
        if self.approx_order == 0:
            vol = 1.0
        else:
            vol = 1.0 + texp * (term20 + term11 + term02)

        pre_fac = alpha * np.power(fwd, vol_betac) * qq_ratio
        vol *= pre_fac * hh
        return vol

    def mass_zero(self, spot, texp, log=False):
        assert self._base_beta is None
        vol_cev = self.vol_for_price(0.0, spot, texp)
        cev_m = cev.Cev(sigma=vol_cev, beta=self.beta)
        mass = cev_m.mass_zero(spot, texp, log=log)
        return mass

    def mass_zero_t0(self, spot, texp):
        """
        Limit value of -T log(M_T) as T -> 0, where M_T is the mass at zero. See Corollary 3.1 of Choi & Wu (2019)

        Args:
            spot: spot (or forward) price
            texp: time to expiry

        Returns:
            -lim_{T->0} T log(M_T)
        """
        fwd = self.forward(spot, texp)
        alpha, betac, rhoc, rho2, _ = self._variables(fwd, texp)
        hh = self._hh(-self.vov / (alpha * betac), self.rho)
        t0 = 0.5 / (betac * alpha * hh)**2
        return t0


class SabrChoiWu2021P(SabrChoiWu2021H, smile.MassZeroABC):
    """
    The CEV volatility approximation of the SABR modelbased on Theorem 2 of Choi & Wu (2019)

    References:
        - Choi, J., & Wu, L. (2019). The equivalent constant-elasticity-of-variance (CEV) volatility of the stochastic-alpha-beta-rho (SABR) model. ArXiv:1911.13123 [q-Fin]. https://arxiv.org/abs/1911.13123

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.SabrChoiWu2021P(sigma=2, vov=0.2, rho=-0.3, beta=0.5)
        >>> m.vol_for_price(np.arange(80, 121, 10), 100, 1.2)
        array([2.07761123, 2.03665311, 2.00332   , 1.97718783, 1.95781579])
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([22.0470526 , 14.56114825,  8.74169054,  4.72447547,  2.29018838])
    """

    def vol_for_price(self, strike, spot, texp):
        # fwd, spot, sigma may be either scalar or np.array.
        # texp, vov, rho, beta should be scholar values

        vol_beta = self.beta if self._base_beta is None else self._base_beta
        vol_betac = 1.0 - vol_beta

        fwd, _, _ = self._fwd_factor(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)

        kk = strike / fwd  # standardized strike

        # explicitly make np.array even if args are all scalar or list
        if isinstance(kk, float):
            kk = np.array([kk])

        ## Eq 32 (leading order)
        vov_over_alpha_safe = self.vov / np.maximum(alpha, np.finfo(float).eps)

        # qq_ratio = qq_vol_beta / qq_beta
        tmp = MathFuncs.avg_pow(kk - 1.0, -self.beta)
        qq_ratio = 1.0 if self._base_beta is None else MathFuncs.avg_pow(kk - 1.0, -vol_beta) / tmp

        zz = vov_over_alpha_safe * tmp * (kk - 1.0)  # zeta = (vov/sigma0) q
        hh = self._hh(zz, self.rho)
        v_m = self._vv(zz, self.rho)

        if abs(betac) < 0.001:
            gg_diff = 0.5 * self.beta * self.rho / (1.0 - rho2) * (v_m - 1.0 - self.rho * zz)
        else:
            t1 = (v_m + self.rho + zz) / rhoc  # array
            t2 = (1 + self.rho) / rhoc  # scalar

            eta = self.vov / (betac * alpha) * np.power(kk, betac) * (rhoc / v_m)
            eta2_m_1 = eta * eta - 1.0
            sqrt_eta2_m_1 = np.sqrt(abs(eta2_m_1))

            num_t1 = self.rho + (eta - rhoc) * t1
            num_t2 = self.rho + (eta - rhoc) * t2

            gg1 = np.arctan(t1)
            gg2 = np.arctan(t2) * np.ones_like(t1)  # t2 is a scalar, so vectorize

            eps = 1e-6

            ind = eta2_m_1 > eps
            gg2[ind] += (
                eta[ind]
                / sqrt_eta2_m_1[ind]
                * np.arctan2(sqrt_eta2_m_1[ind], num_t2[ind])
            )
            gg1[ind] += (
                eta[ind]
                / sqrt_eta2_m_1[ind]
                * np.arctan2(sqrt_eta2_m_1[ind], num_t1[ind])
            )

            ind = eta2_m_1 < -eps
            gg1[ind] += (
                eta[ind]
                / sqrt_eta2_m_1[ind]
                * 0.5
                * np.log(
                    abs(
                        (num_t1[ind] + sqrt_eta2_m_1[ind])
                        / (num_t1[ind] - sqrt_eta2_m_1[ind])
                    )
                )
            )

            # as eta->0 (k->0) the numerator diverge. but the gg2 value should be 0.
            ind = (eta2_m_1 < -eps) * (eta > eps)
            gg2[ind] += (
                eta[ind]
                / sqrt_eta2_m_1[ind]
                * 0.5
                * np.log(
                    abs(
                        (num_t2[ind] + sqrt_eta2_m_1[ind])
                        / (num_t2[ind] - sqrt_eta2_m_1[ind])
                    )
                )
            )
            # when eta is very small, the term above is zero, so do nothing.

            ind = abs(eta2_m_1) <= eps
            gg1[ind] += (
                eta[ind] / num_t1[ind] * (1 - eta2_m_1[ind] / num_t1[ind]**2 / 3)
            )
            gg2[ind] += (
                eta[ind] / num_t2[ind] * (1 - eta2_m_1[ind] / num_t2[ind]**2 / 3)
            )

            gg_diff = self.rho * self.beta / (rhoc * betac) * (gg2 - gg1)

        zz2_safe = np.maximum(zz**2, np.finfo(float).eps)

        if np.isclose(self.beta, vol_beta):
            tmp = 0.0
        else:
            tmp = 0.5 * (self.beta - self._base_beta) * np.log(kk) - np.log(qq_ratio)

        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ## Override ATM (qq=0)
            term20 = np.where(
                np.fabs(qq) < 1e-6,
                (np.square(betac) - np.square(vol_betac)) / 24 * np.square(alpha),
                (0.5 * (self.beta - self._base_beta) * np.log(kk) - np.log(qq_ratio)) * np.square(alpha) / qq
            )
        """
        order1 = (self.vov * hh)**2 / zz2_safe * (tmp + 0.5 * np.log(v_m / np.square(hh)) + gg_diff)

        ## Override ATM (z=0)
        ind_atm = abs(zz) < 1e-6
        order1[ind_atm] = (betac**2 - vol_betac**2)/24 * alpha**2 \
                          + ((self.rho * self.beta / 4) * alpha + (2 - 3 * rho2) / 24 * self.vov) * self.vov  # RHS scalar

        # return value
        if self.approx_order == 0:
            vol = 1.0
        else:
            vol = 1.0 + order1 * texp

        # order0 = (z_base' / z_base) * hh
        vol *= alpha * np.power(fwd, vol_betac) * qq_ratio * hh
        return vol[0] if vol.size == 1 else vol


class SabrLorig2017(SabrVolApproxABC):
    """
    Third-order BSM volatilty approximation of the SABR model by Lorig et al. (2017)

    References:
        Lorig, M., Pagliarani, S., & Pascucci, A. (2017). Explicit Implied Volatilities
        for Multifactor Local-Stochastic Volatility Models.
        Mathematical Finance, 27(3), 926–960. https://doi.org/10.1111/mafi.12105
    """

    _base_beta = 1.0  # should not be changed
    lv_factor = 1.0
    approx_order = 3

    def vol_for_price(self, strike, spot, texp):
        # fwd, spot, sigma may be either scalar or np.array.
        # texp, vov, rho, beta should be scholar values

        fwd, _, _ = self._fwd_factor(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)

        kk = strike / fwd  # standardized strike

        kmx = np.log(kk)
        kmx_2 = kmx * kmx
        kmx_3 = kmx * kmx_2

        t_2 = texp * texp if self.approx_order > 1 else 0.0
        t_3 = texp * t_2 if self.approx_order > 2 else 0.0

        betac_2 = betac * betac
        betac_3 = betac * betac_2

        rho = self.rho
        vov = self.vov
        vov_2 = vov * vov
        vov_3 = vov * vov_2

        s0 = alpha
        s0_2 = s0 * s0
        s0_3 = s0 * s0_2
        s0_4 = s0 * s0_3
        s0_5 = s0 * s0_4

        s10 = s0 / 2 * (-betac) * kmx
        s01 = vov / 4 * (2 * kmx * rho + texp * s0 * (-vov + rho * s0))

        s20 = (
            s0
            * betac_2
            * (texp / 24 * s0_2 - t_2 / 96 * s0_4 + kmx_2 / 12)
            * self.lv_factor
        )

        s11 = -betac * vov
        s11 *= (
            texp / 4 * rho * s0_2
            - t_2 / 48 * rho * s0_4
            - texp / 24 * s0 * (3 * vov - 5 * rho * s0) * kmx
        )

        s02 = vov_2
        s02 *= (
            texp / 24 * (8 - 3 * rho2) * s0
            + t_2
            / 96
            * s0
            * (5 * vov_2 + 2 * s0 * ((6 * rho2 - 2) * s0 - 7 * vov * rho))
            - texp / 24 * rho * (vov - 3 * rho * s0) * kmx
            + (2 - 3 * rho2) / (12 * s0) * kmx_2
        )

        s30 = -betac_3 * s0_3 * kmx
        s30 *= texp / 16 - 5 / 192 * t_2 * s0_2

        s21 = betac_2 * vov
        s21 *= (
            t_2 / 96 * s0_3 * (7 * rho * s0 - 3 * vov)
            + t_3 / 384 * s0_5 * (5 * vov - 7 * rho * s0)
            + 13 / 48 * texp * rho * s0_2 * kmx
            - 13 / 192 * t_2 * rho * s0_4 * kmx
            - texp / 48 * s0 * (vov - 3 * rho * s0) * kmx_2
        )

        s12 = -betac * vov_2
        s12 *= (
            t_2 / 48 * rho * s0_2 * (13 * rho * s0 - 7 * vov)
            + t_3 / 192 * rho * s0_4 * (5 * vov - 7 * rho * s0)
            + texp / 48 * (3 * rho2 + 8) * s0 * kmx
            + t_2
            / 192
            * s0
            * (5 * vov_2 - 22 * vov * rho * s0 + 4 * (5 * rho2 - 3) * s0_2)
            * kmx
            + texp / 24 * rho2 * s0 * kmx_2
            + (3 * rho2 - 2) / (24 * s0) * kmx_3
        )

        s03 = vov_3
        s03 *= (
            t_2 / 96 * s0 * (3 * vov * (rho2 - 4) + rho * (26 - 9 * rho2) * s0)
            + t_3
            / 384
            * s0
            * (
                s0
                * (
                    19 * vov_2 * rho
                    + 2 * s0 * (vov * (8 - 21 * rho2) + rho * (15 * rho2 - 11) * s0)
                )
                - 3 * vov_3
            )
            + texp / 48 * rho * (3 * rho2 - 2) * kmx
            + t_2
            / 192
            * rho
            * (vov_2 + 6 * s0 * (vov * rho + (1 - 2 * rho2) * s0))
            * kmx
            - texp / 16 * rho * (rho2 - 1) * kmx_2
            + rho * (6 * rho2 - 5) / (24 * s0_2) * kmx_3
        )

        vol = s0 + (s10 + s01) + (s20 + s11 + s02) + (s30 + s21 + s12 + s03)

        return vol
