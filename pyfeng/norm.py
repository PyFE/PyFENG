# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:56:58 2017
@author: jaehyuk
"""

import numpy as np
import scipy.stats as spst

from . import opt_abc as opt
from . import bsm


class Norm(opt.OptAnalyticABC):
    """
    Bachelier (normal) model for option pricing.
    Underlying price is assumed to follow arithmetic Brownian motion.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.Norm(sigma=20, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([16.57233446, 10.34711401,  5.77827026,  2.83857367,  1.20910477])
        >>> sigma = np.array([20, 30, 50])[:, None]
        >>> m = pf.Norm(sigma, intr=0.05, divr=0.1) # sigma in axis=0
        >>> m.price(np.array([90, 100, 110]), 100, 1.2, cp=np.array([-1,1,1]))
        array([[ 6.41387836,  5.77827026,  2.83857367],
               [10.48003559,  9.79822867,  6.3002881 ],
               [18.67164469, 17.95246828, 13.98027179]])
    """

    # Coefficients for _impvol_Choi2009
    _POLY_NU = [
        1.266458051348246e4,
        2.493415285349361e4,
        6.106322407867059e3,
        1.848489695437094e3,
        5.988761102690991e2,
        4.980340217855084e1,
        2.100960795068497e1,
        3.994961687345134e-1,
    ]

    _POLY_DE = [
        1.174240599306013e1,
        -2.067719486400926e2,
        3.608817108375034e3,
        2.392008891720782e4,
        1.598919697679745e4,
        1.323614537899738e3,
        1.495105008310999e3,
        3.093573936743112e1,
        4.990534153589422e1,
        1.0,
    ]

    # option value below the intrinsic value by IMPVOL_TOL is considered to be numerical error.
    # volatility will be set to 0
    IMPVOL_TOL = 1000 * np.finfo(float).eps

    @staticmethod
    def price_formula(
        strike, spot, sigma, texp, cp=1, intr=0.0, divr=0.0, is_fwd=False
    ):
        """
        Bachelier model call/put option pricing formula (static method)

        Args:
            strike: strike price
            spot: spot (or forward)
            sigma: model volatility
            texp: time to expiry
            cp: 1/-1 for call/put option
            sigma: model volatility
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.

        Returns:
            Vanilla option price
        """
        df = np.exp(-texp * intr)
        fwd = np.array(spot) * (1.0 if is_fwd else np.exp(-texp * divr) / df)

        sigma_std = np.maximum(np.array(sigma) * np.sqrt(texp), np.finfo(float).eps)
        d = (fwd - strike) / sigma_std

        cp = np.array(cp)
        price = df * (
            cp * (fwd - strike) * spst.norm.cdf(cp * d) + sigma_std * spst.norm.pdf(d)
        )
        return price

    def _impvol_Choi2009(self, price, strike, spot, texp, cp=1, setval=False):
        """
        Bachelier implied volatility by Choi et al. (2007)

        Args:
            price: option price
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put
            setval: if True, sigma is set with the solved implied volatility

        References:
            Choi, J., Kim, K., & Kwak, M. (2009). Numerical Approximation of the Implied
            Volatility Under Arithmetic Brownian Motion. Applied Mathematical Finance,
            16(3), 261–268. https://doi.org/10.1080/13504860802583436

        Returns:
            implied volatility
        """
        fwd, df, _ = self._fwd_factor(spot, texp)
        price_fwd = np.array(price) / df
        strike_std = np.array(cp) * (fwd - strike)

        time_val = price_fwd - np.maximum(0, strike_std)  # option time value
        strd = 2 * price_fwd - strike_std  # straddle value (=call + put)
        # Note: time_val > 0  => strd >= time_val > 0
        # v = |fwd - strike|/(call + put) = np.fabs(strike_std) / strd, ATM when v=0
        # v1 = 1 - v, ATM when v1=1
        # Use v1 instead of v to preserve the option value < machine epsilon
        v1 = np.clip(
            2 * time_val / strd, 0, 1
        )  # bound between 0 and 1 for now. Out-of-bound value will be handleded later
        v_sq = (1 - v1) ** 2

        # ignore 'divide by 0' error when v1 = 0 for now.
        # eta = v / atanh(v) = 2v / log((1+v)/(1-v)) = 2v / log((2-v1)/v1)
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(
                v1 < 0.999,
                2 * (1 - v1) / (np.log((2.0 - v1) / v1)),
                1 / (1 + v_sq * (1 / 3 + v_sq / 5)),
            )
        h_a = (
            np.sqrt(eta)
            * np.polyval(Norm._POLY_NU, eta)
            / np.polyval(Norm._POLY_DE, eta)
        )
        # sigma = sqrt(pi/2T) * (call + put) * h_a
        _sigma = np.where(
            time_val >= -self.IMPVOL_TOL,
            np.sqrt(np.pi / (2 * texp)) * strd * h_a,
            np.nan,
        )

        if setval:
            self.sigma = _sigma

        return _sigma

    def vega(self, strike, spot, texp, cp=1):

        fwd, df, _ = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).eps)
        d = (fwd - strike) / sigma_std

        vega = (
            df * spst.norm.pdf(d) * np.sqrt(texp)
        )  # formula according to lecture notes
        return vega

    def delta(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).eps)
        d = (fwd - strike) / sigma_std

        delta = cp * spst.norm.cdf(cp * d)  # formula according to wikipedia
        delta *= df if self.is_fwd else divf
        return delta

    def cdf(self, strike, spot, texp, cp=1):

        fwd = self.forward(spot, texp)
        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).eps)
        d = (fwd - strike) / sigma_std
        cdf = spst.norm.cdf(cp * d)  # formula according to wikipedia
        return cdf

    def gamma(self, strike, spot, texp, cp=1):

        # cp is not used
        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).eps)
        d = (fwd - strike) / sigma_std

        gamma = df * spst.norm.pdf(d) / sigma_std  # formula according to wikipedia
        if not self.is_fwd:
            gamma *= (divf / df) ** 2
        return gamma

    def theta(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).eps)
        d = (fwd - strike) / sigma_std

        # still not perfect; need to consider the derivative w.r.t. divr and is_fwd = True
        theta = sigma_std * spst.norm.pdf(d)
        theta = -0.5 * theta / texp + self.intr * (
            theta - cp * strike * spst.norm.cdf(cp * d)
        )
        return df * theta

    def price_binary(self, strike, spot, texp, cp=1, opt_type="cash"):
        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).eps)
        d = (fwd - strike) / sigma_std

        if opt_type.lower() == "asset":
            price = df * (
                cp * fwd * spst.norm.cdf(cp * d) + sigma_std * spst.norm.pdf(d)
            )
        else:
            price = df * spst.norm.cdf(cp * d)

        return price

    ####
    impvol = _impvol_Choi2009

    def vol_smile(self, strike, spot, texp, cp=1, model="bsm"):
        """
        Equivalent volatility smile for a given model

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
            model: {'bsm' (default), 'bsm', 'bsm-approx', 'norm'}

        Returns:
            volatility smile under the specified model
        """
        if model.lower() == "norm":
            return self.sigma * np.ones_like(strike + spot + texp + cp)
        if model.lower() == "bsm":
            price = self.price(strike, spot, texp, cp=cp)
            return bsm.Bsm(None).impvol(price, strike, spot, texp, cp=cp)
        elif model.lower() == "bsm-approx":
            fwd, _, _ = self._fwd_factor(spot, texp)
            sigma_std = self.sigma / fwd
            kk = strike / fwd
            lnk = np.log(kk)
            vol = sigma_std / np.sqrt(kk)
            vol *= (1 + vol ** 2 * texp / 24) / (1 + lnk ** 2 / 24)
            return vol
        else:
            raise ValueError(f"Unknown model: {model}")

    def _price_suboptimal(self, strike, spot, texp, cp=1, strike2=None):
        fwd, df, _ = self._fwd_factor(spot, texp)
        strike2 = strike if strike2 is None else strike2

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).eps)
        d2 = (fwd - strike2) / sigma_std

        price = cp * (fwd - strike) * spst.norm.cdf(
            cp * d2
        ) + sigma_std * spst.norm.pdf(d2)
        price *= df
        return price

    def _barrier_params(self, barrier, spot):
        """
        Parameters used for barrier option pricing

        Args:
            barrier: barrier price
            spot: spot price

        Returns:
            barrier option pricing parameters (psi, spot_mirror)
        """
        psi = 1  # need to fix this
        spot_mirror = 2 * barrier - spot
        return psi, spot_mirror

    """
    Inherit price_barrier method from BsmModel. The only change is from `barrier_params`
    """
    price_barrier = bsm.Bsm.price_barrier
