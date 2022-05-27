import abc

import scipy.stats as spst
import warnings
import numpy as np
from itertools import product, combinations
from . import bsm
from . import norm
from . import gamma
from . import nsvh
import pyfeng.opt_abc as opt
from pyfeng.quad import NdGHQ  # Not sure


class OptMaABC(opt.OptABC, abc.ABC):

    n_asset = 1
    rho = None
    cor_m = np.diag([1.0])
    cov_m = np.diag([1.0])
    chol_m = np.diag([1.0])

    def __init__(self, sigma, cor=None, intr=0.0, divr=0.0, is_fwd=False):
        """

        Args:
            sigma: model volatilities of `n_asset` assets. (n_asset, ) array
            cor: correlation. If matrix with shape (n_asset, n_asset), used as it is.
                If scalar, correlation matrix is constructed with all same off-diagonal values.
            intr: interest rate (domestic interest rate)
            divr: vector of dividend/convenience yield (foreign interest rate) 0-D or (n_asset, ) array
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        sigma = np.atleast_1d(sigma)
        self.n_asset = len(sigma)

        super().__init__(sigma, intr, divr, is_fwd)

        if self.n_asset == 1:
            if cor is not None:
                print(f"Ignoring cor={cor} for a single asset")
            self.rho = None
            self.cor_m = np.array([[1.0]])
        elif np.isscalar(cor):
            self.cor_m = cor * np.ones((self.n_asset, self.n_asset)) + (
                1.0 - cor
            ) * np.eye(self.n_asset)
            self.rho = cor
        else:
            assert cor.shape == (self.n_asset, self.n_asset)
            self.cor_m = cor
            if self.n_asset == 2:
                self.rho = cor[0, 1]

        self.cov_m = sigma * self.cor_m * sigma[:, None]
        self.chol_m = np.linalg.cholesky(self.cov_m)

    def price(self, strike, spot, texp, cp=1):
        """
        Call/put option price.

        Args:
            strike: strike price.
            spot: spot (or forward) prices for assets.
                Asset dimension should be the last, e.g. (n_asset, ) or (N, n_asset)
            texp: time to expiry.
            cp: 1/-1 for call/put option.

        Returns:
            option price
        """
        return NotImplementedError


class BsmSpreadKirk(OptMaABC):
    """
    Kirk's approximation for spread option.

    References:
        - Kirk E (1995) Correlation in the energy markets. In: Managing Energy Price Risk, First. Risk Publications, London, pp 71–78

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmSpreadKirk((0.2, 0.3), cor=-0.5)
        >>> m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        array([22.15632247, 17.18441817, 12.98974214,  9.64141666,  6.99942072])
    """

    weight = np.array([1, -1])

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        assert fwd.shape[-1] == self.n_asset

        fwd1 = fwd[..., 0] - np.minimum(strike, 0)
        fwd2 = fwd[..., 1] + np.maximum(strike, 0)

        sig1 = self.sigma[0] * fwd[..., 0] / fwd1
        sig2 = self.sigma[1] * fwd[..., 1] / fwd2
        sig_spd = np.sqrt(sig1 * (sig1 - 2.0 * self.rho * sig2) + sig2 ** 2)
        price = bsm.Bsm.price_formula(fwd2, fwd1, sig_spd, texp, cp=cp, is_fwd=True)
        return df * price


class BsmSpreadBjerksund2014(OptMaABC):
    """
    Bjerksund & Stensland (2014)'s approximation for spread option.

    References:
        - Bjerksund P, Stensland G (2014) Closed form spread option valuation. Quantitative Finance 14:1785–1794. https://doi.org/10.1080/14697688.2011.617775

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmSpreadBjerksund2014((0.2, 0.3), cor=-0.5)
        >>> m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        array([22.13172022, 17.18304247, 12.98974214,  9.54431944,  6.80612597])
    """

    weight = np.array([1, -1])

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        assert fwd.shape[-1] == self.n_asset

        fwd1 = fwd[..., 0]
        fwd2 = fwd[..., 1]

        std11 = self.sigma[0] ** 2 * texp
        std12 = self.sigma[0] * self.sigma[1] * texp
        std22 = self.sigma[1] ** 2 * texp

        aa = fwd2 + strike
        bb = fwd2 / aa
        std = np.sqrt(std11 - 2 * bb * self.rho * std12 + bb ** 2 * std22)

        d3 = np.log(fwd1 / aa)
        d1 = (d3 + 0.5 * std11 - bb * (self.rho * std12 - 0.5 * bb * std22)) / std
        d2 = (d3 - 0.5 * std11 + self.rho * std12 + bb * (0.5 * bb - 1) * std22) / std
        d3 = (d3 - 0.5 * std11 + 0.5 * bb ** 2 * std22) / std

        price = cp * (
            fwd1 * spst.norm.cdf(cp * d1)
            - fwd2 * spst.norm.cdf(cp * d2)
            - strike * spst.norm.cdf(cp * d3)
        )

        return df * price


class NormBasket(OptMaABC):
    """
    Basket option pricing under the multiasset Bachelier model
    """

    weight = None

    def __init__(self, sigma, cor=None, weight=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Initialize an instance for basket option.

        Args:
            sigma: model volatilities of `n_asset` assets. (n_asset, ) array
            cor: correlation. If matrix, used as it is. (n_asset, n_asset)
                If scalar, correlation matrix is constructed with all same off-diagonal values.
            weight: asset weights, If None, equally weighted as 1/n_asset
                If scalar, equal weights of the value
                If 1-D array, uses as it is. (n_asset, )
            intr: interest rate (domestic interest rate)
            divr: vector of dividend/convenience yield (foreign interest rate) 0-D or (n_asset, ) array
            is_fwd: if True, treat `spot` as forward price. False by default.

        See Also:
            init_spread()
        """

        super().__init__(sigma, cor=cor, intr=intr, divr=divr, is_fwd=is_fwd)
        if weight is None:
            self.weight = np.ones(self.n_asset) / self.n_asset
        elif np.isscalar(weight):
            self.weight = np.ones(self.n_asset) * weight
        else:
            assert len(weight) == self.n_asset
            self.weight = np.array(weight)

    @classmethod
    def init_spread(cls, sigma, cor=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Initalize an instance for spread option pricing.
        This is a special case of the initalization with weight = (1, -1)

        Examples:
            >>> import numpy as np
            >>> import pyfeng as pf
            >>> m = pf.NormSpread.init_spread((20, 30), cor=-0.5, intr=0.05)
            >>> m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
            array([17.95676186, 13.74646821, 10.26669936,  7.47098719,  5.29057157])
        """

        return cls(sigma, cor=cor, weight=np.array([1, -1]), intr=intr, divr=divr, is_fwd=is_fwd)

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd @ self.weight
        vol_basket = np.sqrt(self.weight @ self.cov_m @ self.weight)

        price = norm.Norm.price_formula(
            strike, fwd_basket, vol_basket, texp, cp=cp, is_fwd=True
        )
        return df * price


class BsmBasketLevy1992(NormBasket):
    """
    Basket option pricing with the log-normal approximation of Levy & Turnbull (1992)

    References:
        - Levy E, Turnbull S (1992) Average intelligence. Risk 1992:53–57
        - Krekel M, de Kock J, Korn R, Man T-K (2004) An analysis of pricing methods for basket options. Wilmott Magazine 2004:82–89

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.arange(50, 151, 10)
        >>> m = pf.BsmBasketLevy1992(sigma=0.4*np.ones(4), cor=0.5)
        >>> m.price(strike, spot=100*np.ones(4), texp=5)
        array([54.34281026, 47.521086  , 41.56701301, 36.3982413 , 31.92312156,
               28.05196621, 24.70229571, 21.800801  , 19.28360474, 17.09570196,
               15.19005654])
    """

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd * self.weight
        m1 = np.sum(fwd_basket, axis=-1)
        m2 = np.sum(fwd_basket @ np.exp(self.cov_m * texp) * fwd_basket, axis=-1)

        sig = np.sqrt(np.log(m2 / (m1 ** 2)) / texp)
        price = bsm.Bsm.price_formula(strike, m1, sig, texp, cp=cp, is_fwd=True)
        return df * price


class BsmBasketMilevsky1998(NormBasket):
    """
    Basket option pricing with the inverse gamma distribution of Milevsky & Posner (1998)

    References:
        - Milevsky MA, Posner SE (1998) A Closed-Form Approximation for Valuing Basket Options. The Journal of Derivatives 5:54–61. https://doi.org/10.3905/jod.1998.408005
        - Krekel M, de Kock J, Korn R, Man T-K (2004) An analysis of pricing methods for basket options. Wilmott Magazine 2004:82–89

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.arange(50, 151, 10)
        >>> m = pf.BsmBasketMilevsky1998(sigma=0.4*np.ones(4), cor=0.5)
        >>> m.price(strike, spot=100*np.ones(4), texp=5)
        array([51.93069524, 44.40986   , 38.02596564, 32.67653542, 28.21560931,
               24.49577509, 21.38543199, 18.77356434, 16.56909804, 14.69831445,
               13.10186928])
    """

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd * self.weight
        m1 = np.sum(fwd_basket, axis=-1)
        m2 = np.sum(fwd_basket @ np.exp(self.cov_m * texp) * fwd_basket, axis=-1)

        alpha = 1 / (m2 / m1 ** 2 - 1) + 2
        beta = (alpha - 1) * m1

        price = gamma.InvGam.price_formula(
            strike, m1, texp, alpha, beta, cp=cp, is_fwd=True
        )
        return df * price


class BsmMax2(OptMaABC):
    """
    Option on the max of two assets.
    Payout = max( max(F_1, F_2) - K, 0 ) for all or  max( K - max(F_1, F_2), 0 ) for put option

    References:
        - Rubinstein M (1991) Somewhere Over the Rainbow. Risk 1991:63–66

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmMax2(0.2*np.ones(2), cor=0, divr=0.1, intr=0.05)
        >>> m.price(strike=[90, 100, 110], spot=100*np.ones(2), texp=3)
        array([15.86717049, 11.19568103,  7.71592217])
    """

    m_switch = None

    def __init__(self, sigma, cor=None, weight=None, intr=0.0, divr=0.0, is_fwd=False):
        super().__init__(sigma, cor=cor, intr=intr, divr=divr, is_fwd=is_fwd)
        self.m_switch = BsmSpreadKirk(sigma, cor, is_fwd=True)

    def price(self, strike, spot, texp, cp=1):
        sig = self.sigma
        fwd, df, _ = self._fwd_factor(spot, texp)
        assert fwd.shape[-1] == self.n_asset

        sig_std = sig * np.sqrt(texp)
        spd_rho = np.sqrt(np.dot(sig, sig) - 2 * self.rho * sig[0] * sig[1])
        spd_std = spd_rho * np.sqrt(texp)

        # -x and y as rows
        # supposed to be -log(fwd/strike) but strike is added later
        xx = -np.log(fwd) / sig_std - 0.5 * sig_std

        fwd_ratio = fwd[0] / fwd[1]
        yy = np.log([fwd_ratio, 1 / fwd_ratio]) / spd_std + 0.5 * spd_std

        rho12 = (
            np.array([self.rho * sig[1] - sig[0], self.rho * sig[0] - sig[1]]) / spd_rho
        )

        mu0 = np.zeros(2)
        cor_m1 = rho12[0] + (1 - rho12[0]) * np.eye(2)
        cor_m2 = rho12[1] + (1 - rho12[1]) * np.eye(2)

        strike_isscalar = np.isscalar(strike)
        strike = np.atleast_1d(strike)
        cp = cp * np.ones_like(strike)
        n_strike = len(strike)

        # this is the price of max(S1, S2) = max(S1-S2, 0) + S2
        # Used that Kirk approximation strike = 0 is Margrabe's switch option price
        parity = 0 if np.all(cp > 0) else self.m_switch.price(0, fwd, texp) + fwd[1]
        price = np.zeros_like(strike, float)
        for k in range(n_strike):
            xx_ = xx + np.log(strike[k]) / sig_std
            term1 = fwd[0] * (
                spst.norm.cdf(yy[0])
                - spst.multivariate_normal.cdf(np.array([xx_[0], yy[0]]), mu0, cor_m1)
            )
            term2 = fwd[1] * (
                spst.norm.cdf(yy[1])
                - spst.multivariate_normal.cdf(np.array([xx_[1], yy[1]]), mu0, cor_m2)
            )
            term3 = strike[k] * np.array(
                spst.multivariate_normal.cdf(xx_ + sig_std, mu0, self.cor_m)
            )

            assert term1 + term2 + term3 >= strike[k]
            price[k] = term1 + term2 + term3 - strike[k]

            if cp[k] < 0:
                price[k] += strike[k] - parity
        price *= df

        return price[0] if strike_isscalar else price


class BsmBasket1Bm(opt.OptABC):
    """
    Multiasset BSM model for pricing basket/Spread options when all asset prices are driven by a single Brownian motion (BM).

    """

    def __init__(self, sigma, weight=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatilities of `n_asset` assets. (n_asset, )
            weight: asset weights, If None, equally weighted as 1/n_asset
                If scalar, equal weights of the value
                If 1-D array, uses as it is. (n_asset, )
            intr: interest rate (domestic interest rate)
            divr: vector of dividend/convenience yield (foreign interest rate) 0-D or (n_asset, ) array
            is_fwd: if True, treat `spot` as forward price. False by default.
        """

        sigma = np.atleast_1d(sigma)
        self.n_asset = len(sigma)
        if weight is None:
            self.weight = np.ones(self.n_asset) / self.n_asset
        elif np.isscalar(weight):
            self.weight = np.ones(self.n_asset) * weight
        else:
            assert len(weight) == self.n_asset
            self.weight = np.array(weight)
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)

    @staticmethod
    def root(fac, std, strike):
        """
        Calculate the root x of f(x) = sum(fac * exp(std*x)) - strike = 0 using Newton's method

        Each fac and std should have the same signs so that f(x) is a monotonically increasing function.

        fac: factor to the exponents. (n_asset, ) or (n_strike, n_asset). Asset takes the last dimension.
        std: total standard variance. (n_asset, )
        strike: strike prices. scalar or (n_asset, )
        """

        assert np.all(fac * std >= 0.0)
        log = np.min(fac) > 0  # Basket if log=True, spread if otherwise.
        scalar_output = np.isscalar(np.sum(fac * std, axis=-1) - strike)
        strike = np.atleast_1d(strike)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_k = np.where(strike > 0, np.log(strike), 1)

            # Initial guess with linearlized assmption
            x = (strike - np.sum(fac, axis=-1)) / np.sum(fac * std, axis=-1)
            if log:
                np.fmin(x, np.amin(np.log(strike[:, None] / fac) / std, axis=-1), out=x)
            else:
                np.clip(x, -3, 3, out=x)

        # Test x=-9 and 9 for min/max values.
        y_max = np.exp(9 * std)
        y_min = np.sum(fac / y_max, axis=-1) - strike
        y_max = np.sum(fac * y_max, axis=-1) - strike

        x[y_min >= 0] = -np.inf
        x[y_max <= 0] = np.inf
        ind = ~((y_min >= 0) | (y_max <= 0))

        if np.all(~ind):
            return x[0] if scalar_output else x

        for k in range(32):
            y_vec = fac * np.exp(std * x[ind, None])
            y = (
                np.log(np.sum(y_vec, axis=-1)) - log_k[ind]
                if log
                else np.sum(y_vec, axis=-1) - strike[ind]
            )
            dy = (
                np.sum(std * y_vec, axis=-1) / np.sum(y_vec, axis=-1)
                if log
                else np.sum(std * y_vec, axis=-1)
            )
            x[ind] -= y / dy
            if len(y) == 0:
                print(ind, y_vec, y)
            y_err_max = np.amax(np.abs(y))
            if y_err_max < BsmBasket1Bm.IMPVOL_TOL:
                break

        if y_err_max > BsmBasket1Bm.IMPVOL_TOL:
            warn_msg = (
                f"root did not converge within {k} iterations: max error = {y_err_max}"
            )
            warnings.warn(warn_msg, Warning)

        return x[0] if scalar_output else x

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd * self.weight
        sigma_std = self.sigma * np.sqrt(texp)
        cp = np.array(cp)
        d2 = -cp * self.root(
            fwd_basket * np.exp(-(sigma_std ** 2) / 2), sigma_std, strike
        )

        if np.isscalar(d2):
            d1 = d2 + cp * sigma_std
        else:
            d1 = d2[:, None] + np.atleast_1d(cp)[:, None] * sigma_std

        price = np.sum(fwd_basket * spst.norm.cdf(d1), axis=-1)
        price -= strike * spst.norm.cdf(d2)
        price *= cp * df
        return price


class BsmBasketJsu(NormBasket):
    """

    Johnson's SU distribution approximation for Basket option pricing under the multiasset BSM model.

    Note: Johnson's SU distribution is the solution of NSVh with NSVh with lambda = 1.

    References:
        - Posner, S. E., & Milevsky, M. A. (1998). Valuing exotic options by approximating the SPD
        with higher moments. The Journal of Financial Engineering, 7(2). https://ssrn.com/abstract=108539

        - Choi, J., Liu, C., & Seo, B. K. (2019). Hyperbolic normal stochastic volatility model.
        Journal of Futures Markets, 39(2), 186–204. https://doi.org/10.1002/fut.21967

    """

    def moment_vsk(self, fwd, texp):
        """

        Return variance, skewness, kurtosis for Basket options.

        Args:
            fwd: forward price
            texp: time to expiry

        Returns: variance, skewness, kurtosis of Basket options

        """
        n = len(self.weight)

        m1 = sum(self.weight[i] * fwd[i] for i in range(n))

        m2_index = [i for i in product(np.arange(n), repeat=2)]
        m2 = sum(
            self.weight[i]
            * self.weight[j]
            * fwd[i]
            * fwd[j]
            * np.exp(self.sigma[i] * self.sigma[j] * self.cor_m[i][j] * texp)
            for i, j in m2_index
        )

        m3_index = [i for i in product(np.arange(n), repeat=3)]
        m3 = sum(
            self.weight[i]
            * self.weight[j]
            * self.weight[l]
            * fwd[i]
            * fwd[j]
            * fwd[l]
            * np.exp(
                sum(
                    self.sigma[ii] * self.sigma[jj] * self.cor_m[ii][jj]
                    for ii, jj in combinations(np.array([i, j, l]), 2)
                )
                * texp
            )
            for i, j, l in m3_index
        )

        m4_index = [i for i in product(np.arange(n), repeat=4)]
        m4 = sum(
            self.weight[i]
            * self.weight[j]
            * self.weight[l]
            * self.weight[k]
            * fwd[i]
            * fwd[j]
            * fwd[l]
            * fwd[k]
            * np.exp(
                sum(
                    self.sigma[ii] * self.sigma[jj] * self.cor_m[ii][jj]
                    for ii, jj in combinations(np.array([i, j, l, k]), 2)
                )
                * texp
            )
            for i, j, l, k in m4_index
        )

        var = m2 - m1 ** 2
        skew = (m3 - m1 ** 3 - 3 * m2 * m1 + 3 * m1 ** 3) / var ** (3 / 2)
        kurt = (m4 - 3 * m1 ** 4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2) / var ** 2

        return var, skew, kurt

    def price(self, strike, spot, texp, cp=1):
        """

        Basket options price.
        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
        Returns: Basket options price

        """
        fwd, df, _ = self._fwd_factor(spot, texp)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd @ self.weight

        var, skew, kurt = self.moment_vsk(fwd, texp)

        m = nsvh.Nsvh1(sigma=self.sigma)
        m.calibrate_vsk(var, skew, kurt - 3, texp, setval=True)
        price = m.price(strike, fwd_basket, texp, cp)

        return df * price


class BsmBasketChoi2018(NormBasket):
    """
    Choi (2018)'s pricing method for Basket/Spread/Asian options

    References
        - Choi J (2018) Sum of all Black-Scholes-Merton models: An efficient pricing method for spread, basket, and Asian options. Journal of Futures Markets 38:627–644. https://doi.org/10.1002/fut.21909
    """

    n_quad = None
    lam = 4.0

    @classmethod
    def init_lowerbound(cls, sigma, cor=None, weight=None, intr=0.0, divr=0.0, is_fwd=False):
        m = cls(sigma, cor=cor, weight=weight, intr=intr, divr=divr, is_fwd=is_fwd)
        m.n_quad = 0
        return m

    def set_num_params(self, n_quad=None, lam=3.0):
        self.n_quad = n_quad
        self.lam = lam

    @staticmethod
    def householder(vv0):
        """
        Returns a Householder reflection (orthonormal matrix) that maps (1,0,...0) to vv0

        Args:
            vv0: vector

        Returns:
            Reflection matrix

        References
            - https://en.wikipedia.org/wiki/Householder_transformation
        """
        vv1 = vv0 / np.linalg.norm(vv0)
        vv1[0] -= 1.0

        if abs(vv1[0]) < np.finfo(float).eps*100:
            return np.eye(len(vv1))
        else:
            return np.eye(len(vv1)) + vv1[:, None] * vv1 / vv1[0]

    def v_mat(self, fwd):
        """
        Construct the V matrix

        Args:
            fwd: forward vector of assets

        Returns:
            V matrix
        """

        fwd_wts_unit = fwd * self.weight
        fwd_wts_unit /= np.linalg.norm(fwd_wts_unit)

        v1 = self.cov_m @ fwd_wts_unit
        v1 /= np.sqrt(np.sum(v1 * fwd_wts_unit))

        thres = 0.01 * self.sigma
        idx = (np.sign(fwd_wts_unit) * v1 < thres)

        if np.any(idx):
            v1[idx] = (np.sign(fwd_wts_unit) * thres)[idx]
            q1 = np.linalg.solve(self.chol_m, v1)
            q1norm = np.linalg.norm(q1)
            q1 /= q1norm
            v1 /= q1norm
        else:
            q1 = self.chol_m.T @ fwd_wts_unit
            q1 /= np.linalg.norm(q1)

        r_mat = self.householder(q1)

        chol_r_mat = self.chol_m @ r_mat[:, 1:]
        svd_u, svd_d, _ = np.linalg.svd(chol_r_mat, full_matrices=False)

        v_mat = np.hstack((v1[:, None], svd_u @ np.diag(svd_d)))
        len_scale = svd_d / np.sum(fwd_wts_unit * v1)

        if self.n_quad is None:
            n_quad = np.rint(self.lam * len_scale + 1).astype(int)
        else:
            n_quad = self.n_quad

        return v_mat, n_quad

    def v1_fwd_weight(self, fwd, texp):
        """
        Construct v1, forward array, and weights

        Args:
            fwd: forward vector of assets
            texp: time to expiry

        Returns:
            (v1, f_k, ww)
        """

        v_mat, n_quad = self.v_mat(fwd)
        v_mat *= np.sqrt(texp)

        v1 = v_mat[:, 0]

        if n_quad == 0:
            # 1 factor BM model for lower bound
            f_k = np.ones((1, self.n_asset))
            ww = np.array([1.0])
        else:
            v_mat = v_mat[:, 1:len(n_quad)+1]
            quad = NdGHQ(n_quad)
            zz, ww = quad.z_vec_weight()
            f_k = np.exp(zz @ v_mat.T - 0.5*np.sum(v_mat**2, axis=1))

        return v1, f_k, ww

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)

        v1, f_k, ww = self.v1_fwd_weight(fwd, texp)
        m_1bm = BsmBasket1Bm(sigma=v1, weight=self.weight)

        price = np.zeros_like(strike, dtype=float)
        for k, f_k_row in enumerate(f_k):
            price1 = m_1bm.price(strike, f_k_row * fwd, texp=1.0, cp=cp)
            price += price1 * ww[k]

        return df * price