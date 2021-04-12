import scipy.stats as spst
import warnings
import numpy as np
from . import bsm
from . import norm
from . import gamma
from . import opt_abc as opt


class BsmSpreadKirk(opt.OptMaABC):
    """
    Kirk's approximation for spread option.

    References:
        Kirk, E. (1995). Correlation in the energy markets. In Managing Energy Price Risk
        (First, pp. 71–78). Risk Publications.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmSpreadKirk((0.2, 0.3), cor=-0.5)
        >>> m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        array([22.15632247, 17.18441817, 12.98974214,  9.64141666,  6.99942072])
    """

    weight = np.array([1, -1])

    def price(self, strike, spot, texp, cp=1):
        df = np.exp(-texp * self.intr)
        fwd = np.array(spot) * (1.0 if self.is_fwd else np.exp(-texp * np.array(self.divr)) / df)
        assert fwd.shape[-1] == self.n_asset

        fwd1 = fwd[..., 0] - np.minimum(strike, 0)
        fwd2 = fwd[..., 1] + np.maximum(strike, 0)

        sig1 = self.sigma[0] * fwd[..., 0] / fwd1
        sig2 = self.sigma[1] * fwd[..., 1] / fwd2
        sig_spd = np.sqrt(sig1*(sig1 - 2.0*self.rho*sig2) + sig2**2)
        price = bsm.Bsm.price_formula(fwd2, fwd1, sig_spd, texp, cp=cp, is_fwd=True)
        return df * price


class BsmSpreadBjerksund2014(opt.OptMaABC):
    """
    Bjerksund & Stensland (2014)'s approximation for spread option.

    References:
        Bjerksund, P., & Stensland, G. (2014). Closed form spread option valuation.
        Quantitative Finance, 14(10), 1785–1794. https://doi.org/10.1080/14697688.2011.617775

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmSpreadBjerksund2014((0.2, 0.3), cor=-0.5)
        >>> m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        array([22.13172022, 17.18304247, 12.98974214,  9.54431944,  6.80612597])
    """

    weight = np.array([1, -1])

    def price(self, strike, spot, texp, cp=1):
        df = np.exp(-texp * self.intr)
        fwd = np.array(spot) * (1.0 if self.is_fwd else np.exp(-texp * np.array(self.divr)) / df)
        assert fwd.shape[-1] == self.n_asset

        fwd1 = fwd[..., 0]
        fwd2 = fwd[..., 1]

        std11 = self.sigma[0]**2 * texp
        std12 = self.sigma[0]*self.sigma[1] * texp
        std22 = self.sigma[1]**2 * texp

        aa = fwd2 + strike
        bb = fwd2/aa
        std = np.sqrt(std11 - 2*bb*self.rho*std12 + bb**2*std22)

        d3 = np.log(fwd1/aa)
        d1 = (d3 + 0.5*std11 - bb*(self.rho*std12 - 0.5*bb*std22)) / std
        d2 = (d3 - 0.5*std11 + self.rho*std12 + bb*(0.5*bb - 1)*std22) / std
        d3 = (d3 - 0.5*std11 + 0.5*bb**2*std22) / std

        price = cp*(fwd1*spst.norm.cdf(cp*d1) - fwd2*spst.norm.cdf(cp*d2)
                    - strike*spst.norm.cdf(cp*d3))

        return df * price


class NormBasket(opt.OptMaABC):
    """
    Basket option pricing under the multiasset Bachelier model
    """

    weight = None

    def __init__(self, sigma, cor=None, weight=None, intr=0.0, divr=0.0, is_fwd=False):
        """
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
        """

        super().__init__(sigma, cor=cor, intr=intr, divr=divr, is_fwd=is_fwd)
        if weight is None:
            self.weight = np.ones(self.n_asset) / self.n_asset
        elif np.isscalar(weight):
            self.weight = np.ones(self.n_asset) * weight
        else:
            assert len(weight) == self.n_asset
            self.weight = np.array(weight)

    def price(self, strike, spot, texp, cp=1):
        df = np.exp(-texp * self.intr)
        fwd = np.array(spot) * (1.0 if self.is_fwd else np.exp(-texp * np.array(self.divr)) / df)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd @ self.weight
        vol_basket = np.sqrt(self.weight @ self.cov_m @ self.weight)

        price = norm.Norm.price_formula(
            strike, fwd_basket, vol_basket, texp, cp=cp, is_fwd=True)
        return df * price


class NormSpread(opt.OptMaABC):
    """
    Spread option pricing under the Bachelier model.
    This is a special case of NormBasket with weight = (1, -1)

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.NormSpread((20, 30), cor=-0.5, intr=0.05)
        >>> m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        array([17.95676186, 13.74646821, 10.26669936,  7.47098719,  5.29057157])
    """
    weight = np.array([1, -1])

    price = NormBasket.price


class BsmBasketLevy1992(NormBasket):
    """
    Basket option pricing with the log-normal approximation of Levy & Turnbull (1992)

    References:
        - Levy, E., & Turnbull, S. (1992). Average intelligence. Risk, 1992(2), 53–57.
        - Krekel, M., de Kock, J., Korn, R., & Man, T.-K. (2004). An analysis of pricing methods for basket options.
        Wilmott Magazine, 2004(7), 82–89.

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
        df = np.exp(-texp * self.intr)
        fwd = np.array(spot) * (1.0 if self.is_fwd else np.exp(-texp * np.array(self.divr)) / df)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd * self.weight
        m1 = np.sum(fwd_basket, axis=-1)
        m2 = np.sum(fwd_basket @ np.exp(self.cov_m * texp) * fwd_basket, axis=-1)

        sig = np.sqrt(np.log(m2/(m1**2))/texp)
        price = bsm.Bsm.price_formula(
            strike, m1, sig, texp, cp=cp, is_fwd=True)
        return df * price


class BsmBasketMilevsky1998(NormBasket):
    """
    Basket option pricing with the inverse gamma distribution of Milevsky & Posner (1998)

    References:
        Milevsky, M. A., & Posner, S. E. (1998). A Closed-Form Approximation for Valuing Basket Options.
        The Journal of Derivatives, 5(4), 54–61. https://doi.org/10.3905/jod.1998.408005

        Krekel, M., de Kock, J., Korn, R., & Man, T.-K. (2004). An analysis of pricing methods for basket options.
        Wilmott Magazine, 2004(7), 82–89.

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
        df = np.exp(-texp * self.intr)
        fwd = np.array(spot) * (1.0 if self.is_fwd else np.exp(-texp * np.array(self.divr)) / df)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd * self.weight
        m1 = np.sum(fwd_basket, axis=-1)
        m2 = np.sum(fwd_basket @ np.exp(self.cov_m * texp) * fwd_basket, axis=-1)

        alpha = 1/(m2/m1**2-1) + 2
        beta = (alpha-1)*m1

        price = gamma.InvGam.price_formula(strike, m1, texp, alpha, beta, cp=cp, is_fwd=True)
        return df * price


class BsmMax2(opt.OptMaABC):
    """
    Option on the max of two assets.
    Payout = max( max(F_1, F_2) - K, 0 ) for all or  max( K - max(F_1, F_2), 0 ) for put option

    References:
        Rubinstein, M. (1991). Somewhere Over the Rainbow. Risk, 1991(11), 63–66.

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
        df = np.exp(-texp * self.intr)
        fwd = np.array(spot) * (1.0 if self.is_fwd else np.exp(-texp * np.array(self.divr)) / df)
        assert fwd.shape[-1] == self.n_asset

        sig_std = sig * np.sqrt(texp)
        spd_rho = np.sqrt(np.dot(sig, sig) - 2*self.rho*sig[0]*sig[1])
        spd_std = spd_rho * np.sqrt(texp)

        # -x and y as rows
        # supposed to be -log(fwd/strike) but strike is added later
        xx = -np.log(fwd)/sig_std - 0.5*sig_std

        fwd_ratio = fwd[0]/fwd[1]
        yy = np.log([fwd_ratio, 1/fwd_ratio])/spd_std + 0.5*spd_std

        rho12 = np.array([self.rho*sig[1]-sig[0], self.rho*sig[0]-sig[1]])/spd_rho

        mu0 = np.zeros(2)
        cor_m1 = rho12[0] + (1-rho12[0])*np.eye(2)
        cor_m2 = rho12[1] + (1-rho12[1])*np.eye(2)

        strike_isscalar = np.isscalar(strike)
        strike = np.atleast_1d(strike)
        cp = cp * np.ones_like(strike)
        n_strike = len(strike)

        # this is the price of max(S1, S2) = max(S1-S2, 0) + S2
        # Used that Kirk approximation strike = 0 is Margrabe's switch option price
        parity = 0 if np.all(cp > 0) else self.m_switch.price(0, fwd, texp) + fwd[1]
        price = np.zeros_like(strike, float)
        for k in range(n_strike):
            xx_ = xx + np.log(strike[k])/sig_std
            term1 = fwd[0] * (spst.norm.cdf(yy[0]) - spst.multivariate_normal.cdf(np.array([xx_[0], yy[0]]), mu0, cor_m1))
            term2 = fwd[1] * (spst.norm.cdf(yy[1]) - spst.multivariate_normal.cdf(np.array([xx_[1], yy[1]]), mu0, cor_m2))
            term3 = strike[k] * np.array(spst.multivariate_normal.cdf(xx_ + sig_std, mu0, self.cor_m))

            assert(term1 + term2 + term3 >= strike[k])
            price[k] = (term1 + term2 + term3 - strike[k])

            if cp[k] < 0:
                price[k] += strike[k] - parity
        price *= df

        return price[0] if strike_isscalar else price


class BsmBasket1Bm(opt.OptABC):
    """
    Basket/Spread option when all asset prices are driven by a single

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

        with np.errstate(divide='ignore', invalid='ignore'):
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

        for k in range(32):
            y_vec = fac * np.exp(std * x[ind, None])
            y = np.log(np.sum(y_vec, axis=-1)) - log_k[ind] if log else np.sum(y_vec, axis=-1) - strike[ind]
            dy = np.sum(std * y_vec, axis=-1) / np.sum(y_vec, axis=-1) if log else np.sum(std * y_vec, axis=-1)
            x[ind] -= y / dy

            y_err_max = np.amax(np.abs(y))
            if y_err_max < BsmBasket1Bm.IMPVOL_TOL:
                break

        if y_err_max > BsmBasket1Bm.IMPVOL_TOL:
            warn_msg = f'root did not converge within {k} iterations: max error = {y_err_max}'
            warnings.warn(warn_msg, Warning)

        return x[0] if scalar_output else x

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd * self.weight
        sigma_std = self.sigma * np.sqrt(texp)
        cp = np.array(cp)
        d2 = -cp * self.root(fwd_basket * np.exp(-sigma_std**2/2), sigma_std, strike)

        if np.isscalar(d2):
            d1 = d2 + cp*sigma_std
        else:
            d1 = d2[:, None] + np.atleast_1d(cp)[:, None]*sigma_std

        price = np.sum(fwd_basket*spst.norm.cdf(d1), axis=-1)
        price -= strike * spst.norm.cdf(d2)
        price *= cp*df
        return price