import scipy.stats as scst
import numpy as np
from . import bsm
from . import norm
from . import opt_abc as opt


class BsmSpreadKirk(opt.OptMaABC):
    """
    Kirk's approximation for spread option.

    References:
        Kirk, E. (1995). Correlation in the energy markets. In Managing Energy Price Risk
        (First, pp. 71–78). Risk Publications.
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
    This is a special case of NormBasket with weight = (1, -1)    """
    weight = np.array([1, -1])

    price = NormBasket.price


class BsmBasketLevy1992(NormBasket):
    """
    Basket option pricing with the log-normal approximation of Levy & Turnbull (1992)

    References:
        Levy, E., & Turnbull, S. (1992). Average intelligence. Risk, 1992(2), 53–57.

        Krekel, M., de Kock, J., Korn, R., & Man, T.-K. (2004). An analysis of pricing methods for basket options.
        Wilmott Magazine, 2004(7), 82–89.
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


class BsmRainbow2(opt.OptMaABC):
    """
    Option on the max of two assets.
    Payout = max( max(F_1, F_2) - K, 0 )

    References:
        Rubinstein, M. (1991). Somewhere Over the Rainbow. Risk, 1991(11), 63–66.

    """
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
        n_strike = len(strike)

        price = np.zeros_like(strike, float)
        for k in range(n_strike):
            xx_ = xx + np.log(strike[k])/sig_std
            term1 = fwd[0] * (scst.norm.cdf(yy[0]) - scst.multivariate_normal.cdf(np.array([xx_[0], yy[0]]), mu0, cor_m1))
            term2 = fwd[1] * (scst.norm.cdf(yy[1]) - scst.multivariate_normal.cdf(np.array([xx_[1], yy[1]]), mu0, cor_m2))
            term3 = strike[k] * np.array(scst.multivariate_normal.cdf(xx_ + sig_std, mu0, self.cor_m))

            assert(term1 + term2 + term3 >= strike[k])
            price[k] = (term1 + term2 + term3 - strike[k])

        price *= df

        return price[0] if strike_isscalar else price
