import scipy.stats as ss
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

    def price(self, strike, spot, texp, cp=1):
        df = np.exp(-texp * self.intr)
        fwd1 = spot[0] * (1.0 if self.is_fwd else np.exp(-texp * self.divr) / df)
        fwd2 = spot[1] * (1.0 if self.is_fwd else np.exp(-texp * self.divr) / df)

        strike_m = np.minimum(strike, 0)
        strike_p = np.maximum(strike, 0)

        sig1 = self.sigma[0] * fwd1 / (fwd1 - strike_m)
        sig2 = self.sigma[1] * fwd2 / (fwd2 + strike_p)
        sig_spd = np.sqrt(sig1*(sig1 - 2.0*self.rho*sig2) + sig2**2)
        price = bsm.Bsm.price_formula(fwd2+strike_p, fwd1-strike_m, sig_spd, texp, cp=cp, is_fwd=True)
        return df * price


class NormBasket(opt.OptMaABC):
    """
    Basket option pricing under the multiasset Bachelier model
    """

    weight = None

    def __init__(self, sigma, cor=None, weight=None, intr=0.0, divr=0.0, is_fwd=False):
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
        fwd = np.array(spot) * (1.0 if self.is_fwd else np.exp(-self.divr*texp)/df)
        fwd_basket = np.sum(self.weight * fwd)
        vol_basket = np.sqrt(self.weight @ self.cov_m @ self.weight)

        price = norm.Norm.price_formula(
            strike, fwd_basket, vol_basket, texp, cp=cp, is_fwd=True)
        return df * price


class NormSpread(opt.OptMaABC):
    """
    Spread option pricing under the Bachelier model.
    This is a special case of NormBasket with weight = (1, -1)
    """
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
        fwd = np.array(spot) * (1.0 if self.is_fwd else np.exp(-self.divr*texp)/df)
        weight_fwd = self.weight * fwd
        m1 = np.sum(weight_fwd)
        m2 = weight_fwd @ np.exp(self.cov_m * texp) @ weight_fwd

        sig = np.sqrt(np.log(m2/(m1**2))/texp)
        price = bsm.Bsm.price_formula(
            strike, m1, sig, texp, cp=cp, is_fwd=True)
        return df * price


class BsmRainbow2(opt.OptMaABC):
    def price(self, strike, spot, texp, cp=1):
        sig = self.sigma
        df = np.exp(-texp * self.intr)
        fwd = spot * (1.0 if self.is_fwd else np.exp(-texp * self.divr) / df)

        sig_std = sig * np.sqrt(texp)
        spd_rho = np.sqrt(np.dot(sig, sig) - 2*self.rho*sig[0]*sig[1])
        spd_std = spd_rho * np.sqrt(texp)

        # -x and y as rows
        # supposed to be -log(fwd/strike) but strike is added later
        xx = -np.log(fwd)/sig_std - 0.5*sig_std

        fwd_ratio = fwd[0]/fwd[1]
        yy = np.log([fwd_ratio, 1/fwd_ratio])/spd_std + 0.5*spd_std

        rho12 = np.array([self.rho*sig[1]-sig[0], self.rho*sig[0]-sig[1]])/spd_rho

        low = np.array([-10, -10])
        mu0 = np.array([0, 0])

        cor_m1 = rho12[0]*np.ones((2, 2)) + (1-rho12[0])*np.eye(2)
        cor_m2 = rho12[1]*np.ones((2, 2)) + (1-rho12[1])*np.eye(2)

        strike = np.atleast_1d(strike)
        n_strike = len(strike)

        price = np.zeros_like(strike, float)
        for k in range(n_strike):
            xx_ = xx + np.log(strike[k])/sig_std
            term1, i1 = fwd[0] * (ss.norm.cdf(yy[0]) - ss.mvn.mvnun(low, np.array([xx_[0], yy[0]]), mu0, cor_m1))
            term2, i2 = fwd[1] * (ss.norm.cdf(yy[1]) - ss.mvn.mvnun(low, np.array([xx_[1], yy[1]]), mu0, cor_m2))
            term3, i3 = strike[k] * np.array(ss.mvn.mvnun(low, xx_ + sig_std, mu0, self.cor_m))

            assert(term1 + term2 + term3 >= strike[k])
            price[k] = (term1 + term2 + term3 - strike[k])

        price *= df

        return price if n_strike > 1 else price[0]
