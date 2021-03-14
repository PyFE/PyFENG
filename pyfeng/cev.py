import scipy.stats as spst
import scipy.special as spsp
import numpy as np
from . import opt_abc as opt
from . import opt_smile_abc as smile


class Cev(opt.OptAnalyticABC, smile.OptSmileABC, smile.MassZeroABC):
    """
    Constant Elasticity of Variance (CEV) model.

    Underlying price is assumed to follow CEV process:
    dS_t = (r - q) S_t dt + sigma S_t^beta dW_t, where dW_t is a standard Brownian motion.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.Cev(sigma=0.2, beta=0.5, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([16.11757214, 10.00786871,  5.64880408,  2.89028476,  1.34128656])
    """
    sigma = None
    beta = 0.5
    is_bsm_sigma = False

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
        extra = {'beta': self.beta}
        return {**params, **extra}  # Py 3.9, params | extra

    def mass_zero(self, spot, texp, log=False):
        fwd = self.forward(spot, texp)

        betac = 1.0 - self.beta
        a = 0.5 / betac
        sigma_std = np.maximum(self.sigma / np.power(fwd, betac) * np.sqrt(texp), np.finfo(float).eps)
        x = 0.5 / np.square(betac * sigma_std)

        if log:
            log_mass = (a - 1)*np.log(x) - x - np.log(spsp.gamma(a))
            log_mass += np.log(1 + (a-1)/x*(1 + (a-2)/x*(1 + (a-3)/x*(1 + (a-4)/x))))
            with np.errstate(divide='ignore'):
                log_mass = np.where(x > 100, log_mass, np.log(spst.gamma.sf(x=x, a=a)))
            return log_mass
        else:
            return spst.gamma.sf(x=x, a=a)

    def mass_zero_t0(self, spot, texp):
        """
        Limit value of -T log(M_T) as T -> 0, where M_T is the mass at zero.

        Args:
            spot: spot (or forward) price

        Returns:
            - lim_{T->0} T log(M_T)
        """
        fwd = self.forward(spot, texp)
        betac = 1.0 - self.beta
        alpha = self.sigma/np.power(fwd, betac)
        t0 = 0.5/(betac*alpha)**2
        return t0

    @staticmethod
    def price_formula(strike, spot, texp, cp=1, sigma=None, beta=0.5, intr=0.0, divr=0.0, is_fwd=False):
        """

        Args:
            strike:
            spot:
            texp:
            cp:
            sigma:
            beta:
            intr:
            divr:
            is_fwd:

        Returns:

        """

        disc_fac = np.exp(-texp * intr)
        fwd = spot * (1.0 if is_fwd else np.exp(-texp * divr) / disc_fac)

        betac = 1.0 - beta
        betac_inv = 1.0 / betac
        alpha = sigma/np.power(fwd, betac)
        sigma_std = np.maximum(alpha*np.sqrt(texp), np.finfo(float).eps)
        kk = strike / fwd
        x = 1.0 / np.square(betac * sigma_std)
        y = np.power(kk, 2 * betac) * x

        # Need to clean up the case beta > 0
        if beta > 1.0:
            raise ValueError('Cannot handle beta value higher than 1.0')

        ncx2_sf = spst.ncx2.sf
        ncx2_cdf = spst.ncx2.cdf

        # Computing call and put is a bit of computtion waste, but do this for vectorization.
        price = np.where(
            cp > 0,
            fwd * ncx2_sf(y, 2 + betac_inv, x) - strike * ncx2_cdf(x, betac_inv, y),
            strike * ncx2_sf(x, betac_inv, y) - fwd * ncx2_cdf(y, 2 + betac_inv, x)
        )
        return disc_fac * price

    def delta(self, strike, spot, texp, cp_sign=1):
        fwd, df, divf = self._fwd_factor(spot, texp)
        betac_inv = 1 / (1 - self.beta)

        k_star = 1.0 / np.square(self.sigma / betac_inv) / texp
        x = k_star * np.power(fwd, 2 / betac_inv)
        y = k_star * np.power(strike, 2 / betac_inv)

        if self.beta < 1.0:
            delta = 0.5 * (cp_sign - 1) + spst.ncx2.sf(y, 2 + betac_inv, x) + 2 * x / betac_inv * \
                    (spst.ncx2.pdf(y, 4 + betac_inv, x) - strike / fwd * spst.ncx2.pdf(x, betac_inv, y))
        else:
            delta = 0.5 * (cp_sign - 1) + spst.ncx2.sf(x, -betac_inv, y) - 2 * x / betac_inv * \
                    (spst.ncx2.pdf(x, -betac_inv, y) - strike / fwd * spst.ncx2.pdf(y, 4 - betac_inv, x))

        delta *= df if self.is_fwd else divf
        return delta

    def gamma(self, strike, spot, texp, cp_sign=1):
        fwd, df, divf = self._fwd_factor(spot, texp)
        betac_inv = 1 / (1 - self.beta)

        k_star = 1.0 / np.square(self.sigma / betac_inv) / texp
        x = k_star * np.power(fwd, 2 / betac_inv)
        y = k_star * np.power(strike, 2 / betac_inv)

        if self.beta < 1.0:
            gamma = (2 + betac_inv - x) * spst.ncx2.pdf(y, 4 + betac_inv, x) + x * spst.ncx2.pdf(y, 6 + betac_inv, x) + \
                    strike / fwd * (x * spst.ncx2.pdf(x, betac_inv, y) - y * spst.ncx2.pdf(x, 2 + betac_inv, y))
        else:
            gamma = (x * spst.ncx2.pdf(x, -betac_inv, y) - y * spst.ncx2.pdf(x, 2 - betac_inv, y)) + \
                    strike / fwd * ((2 - betac_inv - x) * spst.ncx2.pdf(y, 4 - betac_inv, x) + \
                                    x * spst.ncx2.pdf(y, 6 - betac_inv, x))

        gamma *= 2 * (divf/betac_inv)**2 / df * x / fwd

        if self.is_fwd:
            gamma *= (df/divf)**2

        return gamma

    def vega(self, strike, spot, texp, cp_sign=1):
        fwd, df, divf = self._fwd_factor(spot, texp)
        spot = fwd * df / divf

        betac_inv = 1 / (1 - self.beta)

        k_star = 1.0 / np.square(self.sigma / betac_inv) / texp
        x = k_star * np.power(fwd, 2 / betac_inv)
        y = k_star * np.power(strike, 2 / betac_inv)

        if self.beta < 1.0:
            vega = - fwd * spst.ncx2.pdf(y, 4 + betac_inv, x) + strike * spst.ncx2.pdf(x, betac_inv, y)
        else:
            vega = fwd * spst.ncx2.pdf(x, -betac_inv, y) - strike * spst.ncx2.pdf(y, 4 - betac_inv, x)

        sigma = self.sigma * spot ** (self.beta - 1)

        vega *= df * 2 * x / sigma
        return vega

    def theta(self, strike, spot, texp, cp=1):
        ### Need to implement this
        return self.theta_numeric(strike, spot, texp, cp=cp)
