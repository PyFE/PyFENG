import scipy.stats as spst
import numpy as np
from . import opt_smile_abc as smile


class InvGam(smile.OptSmileABC):
    """
    Option pricing model with the inverse gamma (reciprocal gamma) distribution.

    The parameters (alpha, beta) is from Wikipedia. https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    Note that the n-th moment of the inverse gamma RV is beta^n / (alpha-1)*...*(alpha-n).
    Alpha and beta is calibrated to match the first two moments of the lognormal distribution with volatility sigma
    so that the option price is similar to that of the BSM model with volatility sigma.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.InvGam(sigma=0.2, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([15.49803779,  9.53595458,  5.49889751,  3.02086661,  1.60505654])
    """

    sigma = None

    @staticmethod
    def price_formula(
        strike, spot, texp, alpha, beta, cp=1, intr=0.0, divr=0.0, is_fwd=False
    ):
        disc_fac = np.exp(-texp * intr)
        fwd_scale = spot * (1.0 if is_fwd else np.exp(-texp * divr) / disc_fac) / beta
        kk = strike / beta

        price = np.where(
            cp > 0,
            fwd_scale * spst.gamma.cdf(x=1 / kk, a=alpha - 1)
            - kk * spst.gamma.cdf(x=1 / kk, a=alpha),
            kk * spst.gamma.sf(x=1 / kk, a=alpha)
            - fwd_scale * spst.gamma.sf(x=1 / kk, a=alpha - 1),
        )
        return disc_fac * beta * price

    def alpha_beta(self, spot, texp):
        """
        Computes the inverse gamma distribution parameters (alpha, beta) from sigma, spot, texp.

            m1 = beta/(alpha-1)

            m2/m1^2 = exp(sigma^2 T) = (alpha-1)/(alpha-2)

        Args:
            spot: spot (or forward) price
            texp: time to expiry

        Returns:
            (alpha, beta)
        """

        fwd = self.forward(spot, texp)
        alpha = 1 / (np.exp(self.sigma ** 2 * texp) - 1) + 2
        beta = (alpha - 1) * fwd
        return alpha, beta

    def price(self, strike, spot, texp, cp=1):
        alpha, beta = self.alpha_beta(spot, texp)
        return self.price_formula(
            strike, spot, texp, alpha, beta, cp=cp, intr=self.intr, divr=self.divr
        )

    def cdf(self, strike, spot, texp, cp=1):
        alpha, beta = self.alpha_beta(spot, texp)
        x = strike / beta
        cdf = np.where(
            cp > 0, spst.gamma.cdf(1 / x, a=alpha), spst.gamma.sf(1 / x, a=alpha)
        )
        return cdf


class InvGauss(smile.OptSmileABC):
    """
    Option pricing model with the inverse Gaussian (IG) distribution.

    The IG distribution with (gamma, delta) is modeled by scipy.stats.invgauss.invgauss(mu=1/(gamma*delta), scale=delta**2)
    When, sig = gamma = delta, the IG variable has mean 1 and variance 1/sig^2. We match the momoent by

        m2/m1^2 = 1/sig^2 = exp(sigma^2 T)

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.InvGauss(sigma=0.2, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([15.71924064,  9.70753358,  5.54459412,  2.95300168,  1.48019682])
    """

    sigma = None

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        sig2_inv = np.exp(self.sigma ** 2 * texp) - 1
        ig = spst.invgauss(mu=sig2_inv, scale=1 / sig2_inv)
        kk = strike / fwd
        price = np.where(
            cp > 0,
            ig.cdf(1 / kk) - kk * ig.sf(kk),
            kk * ig.cdf(kk) - ig.sf(1 / kk),
        )
        return df * fwd * price

    def cdf(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        sig2_inv = np.exp(self.sigma ** 2 * texp) - 1
        ig = spst.invgauss(mu=sig2_inv, scale=1 / sig2_inv)
        x = strike / fwd
        cdf = np.where(cp > 0, ig.sf(x), ig.cdf(x))
        return cdf
