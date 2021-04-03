import numpy as np
import scipy.stats as spst
#import numpy.polynomial as nppoly
from. import sabr


class Nsvh1(sabr.SabrABC):
    """
    Hyperbolic Normal Stochastic Volatility (NSVh) model with lambda=1 by Choi et al. (2019)

    References:
        Choi, J., Liu, C., & Seo, B. K. (2019). Hyperbolic normal stochastic volatility model.
        Journal of Futures Markets, 39(2), 186–204. https://doi.org/10.1002/fut.21967

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.Nsvh1(sigma=20, vov=0.2, rho=-0.3)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([22.45639334, 14.89800673,  8.88641613,  4.65917923,  2.10575204])
    """

    beta = 0.0  # beta is already defined in the parent class, but the default value set as 0

    def __init__(self, sigma, vov=0.0, rho=0.0, beta=None, intr=0.0, divr=0.0, is_fwd=False, atmvol=False):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            beta: elasticity parameter. should be 0 or None.
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        # Make sure beta = 0
        if beta is not None and not np.isclose(beta, 0.0):
            print(f'Ignoring beta = {beta}...')
        self._atmvol = atmvol
        super().__init__(sigma, vov, rho, beta=0, intr=intr, divr=divr, is_fwd=is_fwd)

    def _sig0_from_atmvol(self, texp):
        s_sqrt = self.vov * np.sqrt(texp)
        vov_var = np.exp(0.5 * s_sqrt**2)
        rhoc = np.sqrt(1 - self.rho**2)

        d = (np.arctanh(self.rho) - np.arcsinh(self.rho*vov_var/rhoc)) / s_sqrt
        ncdf_p = spst.norm.cdf(d + s_sqrt)
        ncdf_m = spst.norm.cdf(d - s_sqrt)
        ncdf = spst.norm.cdf(d)

        price = 0.5/self.vov*vov_var * ((1+self.rho)*ncdf_p - (1-self.rho)*ncdf_m - 2*self.rho*ncdf)
        sig0 = self.sigma*np.sqrt(texp/2/np.pi)/price
        return sig0

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)

        s_sqrt = self.vov * np.sqrt(texp)
        if self._atmvol:
            sig0 = self._sig0_from_atmvol(texp)
        else:
            sig0 = self.sigma

        sig_sqrt = sig0 * np.sqrt(texp)

        vov_var = np.exp(0.5 * s_sqrt**2)
        rhoc = np.sqrt(1 - self.rho**2)

        d = (np.arctanh(self.rho) + np.arcsinh(((fwd-strike)*s_sqrt/sig_sqrt - self.rho*vov_var)/rhoc)) / s_sqrt
        ncdf_p = spst.norm.cdf(cp*(d + s_sqrt))
        ncdf_m = spst.norm.cdf(cp*(d - s_sqrt))
        ncdf = spst.norm.cdf(cp*d)

        price = 0.5*sig_sqrt/s_sqrt*vov_var\
            * ((1+self.rho)*ncdf_p - (1-self.rho)*ncdf_m - 2*self.rho*ncdf)\
            + (fwd-strike) * ncdf
        price *= cp * df
        return price

    def cdf(self, strike, spot, texp, cp=-1):
        fwd = self.forward(spot, texp)

        s_sqrt = self.vov * np.sqrt(texp)
        sig_sqrt = self.sigma * np.sqrt(texp)
        vov_var = np.exp(0.5 * s_sqrt**2)
        rhoc = np.sqrt(1 - self.rho**2)

        d = (np.arctanh(self.rho) +
             np.arcsinh(((fwd - strike) * s_sqrt / sig_sqrt - self.rho * vov_var) / rhoc)) / s_sqrt
        return spst.norm.cdf(cp*d)
