import numpy as np
import scipy.stats as spst
from . import sabr


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
    is_atmvol = False

    def __init__(self, sigma, vov=0.0, rho=0.0, beta=None, intr=0.0, divr=0.0, is_fwd=False, is_atmvol=False):
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
            print(f'Ignoring beta = {beta}...')
        self.is_atmvol = is_atmvol
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
        if self.is_atmvol:
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


class NsvhMc(sabr.SabrABC):
    """
    Monte-Carlo model of Hyperbolic Normal Stochastic Volatility (NSVh) model

    References:
        Choi, J., Liu, C., & Seo, B. K. (2019). Hyperbolic normal stochastic volatility model.
        Journal of Futures Markets, 39(2), 186–204. https://doi.org/10.1002/fut.21967
    """

    lam = 0
    n_path = int(1e6)
    rn_seed = None
    rng = np.random.default_rng(None)
    antithetic = True

    def __init__(self, sigma, vov=0.0, rho=0.0, lam=0, beta=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            lam: lambda. Norma SABR if 0, Johnson's SU if 1 (same as `Nsvh1`)
            beta: elasticity parameter. should be 0 or None.
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        # Make sure beta = 0
        if beta is not None and not np.isclose(beta, 0.0):
            print(f'Ignoring beta = {beta}...')
        self.lam = lam
        super().__init__(sigma, vov, rho, beta=0, intr=intr, divr=divr, is_fwd=is_fwd)

    def set_mc_params(self, n_path=1e6, rn_seed=None, antithetic=True):
        self.n_path = int(n_path)
        self.rn_seed = rn_seed
        self.antithetic = antithetic
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)

    def mc_vol_price(self, texp):
        """
        Simulate volatility and price pair

        Args:
            texp: time-to-expiry

        Returns: (vol, price). vol: (n_path, ), price: (n_path, 2)

        """
        # forward path starting from zero
        # returns both sigma and price

        rhoc = np.sqrt(1-self.rho**2)
        vol_var = self.vov**2 * texp
        vol_std = np.sqrt(vol_var)

        z_rn = self.rng.normal(size=(int(self.n_path/2), 3))
        z_rn = np.stack([z_rn, -z_rn], axis=1).reshape((-1, 3))
        z_rn[:, 2] += 0.5*(self.lam-1)*vol_std  # add shift

        r2 = np.sum(z_rn[:, 0:2]**2, axis=1)
        exp_plus = np.exp(0.5 * vol_std * z_rn[:, 2])

        phi_r1 = np.sqrt(2/r2) * np.sqrt(np.cosh(np.sqrt(r2 + z_rn[:, 2]**2)*vol_std) - np.cosh(z_rn[:, 2]*vol_std))

        df_z = exp_plus**2
        df_w = exp_plus[:, None] * z_rn[:, 0:2] * phi_r1[:, None]  # use both X and Y components

        path = (
            df_z,
            (self.sigma/self.vov) * (self.rho*(df_z[:, None] - np.exp(0.5*self.lam*vol_var)) + rhoc*df_w)
        )
        return path

    def price(self, strike, spot, texp, cp=1):
        """
        Vanilla option price from MC simulation of NSVh model.

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1 or call, -1 for put option

        Returns:
            vanilla option price
        """

        fwd, df, _ = self._fwd_factor(spot, texp)
        mc_path = self.mc_vol_price(texp)
        strike_std = strike - fwd
        scalar_output = np.isscalar(strike_std)
        cp *= np.ones_like(strike_std)

        price = np.array([np.mean(np.fmax(cp[k] * (mc_path[1] - strike_std[k]), 0)) for k in range(len(strike_std))])
        if scalar_output:
            price = price[0]
        return df*price
