import numpy as np
import scipy.stats as spst
import scipy.special as spsp
import scipy.optimize as spop
from . import sabr


class Nsvh1(sabr.SabrABC):
    """
    Hyperbolic Normal Stochastic Volatility (NSVh) model with lambda=1 by Choi et al. (2019)

    References:
        - Choi J, Liu C, Seo BK (2019) Hyperbolic normal stochastic volatility model. J Futures Mark 39:186–204. https://doi.org/10.1002/fut.21967

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.Nsvh1(sigma=20, vov=0.8, rho=-0.3)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([25.51200027, 17.87539874, 11.47308947,  6.75128331,  3.79464422])
    """

    model_type = "Nsvh"
    beta = 0.0  # beta is already defined in the parent class, but the default value set as 0
    is_atmvol = False

    def _sig0_from_atmvol(self, texp):
        vovn = self.vov * np.sqrt(texp)
        vov_var = np.exp(0.5 * vovn**2)
        rhoc = np.sqrt(1 - self.rho**2)

        d = (np.arctanh(self.rho) - np.arcsinh(self.rho * vov_var / rhoc)) / vovn
        ncdf_p = spst.norm.cdf(d + vovn)
        ncdf_m = spst.norm.cdf(d - vovn)
        ncdf = spst.norm.cdf(d)

        price = 0.5/self.vov*vov_var * ((1 + self.rho) * ncdf_p - (1 - self.rho) * ncdf_m - 2 * self.rho * ncdf)
        sig0 = self.sigma * np.sqrt(texp/2/np.pi) / price
        return sig0

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
            is_atmvol: If True, use `sigma` as the ATM normal vol. False by default.
        """
        # Make sure beta = 0
        if beta is not None and not np.isclose(beta, 0.0):
            print(f"Ignoring beta = {beta}...")
        self.is_atmvol = is_atmvol
        super().__init__(sigma, vov, rho, beta=0, intr=intr, divr=divr, is_fwd=is_fwd)

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)

        vovn = self.vov * np.sqrt(texp)
        sig0 = self._sig0_from_atmvol(texp) if self.is_atmvol else self.sigma
        sig_sqrt = sig0 * np.sqrt(texp)

        vov_var = np.exp(0.5 * vovn**2)
        rhoc = np.sqrt(1 - self.rho**2)

        d = (np.arctanh(self.rho) + np.arcsinh(((fwd - strike)*vovn/sig_sqrt - self.rho*vov_var) / rhoc)) / vovn
        ncdf_p = spst.norm.cdf(cp * (d + vovn))
        ncdf_m = spst.norm.cdf(cp * (d - vovn))
        ncdf = spst.norm.cdf(cp * d)

        price = 0.5*sig_sqrt/vovn*vov_var \
                * ((1 + self.rho)*ncdf_p - (1 - self.rho)*ncdf_m - 2*self.rho*ncdf) + (fwd - strike)*ncdf
        price *= cp * df

        return price

    def cdf(self, strike, spot, texp, cp=-1):
        fwd = self.forward(spot, texp)

        vovn = self.vov * np.sqrt(texp)
        sig_sqrt = self.sigma * np.sqrt(texp)
        vov_var = np.exp(0.5 * vovn**2)
        rhoc = np.sqrt(1 - self.rho**2)

        d = (np.arctanh(self.rho) + np.arcsinh(((fwd - strike)*vovn/sig_sqrt - self.rho*vov_var) / rhoc)) / vovn
        return spst.norm.cdf(cp * d)

    def moments_vsk(self, texp=1):
        """
        Variance, skewness, and ex-kurtosis

        Args:
            texp: time-to-expiry

        Returns:
            (variance, skewness, and ex-kurtosis)
        """
        vovn = self.vov * np.sqrt(texp)
        ww = np.exp(vovn**2)
        rho2 = self.rho**2
        rho3 = self.rho * rho2
        rho4 = rho2**2

        c20 = ww + 1
        c22 = ww - 1

        c31 = 3 * (ww + 1)**2
        c33 = (ww - 1) * (ww + 3)

        # C40 = (ww + 1)**2*(ww**4 + 2*ww**2 + 3)
        # C42 = 6*(ww**2 - 1)*(ww**4 + 2*ww**3 + 4*ww**2 + 2*ww + 1)
        # C44 = (ww - 1)**2*(ww**4 + 4*ww**3 + 10*ww**2 + 12*ww + 3)
        # M3 = (1 / 4)*np.sqrt(ww)*(ww - 1)**2*(self.rho*C_31 + rho3*C_33)
        # M4 = (1 / 8)*(ww - 1)**2*(C_40 + rho2*C_42 + rho4*C_44)

        m2 = (1 / 2) * (ww - 1) * (c20 + rho2 * c22)

        k0 = (ww + 1)**3 * (ww**2 + 3)
        k2 = 6 * (ww + 1)**2 * (ww**3 + ww**2 + 3 * ww - 1)
        k4 = (ww - 1) * (ww**4 + 4 * ww**3 + 10 * ww**2 + 12 * ww - 3)

        skew = (
            np.sqrt(ww * (ww - 1) / 2)
            * (self.rho * c31 + rho3 * c33)
            / np.power(c20 + rho2 * c22, 1.5)
        )
        exkurt = 0.5 * (ww - 1) * (k0 + rho2 * k2 + rho4 * k4) / (c20 + rho2 * c22)**2

        return m2 * (self.sigma / self.vov)**2, skew, exkurt

    def calibrate_vsk(self, var, skew, exkurt, texp=1, setval=False):
        """
        Calibrate parameters to the moments: variance, skewness, ex-kurtosis.

        Args:
            texp: time-to-expiry
            var: variance
            skew: skewness
            exkurt: ex-kurtosis. should be > 0.

        Returns: (sigma, vov, rho)

        References:
            Tuenter, H. J. H. (2001). An algorithm to determine the parameters of SU-curves in the johnson system of probabillity distributions by moment matching. Journal of Statistical Computation and Simulation, 70(4), 325–347. https://doi.org/10.1080/00949650108812126
        """
        assert exkurt > 0
        beta1 = skew**2
        beta2 = exkurt + 3

        # min of w search
        roots = np.roots(np.array([1, 2, 3, 0, -3 - beta2]))
        roots = roots[(roots.real > 0) & np.isclose(roots.imag, 0)]
        assert len(roots) == 1
        w_min = roots.real[0]
        w_max = np.sqrt(-1 + np.sqrt(2 * (beta2 - 1)))

        def f_beta1(w):
            term1 = np.sqrt(4 + 2 * (w * w - (beta2 + 3) / (w * w + 2 * w + 3)))
            return (w + 1 - term1) * (w + 1 + 0.5 * term1)**2 - beta1

        assert f_beta1(w_min) >= 0
        # print(w_min, f_beta1(w_min), w_max, f_beta1(w_max))

        # root finding for w = np.exp(S) = np.exp(vov^2 texp)
        w_root = spop.brentq(f_beta1, w_min, w_max)
        m = -2 + np.sqrt(4 + 2 * (w_root**2 - (beta2 + 3) / (w_root**2 + 2 * w_root + 3)))
        term = (w_root + 1) / (2 * w_root) * ((w_root - 1) / m - 1)  # - sinh(Omega) = rho / rhoc*

        # if term is slightly negative, next line error in sqrt
        if abs(term) < np.finfo(float).eps * 100:
            term = 0.0

        rho = np.sign(skew) * np.sqrt(1 - 1 / (1 + term))
        vov = np.sqrt(np.log(w_root) / texp)
        m2 = 0.5 * (w_root - 1) * ((w_root + 1) + rho**2 * (w_root - 1))
        sig0 = np.sqrt(var / m2) * vov

        if setval:
            self.sigma = sig0
            self.vov = vov
            self.rho = rho

        return sig0, vov, rho


class NsvhMc(sabr.SabrABC):
    """
    Monte-Carlo model of Hyperbolic Normal Stochastic Volatility (NSVh) model.

    NSVh with lambda = 0 is the normal SABR model, and NSVh with lambda = 1 has analytic pricing (Nsvh1)

    References:
        - Choi J, Liu C, Seo BK (2019) Hyperbolic normal stochastic volatility model. J Futures Mark 39:186–204. https://doi.org/10.1002/fut.21967

    See Also:
        Nsvh1

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.NsvhMc(sigma=20, vov=0.8, rho=-0.3, lam=0.0)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([23.52722081, 15.63212633,  9.19644639,  4.81061848,  2.39085097])
        >>> m1 = pf.NsvhMc(sigma=20, vov=0.8, rho=-0.3, lam=1.0)
        >>> m2 = pf.Nsvh1(sigma=20, vov=0.8, rho=-0.3)
        >>> p1 = m1.price(np.arange(80, 121, 10), 100, 1.2)
        >>> p2 = m2.price(np.arange(80, 121, 10), 100, 1.2)
        >>> p1 - p2
        array([-0.00328887,  0.00523714,  0.00808885,  0.0069694 ,  0.00205566])
    """

    lam = 0
    n_path = int(16e4)
    rn_seed = None
    rng = np.random.default_rng(None)
    antithetic = True

    def __init__(self, sigma, vov=0.0, rho=0.0, lam=0.0, beta=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            lam: lambda. Normal SABR if 0, Johnson's SU if 1 (same as `Nsvh1`)
            beta: elasticity parameter. should be 0 or None.
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        # Make sure beta = 0
        if beta is not None and not np.isclose(beta, 0.0):
            print(f"Ignoring beta = {beta}...")
        self.lam = lam
        super().__init__(sigma, vov, rho, beta=0, intr=intr, divr=divr, is_fwd=is_fwd)

    def set_num_params(self, n_path=1e6, rn_seed=None, antithetic=True):
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

        rhoc = np.sqrt(1 - self.rho**2)
        vol_var = self.vov**2 * texp
        vol_std = np.sqrt(vol_var)

        z_rn = self.rng.normal(size=(int(self.n_path / 2), 3))
        z_rn = np.stack([z_rn, -z_rn], axis=1).reshape((-1, 3))
        z_rn[:, 2] += 0.5 * (self.lam - 1) * vol_std  # add shift

        r2 = np.sum(z_rn[:, 0:2]**2, axis=1)
        exp_plus = np.exp(0.5 * vol_std * z_rn[:, 2])

        phi_r1 = np.sqrt(2/r2) * np.sqrt(np.cosh(np.sqrt(r2 + z_rn[:, 2]**2) * vol_std) - np.cosh(z_rn[:, 2] * vol_std))

        df_z = exp_plus**2
        # use both X and Y components
        df_w = (exp_plus[:, None] * z_rn[:, 0:2] * phi_r1[:, None])

        path = (
            df_z,
            (self.sigma / self.vov)
            * (
                self.rho * (df_z[:, None] - np.exp(0.5 * self.lam * vol_var))
                + rhoc * df_w
            ),
        )
        return path

    def price(self, strike, spot, texp, cp=1):
        """
        Vanilla option price from MC simulation of NSVh model.

        Args:
            strike: strike price
            spot: spot price
            texp: time to np.expiry
            cp: 1/-1 for call/put

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
        return df * price


class NsvhGaussQuad(sabr.SabrABC):
    """
    Quadrature integration method of Hyperbolic Normal Stochastic Volatility (NSVh) model.

    NSVh with lambda = 0 is the normal SABR model, and NSVh with lambda = 1 has analytic pricing (Nsvh1)

    References:
        - Choi J, Liu C, Seo BK (2019) Hyperbolic normal stochastic volatility model. J Futures Mark 39:186–204. https://doi.org/10.1002/fut.21967

    See Also:
        Nsvh1, SabrNormalVolApprox

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> #### Nsvh1: comparison with analytic pricing
        >>> m1 = pf.NsvhGaussQuad(sigma=20, vov=0.8, rho=-0.3, lam=1.0)
        >>> m2 = pf.Nsvh1(sigma=20, vov=0.8, rho=-0.3)
        >>> p1 = m1.price(np.arange(80, 121, 10), 100, 1.2)
        >>> p2 = m2.price(np.arange(80, 121, 10), 100, 1.2)
        >>> p1 - p2
        array([0.00345526, 0.00630649, 0.00966333, 0.00571175, 0.00017924])
        >>> #### Normal SABR: comparison with vol approximation
        >>> m1 = pf.NsvhGaussQuad(sigma=20, vov=0.8, rho=-0.3, lam=0.0)
        >>> m2 = pf.SabrNormVolApprox(sigma=20, vov=0.8, rho=-0.3)
        >>> p1 = m1.price(np.arange(80, 121, 10), 100, 1.2)
        >>> p2 = m2.price(np.arange(80, 121, 10), 100, 1.2)
        >>> p1 - p2
        array([-0.17262802, -0.10160687, -0.00802731,  0.0338126 ,  0.01598512])

    References:
        Choi J (2023), Option pricing under the normal SABR model with Gaussian quadratures. Unpublished Working Paper.

    """

    n_quad = (7, 7)

    def __init__(self, sigma, vov=0.0, rho=0.0, lam=0.0, beta=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            lam: lambda. Normal SABR if 0, Johnson's SU if 1 (same as `Nsvh1`)
            beta: elasticity parameter. should be 0 or None.
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        # Make sure beta = 0
        if beta is not None and not np.isclose(beta, 0.0):
            print(f"Ignoring beta = {beta}...")
        self.lam = lam
        super().__init__(sigma, vov, rho, beta=0, intr=intr, divr=divr, is_fwd=is_fwd)

    def price(self, strike, spot, texp, cp=1):

        fwd, df, _ = self._fwd_factor(spot, texp)
        _, _, rhoc, rho2, vovn = self._variables(1.0, texp)

        ### axis 1: nodes of x,y,z , get the weight of z,v
        z_value, z_weight = spsp.roots_hermitenorm(self.n_quad[0])
        z_weight /= np.sqrt(2 * np.pi)
        z_weight /= np.sum(np.exp(vovn/2*z_value)*z_weight) / np.exp(vovn**2/8)
        #print(np.sum(np.exp(-vovn/2*z_value)*z_weight) - np.exp(vovn**2/8))

        # quadrature point & weight for exp(-v/2)/(2 pi) derived from sqrt(v) * np.exp(-v)
        v_value, v_weight = spsp.roots_genlaguerre(self.n_quad[1], 0.5)
        v_weight /= np.pi * np.sqrt(v_value)
        v_value *= 2.0

        ### axis 0: dependence on v
        v_value = v_value[:, None]
        v_weight = v_weight[:, None]

        vov_var = np.exp(0.5 * self.lam * vovn**2)

        #### effective strike
        strike_eff = (self.vov/self.sigma) * (strike - fwd)
        scalar_output = np.isscalar(strike_eff)

        strike_eff, cp = np.broadcast_arrays(np.atleast_1d(strike_eff), cp)
        
        u_hat = (z_value + 0.5 * self.lam * vovn)  # column (z direction)
        exp_plus = np.exp(vovn * u_hat/2)
        z_star_cosh = (exp_plus**2 + 1/exp_plus**2)/2
        price = np.zeros_like(strike, dtype=float)
        
        for i, k_eff in enumerate(strike_eff):
            g_vec = self.rho * exp_plus - (self.rho * vov_var + k_eff) / exp_plus
            temp1 = z_star_cosh + 0.5 * g_vec**2 / (1 - rho2)
            v_0 = (np.arccosh(temp1) / vovn)**2 - u_hat**2
            h_mat = rhoc * np.sqrt(2*np.cosh(vovn * np.sqrt((u_hat**2 + v_0 + v_value))) - 2*np.cosh(vovn * u_hat))
            theta_mat = np.arccos(np.abs(g_vec) / h_mat)

            int_z_v = np.sqrt(h_mat**2 - g_vec**2) - np.abs(g_vec) * theta_mat

            int_z = np.sum(int_z_v * v_weight, axis=0)  # integrating over v (column)
            int_z[:] = int_z * np.exp(-v_0/2) + np.fmax(cp[i] * g_vec, 0.0)  # in-place operation

            price[i] = np.sum(int_z * z_weight)

        price *= np.exp((2*self.lam - 1)/8 * vovn**2) * (self.sigma/self.vov) * df

        if scalar_output:
            price = price[0]

        return price

    def cdf(self, strike, spot, texp, cp=-1):
        fwd = self.forward(spot, texp)
        _, _, rhoc, rho2, vovn = self._variables(1.0, texp)

        ### axis 1: nodes of x,y,z , get the weight of z,v
        z_value, z_weight = spsp.roots_hermitenorm(self.n_quad[0])
        z_weight /= np.sqrt(2 * np.pi)
        z_weight /= np.sum(np.exp(vovn/2*z_value)*z_weight) / np.exp(vovn**2/8)

        # quadrature point & weight for exp(-v/2)/(2 pi) derived from np.exp(-v)
        v_value, v_weight = spsp.roots_genlaguerre(self.n_quad[1], 0.0)
        v_weight /= np.pi
        v_value *= 2

        ### axis 0: dependence on v
        v_value = v_value[:, None]
        v_weight = v_weight[:, None]

        vov_var = np.exp(0.5 * self.lam * vovn ** 2)

        #### effective strike
        strike_eff = (self.vov / self.sigma) * (strike - fwd)
        scalar_output = np.isscalar(strike_eff)

        strike_eff, cp = np.broadcast_arrays(np.atleast_1d(strike_eff), cp)

        u_hat = (z_value + 0.5 * self.lam * vovn)  # column (z direction)
        exp_plus = np.exp(vovn * u_hat / 2)
        z_star_cosh = (exp_plus ** 2 + 1 / exp_plus ** 2) / 2
        cdf = np.zeros_like(strike, dtype=float)

        for i, k_eff in enumerate(strike_eff):
            g_vec = self.rho * exp_plus - (self.rho * vov_var + k_eff) / exp_plus
            temp1 = z_star_cosh + 0.5 * g_vec ** 2 / (1 - rho2)
            v_0 = (np.arccosh(temp1) / vovn) ** 2 - u_hat ** 2
            h_mat = rhoc * np.sqrt(
                2 * np.cosh(vovn * np.sqrt((u_hat ** 2 + v_0 + v_value))) - 2 * np.cosh(vovn * u_hat))
            theta_mat = np.arccos(np.abs(g_vec) / h_mat)

            int_z = np.sum(theta_mat * np.exp(-v_0/2) * v_weight, axis=0)  # integrating over v (column)
            int_z[cp[i]*g_vec > 0] = 1.0 - int_z[cp[i]*g_vec > 0]
            int_z *= np.exp(-z_value*vovn/2 - vovn**2/8)
            cdf[i] = np.sum(int_z * z_weight)

        if scalar_output:
            cdf = cdf[0]

        return cdf
