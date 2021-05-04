import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv


class OusvIft(sv.SvABC):
    """
    The implementation of Schobel & Zhu (1998)'s inverse FT pricing formula for European
    options the Ornstein-Uhlenbeck driven stochastic volatility process.

    References: Schöbel, R., & Zhu, J. (1999). Stochastic Volatility With an Ornstein–Uhlenbeck Process: An Extension. Review of Finance, 3(1), 23–46. https://doi.org/10.1023/A:1009803506170

    Examples:
        >>> import pyfeng as pf
        >>> model = pf.OusvIft(0.2, mr=4, vov=0.1, rho=-0.7, intr=0.09531)
        >>> model.price(100, 100, texp=np.array([1, 5, 10]))
        array([13.21493, 40.79773, 62.76312])
        >>> model = pf.OusvIft(0.25, mr=8, vov=0.3, rho=-0.6, intr=0.09531)
        >>> model.price(np.array([90, 100, 110]), 100, texp=1)
        array([21.41873, 15.16798, 10.17448])
    """

    def D_B_C(self, s1, s2, s3, texp):
        # implement the formula for D(t,T), B(t,T), C(t,T) in paper appendix
        mr, theta, vov = self.mr, self.theta, self.vov

        gamma1 = np.sqrt(2 * vov ** 2 * s1 + mr ** 2)
        gamma2 = (mr - 2 * vov ** 2 * s3) / gamma1
        gamma3 = mr ** 2 * theta - s2 * vov ** 2
        sinh = np.sinh(gamma1 * texp)
        cosh = np.cosh(gamma1 * texp)
        sincos = sinh + gamma2 * cosh
        cossin = cosh + gamma2 * sinh
        ktg3 = mr * theta * gamma1 - gamma2 * gamma3
        s2g3 = vov ** 2 * gamma1 ** 3

        D = (mr - gamma1 * sincos / cossin) / vov ** 2
        B = ((ktg3 + gamma3 * sincos) / cossin - mr * theta * gamma1) / (vov ** 2 * gamma1)
        C = -0.5 * np.log(cossin) + 0.5 * mr * texp + ((mr * theta * gamma1) ** 2 - gamma3 ** 2) / (2 * s2g3) * (
                sinh / cossin - gamma1 * texp) \
            + ktg3 * gamma3 / s2g3 * ((cosh - 1) / cossin)

        return D, B, C

    def f_1(self, phi, fwd, texp):
        # implement the formula (12)
        mr, theta, vov, rho = self.mr, self.theta, self.vov, self.rho
        tmp = 1 + 1j * phi
        s1 = 0.5 * tmp * (-tmp * (1 - rho ** 2) + (1 - 2 * mr * rho / vov))
        s2 = tmp * mr * theta * rho / vov
        s3 = 0.5 * tmp * rho / vov

        res = 1j * phi * np.log(fwd) - 0.5 * rho * (1 + 1j * phi) * (self.sigma ** 2 / vov + vov * texp)
        D, B, C = self.D_B_C(s1, s2, s3, texp)
        res += 0.5 * D * self.sigma ** 2 + B * self.sigma + C
        return np.exp(res)

    def f_2(self, phi, fwd, texp):
        # implement the formula (13)
        mr, theta, vov, rho = self.mr, self.theta, self.vov, self.rho

        s1 = 0.5 * (phi ** 2 * (1 - rho ** 2) + 1j * phi * (1 - 2 * mr * rho / vov))
        s2 = 1j * phi * mr * theta * rho / vov
        s3 = 0.5 * 1j * phi * rho / vov

        res = 1j * phi * np.log(fwd) - 0.5 * 1j * phi * rho * (self.sigma ** 2 / vov + vov * texp)
        D, B, C = self.D_B_C(s1, s2, s3, texp)
        res += 0.5 * D * self.sigma ** 2 + B * self.sigma + C
        return np.exp(res)

    def price(self, strike, spot, texp, cp=1):
        # implement the formula (14) and (15)
        fwd, df, _ = self._fwd_factor(spot, texp)

        log_k = np.log(strike)
        J, h = 100001, 0.001
        phi = (np.arange(J)[:, None] + 1) * h  # shape=(J,1)
        ff1 = 0.5 + 1 / np.pi * scint.simps((self.f_1(phi, fwd, texp) * np.exp(-1j * phi * log_k) / (1j * phi)).real,
                                            dx=h, axis=0)
        ff2 = 0.5 + 1 / np.pi * scint.simps((self.f_2(phi, fwd, texp) * np.exp(-1j * phi * log_k) / (1j * phi)).real,
                                            dx=h, axis=0)

        price = np.where(cp > 0, fwd * ff1 - strike * ff2, strike * (1 - ff2) - fwd * (1 - ff1))
        if len(price) == 1:
            price = price[0]

        return df * price


class OusvCondMC(sv.SvABC, sv.CondMcBsmABC):
    """
        OUSV model with conditional Monte-Carlo simulation
        The SDE of SV is: dsigma_t = mr (theta - sigma_t) dt + vov dB_T
        """

    def _bm_incr(self, tobs, cum=False, n_path=None):
        """
        Calculate incremental Brownian Motions

        Args:
            tobs: observation times (array). 0 is not included.
            cum: return cumulative values if True
            n_path: number of paths. If None (default), use the stored one.

        Returns:
            price path (time, path)
        """
        # dt = np.diff(np.atleast_1d(tobs), prepend=0)
        n_dt = len(tobs)

        tobs_lag = tobs[:-1]
        tobs_lag = np.insert(tobs_lag, 0, 0)
        bm_var = np.exp(2 * self.mr * tobs) - np.exp(2 * self.mr * tobs_lag)
        n_path = n_path or self.n_path

        if self.antithetic:
            # generate random number in the order of path, time, asset and transposed
            # in this way, the same paths are generated when increasing n_path
            bm_incr = self.rng.normal(size=(int(n_path / 2), n_dt)).T * np.sqrt(bm_var[:, None])
            bm_incr = np.stack([bm_incr, -bm_incr], axis=-1).reshape((-1, n_path))
        else:
            # bm_incr = np.random.randn(n_path, n_dt).T * np.sqrt(bm_var[:, None])
            bm_incr = self.rng.normal(size=(n_path, n_dt)).T * np.sqrt(bm_var[:, None])

        if cum:
            np.cumsum(bm_incr, axis=0, out=bm_incr)

        return bm_incr

    def vol_paths(self, tobs):
        """
        sigma_t = np.exp(-mr * tobs) * (sigma0 - theta * mr + vov / np.sqrt(2 * mr) * bm) + theta * mr
        Args:
            tobs: observation time (array)
            mr: coefficient of dt
            theta: the long term average
            mu: rn-derivative

        Returns: volatility path (time, path) including the value at t=0
        """
        bm_path = self._bm_incr(tobs, cum=True)  # B_s (0 <= s <= 1)
        sigma_t = self.theta + (self.sigma - self.theta + self.vov / np.sqrt(2 * self.mr) * bm_path) * np.exp(
            -self.mr * tobs[:, None])
        sigma_t = np.insert(sigma_t, 0, np.array([self.sigma] * sigma_t.shape[1]), axis=0)
        return sigma_t

    def cond_fwd_vol(self, texp):
        """
            Returns new forward and volatility conditional on volatility path (e.g., sigma_T, integrated variance)
            The forward and volatility are standardized in the sense that F_0 = 1 and sigma_0 = 1
            Therefore, they should be scaled by the original F_0 and sigma_0 values

            Args:
                theta: the long term average
                mr: coefficient of dt
                texp: time-to-expiry

            Returns: (forward, volatility)
        """
        rhoc = np.sqrt(1.0 - self.rho ** 2)
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        sigma_paths = self.vol_paths(tobs)
        sigma_final = sigma_paths[-1, :]
        int_sigma = scint.simps(sigma_paths, dx=texp/n_dt, axis=0)
        int_var = scint.simps(sigma_paths ** 2, dx=texp/n_dt, axis=0)

        fwd_cond = np.exp(self.rho * ((sigma_final ** 2 - self.sigma ** 2) / (2 * self.vov) - self.vov * 0.5 * texp -
                                      self.mr * self.theta / self.vov * int_sigma +
                                      (self.mr / self.vov - self.rho * 0.5) * int_var))  # scaled by initial value

        vol_cond = rhoc * np.sqrt(int_var / texp)

        return fwd_cond, vol_cond

    def price(self, strike, spot, texp, cp=1):
        """
        Calculate option price based on BSM
        Args:
            strike: strike price
            spot: spot price
            texp: time to maturity
            cp: cp=1 if call option else put option

        Returns: price
        """
        price = []
        texp = [texp] if isinstance(texp, (int, float)) else texp
        for t in texp:
            kk = strike / spot
            kk = np.atleast_1d(kk)

            fwd_cond, vol_cond = self.cond_fwd_vol(t)

            base_model = self.base_model(vol_cond)
            price_grid = base_model.price(kk[:, None], fwd_cond, texp=t, cp=cp)

            #np.set_printoptions(suppress=True, precision=6)
            price.append(spot * np.mean(price_grid, axis=1))  # in cond_fwd_vol, S_0 = 1

        return np.array(price).T
