import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv


class OusvIFT(sv.SvABC):
    """
    The implementation of Schobel & Zhu (1998)'s inverse FT pricing formula for European
    options the Ornstein-Uhlenbeck driven stochastic volatility process.

    References: Schöbel, R., & Zhu, J. (1999). Stochastic Volatility With an Ornstein–Uhlenbeck Process: An Extension. Review of Finance, 3(1), 23–46. https://doi.org/10.1023/A:1009803506170

    Examples:
        >>> import pyfeng as pf
        >>> model = pf.OusvIFT(0.2, mr=4, vov=0.1, rho=-0.7, intr=0.09531)
        >>> model.price(100, 100, texp=np.array([1, 5, 10]))
        array([13.21493, 40.79773, 62.76312])
        >>> model = pf.OusvIFT(0.25, mr=8, vov=0.3, rho=-0.6, intr=0.09531)
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
        B = ((ktg3 + gamma3 * sincos) / cossin - mr * theta * gamma1) / (
            vov ** 2 * gamma1
        )
        C = (
            -0.5 * np.log(cossin)
            + 0.5 * mr * texp
            + ((mr * theta * gamma1) ** 2 - gamma3 ** 2)
            / (2 * s2g3)
            * (sinh / cossin - gamma1 * texp)
            + ktg3 * gamma3 / s2g3 * ((cosh - 1) / cossin)
        )

        return D, B, C

    def f_1(self, phi, texp):
        # implement the formula (12)
        mr, theta, vov, rho = self.mr, self.theta, self.vov, self.rho

        tmp = 1 + 1j * phi
        s1 = 0.5 * tmp * (-tmp * (1 - rho ** 2) + (1 - 2 * mr * rho / vov))
        s2 = tmp * mr * theta * rho / vov
        s3 = 0.5 * tmp * rho / vov

        res = -0.5 * rho * tmp * (self.sigma ** 2 / vov + vov * texp)
        D, B, C = self.D_B_C(s1, s2, s3, texp)
        res += (D/2 * self.sigma + B) * self.sigma + C
        return np.exp(res)

    def f_2(self, phi, texp):
        # implement the formula (13)
        mr, theta, vov, rho = self.mr, self.theta, self.vov, self.rho

        s1 = 0.5 * phi * (phi * (1 - rho ** 2) + 1j * (1 - 2 * mr * rho / vov))
        s2 = 1j * phi * mr * theta * rho / vov
        s3 = 0.5 * 1j * phi * rho / vov

        res = -0.5 * 1j * phi * rho * (self.sigma ** 2 / vov + vov * texp)
        D, B, C = self.D_B_C(s1, s2, s3, texp)
        res += (D/2 * self.sigma + B) * self.sigma + C
        return np.exp(res)

    def price(self, strike, spot, texp, cp=1):
        # implement the formula (14) and (15)
        fwd, df, _ = self._fwd_factor(spot, texp)

        kk = strike / fwd
        log_k = np.log(kk)
        J, h = 100001, 0.001   # need to take these as parameters
        phi = (np.arange(J)[:, None] + 1) * h  # shape=(J,1)

        ff = self.f_1(phi, texp) - kk * self.f_2(phi, texp)

        ## Need to convert using iFFT later
        price = scint.simps(
            (ff * np.exp(-1j * phi * log_k) / (1j * phi)).real,
            dx=h, axis=0,
        ) / np.pi

        price += (1 - kk) / 2 * np.where(cp > 0, 1, -1)

        if len(price) == 1:
            price = price[0]

        return df * fwd * price


class OusvCondMC(sv.SvABC, sv.CondMcBsmABC):
    """
    OUSV model with conditional Monte-Carlo simulation
    The SDE of SV is: d sigma_t = mr (theta - sigma_t) dt + vov dB_T
    """

    def vol_paths(self, tobs):
        # 2d array of (time, path) including t=0
        exp_tobs = np.exp(self.mr * tobs)

        bm_path = self._bm_incr(exp_tobs**2 - 1, cum=True)  # B_s (0 <= s <= 1)
        sigma_t = self.theta + (
            self.sigma - self.theta + self.vov / np.sqrt(2 * self.mr) * bm_path
        ) / exp_tobs[:, None]
        sigma_t = np.insert(sigma_t, 0, self.sigma, axis=0)
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
        int_sigma = scint.simps(sigma_paths, dx=texp / n_dt, axis=0)
        int_var = scint.simps(sigma_paths ** 2, dx=texp / n_dt, axis=0)

        fwd_cond = np.exp(
            self.rho
            * (
                (sigma_final ** 2 - self.sigma ** 2) / (2 * self.vov)
                - self.vov * texp / 2
                - self.mr * self.theta / self.vov * int_sigma
                + (self.mr / self.vov - self.rho / 2) * int_var
            )
        )  # scaled by initial value

        # scaled by initial volatility
        vol_cond = rhoc * np.sqrt(int_var) / (self.sigma * np.sqrt(texp))

        return fwd_cond, vol_cond
