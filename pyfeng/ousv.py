import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv


class OusvIFT(sv.SvABC):
    """
    The implementation of Schobel & Zhu (1998)'s inverse FT pricing formula for European
    options the Ornstein-Uhlenbeck driven stochastic volatility process.

    References:
        - Schöbel, R., & Zhu, J. (1999). Stochastic Volatility With an Ornstein–Uhlenbeck Process: an Extension. Review of Finance, 3(1), 23–46. https://doi.org/10.1023/A:1009803506170

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

        bm_path = self._bm_incr(exp_tobs**2 - 1, cum=True, rng_index=0)  # B_s (0 <= s <= 1)
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


class OusvExactMC(sv.SvABC, sv.CondMcBsmABC):


    # rng_index: 0 for vol_path, 1 for sin, 2 for price
    int_sig = None
    int_var = None
    sighat = None

    vol_paths = OusvCondMC.vol_paths

    def set_mc_params(self, n_path=10000, dt=0.05, n_sin=2, rn_seed=None):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step for Euler/Milstein steps
            rn_seed: random number seed
            antithetic: antithetic
        """
        self.n_sin = n_sin
        super().set_mc_params(n_path, dt, rn_seed, True)

    @staticmethod
    def _a2sum(mr_t, ns=0, odd=None):
        if odd == 2:  # even
            rv = OusvExactMC._a2sum(mr_t / 2) / 2 ** 2
        elif odd == 1:  # odd
            rv = (mr_t / np.tanh(mr_t) - 1) / mr_t ** 2 - OusvExactMC._a2sum(mr_t / 2) / 2 ** 2
        else:  # all
            rv = (mr_t / np.tanh(mr_t) - 1) / mr_t ** 2

        if ns == 0:
            return rv

        n_pi_2 = (np.arange(1, ns + 1) * np.pi) ** 2
        a2 = 2 / (mr_t ** 2 + n_pi_2)

        if odd == 2:  # even
            rv -= np.sum(a2[1::2])
        elif odd == 1:  # odd
            rv -= np.sum(a2[::2])
        else:  # all
            rv -= np.sum(a2)
        return rv

    @staticmethod
    def _a2overn2sum(mr_t, ns=0, odd=None):
        if odd == 2:  # even
            rv = OusvExactMC._a2overn2sum(mr_t / 2) / 2 ** 4
        elif odd == 1:  # odd
            rv = (1 / 3 - (mr_t / np.tanh(mr_t) - 1) / mr_t ** 2) / mr_t ** 2 - OusvExactMC._a2overn2sum(mr_t / 2) / 2 ** 4
        else:  # all
            rv = (1 / 3 - (mr_t / np.tanh(mr_t) - 1) / mr_t ** 2) / mr_t ** 2

        if ns == 0:
            return rv

        n_pi_2 = (np.arange(1, ns + 1) * np.pi) ** 2
        a2overn2 = 2 / n_pi_2 / (mr_t ** 2 + n_pi_2)

        if odd == 2:  # even
            rv -= np.sum(a2overn2[1::2])
        elif odd == 1:  # odd
            rv -= np.sum(a2overn2[::2])
        else:  # all
            rv -= np.sum(a2overn2)
        return rv

    @staticmethod
    def _a4sum(mr_t, ns=0, odd=None):
        if odd == 2:  # even
            rv = OusvExactMC._a4sum(mr_t / 2) / 2 ** 4
        elif odd == 1:  # odd
            rv = (mr_t / np.tanh(mr_t) + mr_t ** 2 / np.sinh(mr_t) ** 2 - 2) / mr_t ** 4 - OusvExactMC._a4sum(mr_t / 2) / 2 ** 4
        else:  # all
            rv = (mr_t / np.tanh(mr_t) + mr_t ** 2 / np.sinh(mr_t) ** 2 - 2) / mr_t ** 4

        if ns == 0:
            return rv

        n_pi_2 = (np.arange(1, ns + 1) * np.pi) ** 2
        a4 = 4 / (mr_t ** 2 + n_pi_2) ** 2

        if odd == 2:  # even
            rv -= np.sum(a4[1::2])
        elif odd == 1:  # odd
            rv -= np.sum(a4[::2])
        else:  # all
            rv -= np.sum(a4)
        return rv

    @staticmethod
    def _a6sum(mr_t, ns=0, odd=None):
        if odd == 2:  # even
            rv = OusvExactMC._a6sum(mr_t / 2) / 2 ** 6
        elif odd == 1:  # odd
            rv = (3 * mr_t / np.tanh(mr_t) + (3 + 2 * mr_t / np.tanh(mr_t)) * mr_t ** 2 / np.sinh(mr_t) ** 2 - 8) / (
                        2 * mr_t ** 6) - OusvExactMC._a6sum(mr_t / 2) / 2 ** 6
        else:  # all
            rv = (3 * mr_t / np.tanh(mr_t) + (3 + 2 * mr_t / np.tanh(mr_t)) * mr_t ** 2 / np.sinh(mr_t) ** 2 - 8) / (
                        2 * mr_t ** 6)

        if ns == 0:
            return rv

        n_pi_2 = (np.arange(1, ns + 1) * np.pi) ** 2
        a6 = 8 / (mr_t ** 2 + n_pi_2) ** 3

        if odd == 2:  # even
            rv -= np.sum(a6[1::2])
        elif odd == 1:  # odd
            rv -= np.sum(a6[::2])
        else:  # all
            rv -= np.sum(a6)
        return rv

    @staticmethod
    def _a6n2sum(mr_t, ns=0, odd=None):
        if odd == 2:  # even
            rv = OusvExactMC._a6n2sum(mr_t / 2) / 2 ** 4
        elif odd == 1:  # odd
            rv = 2 * OusvExactMC._a4sum(mr_t) - mr_t ** 2 * OusvExactMC._a6sum(mr_t) - OusvExactMC._a6n2sum(mr_t / 2) / 2 ** 4
        else:  # all
            rv = 2 * OusvExactMC._a4sum(mr_t) - mr_t ** 2 * OusvExactMC._a6sum(mr_t)

        if ns == 0:
            return rv

        n_pi_2 = (np.arange(1, ns + 1) * np.pi) ** 2
        a6n2 = n_pi_2 * 8 / (mr_t ** 2 + n_pi_2) ** 3

        if odd == 2:  # even
            rv -= np.sum(a6n2[1::2])
        elif odd == 1:  # odd
            rv -= np.sum(a6n2[::2])
        else:  # all
            rv -= np.sum(a6n2)
        return rv


    def cond_states(self, tobs, n_path, n_sin, n_sin_max=None):
        assert n_sin // 2 * 2 == n_sin
        n_sin_max = n_sin_max or n_sin

        n_path_half = int(n_path//2)
        mr, vov, vinf = self.mr, self.vov, self.theta

        dt_arr = np.diff(tobs, prepend=0)
        n_dt = dt_arr.shape[0]

        self.sigma_t = self.vol_paths(tobs)
        self.int_sig = np.zeros((n_dt, n_path_half))
        self.int_var = np.zeros((n_dt, n_path))

        sig0 = self.sigma

        # random number for (time, path,
        zn = self.rng_array[1].standard_normal(size=(int(n_path//2), n_dt, n_sin_max + 5)).transpose((1, 0, 2))
        zn = np.stack([zn, -zn], axis=2).reshape(n_dt, 2*n_path, -1)

        for (i, dt) in enumerate(dt_arr):
            # scalars
            mr_t = mr * dt
            e_mr = np.exp(-mr_t)
            sinh = np.sinh(mr_t)
            cosh = np.cosh(mr_t)

            z_g = zn[i, :, 0]
            z_p = zn[i, :, 1]
            z_q = zn[i, :, 2]
            z_r = zn[i, :, 3]
            z_sin = zn[i, :, 4:n_sin + 4]

            self.sighat[i, :] = sighat = vov * np.sqrt(e_mr * sinh / mr) * z0

            n_pi = np.pi * np.arange(1, n_sin + 1)
            an2 = 2 / (mr_t ** 2 + n_pi ** 2)  ## Careful: an contains texp in sqrt
            an = np.sqrt(an2)
            an3_n_pi = an2 * an * n_pi

            g_std = np.sqrt(OusvExactMC._a2overn2sum(mr_t, ns=n_sin, odd=1))
            p_std = np.sqrt(OusvExactMC._a6n2sum(mr_t, ns=n_sin, odd=1))
            q_std = np.sqrt(OusvExactMC._a6n2sum(mr_t, ns=n_sin, odd=2))  # even
            corr = OusvExactMC._a4sum(mr_t, ns=n_sin, odd=1) / (g_std * p_std)

            z_g = corr * z_p + np.sqrt(1 - corr ** 2) * z_g
            z_g *= g_std
            z_p *= p_std
            z_q *= q_std
            z_g += z_sin[:, ::2] @ (an[::2] / n_pi[::2])
            z_p += z_sin[:, ::2] @ an3_n_pi[::2]
            z_q += z_sin[:, 1::2] @ an3_n_pi[1::2]

            UT = (cosh - 1) / (mr * sinh) * sighat + 2 * vov * dt * np.sqrt(dt) * z_g
            UT = vinf * dt + (sig0 - vinf) * (1 - e_mr) / mr + np.array([UT, -UT])

            # VT: odd terms
            VT = sighat * (sig0 - vinf) * (dt / sinh - e_mr / mr) + (sig0 - vinf) * vov * dt * np.sqrt(dt) * (
                        (1 + e_mr) * z_p + (1 - e_mr) * z_q)
            VT = 2 * vinf * UT + np.array([VT, -VT])

            ## VT: constant terms
            VT += -vinf ** 2 * dt + (sig0 - vinf) ** 2 * (1 - e_mr ** 2) / (2 * mr)
            ## VT: even terms
            VT += (sinh * cosh - mr_t) / (2 * mr * sinh ** 2) * sighat ** 2 + sighat * vov * dt * np.sqrt(dt) * (
                        z_p - z_q)

            # LN variate (even term)
            m1 = OusvExactMC._a2sum(mr_t, ns=n_sin)
            var = 2 * OusvExactMC._a4sum(mr_t, ns=n_sin)
            ln_sig = np.sqrt(np.log(1 + var / m1 ** 2))
            VT += 0.5 * (dt * vov) ** 2 * (
                        z_sin ** 2 @ an ** 2 + m1 * np.exp(ln_sig * (np.array([z_r, -z_r]) - 0.5 * ln_sig)))

            sighat = vinf + (sig0 - vinf) * e_mr + np.array([sighat, -sighat])

        return sighat.flatten('F'), UT.flatten('F'), VT.flatten('F')

    def cond_fwd_vol(self, texp):
        n_sin = np.min(np.int(self.n_sin * texp / 2) * 2, 2)
        s_t, u_t, v_t = self.cond_states(texp, self.n_path, n_sin)

        fwd_cond = np.exp(
            self.rho * ((s_t ** 2 - self.sigma ** 2) / (2 * self.vov) - self.vov * texp / 2 \
            - (self.mr * self.theta / self.vov) * u_t + (self.mr / self.vov - self.rho / 2) * v_t) \
        )
        if self.correct:
            fwd_err = np.mean(fwd_cond) - 1
            fwd_cond /= (1 + fwd_err)
        else:
            fwd_err = None

        vol_cond = np.sqrt((1-self.rho ** 2) * v_t / texp / self.sigma**2)

        return fwd_cond, vol_cond

    def price_paths(self, tobs):
        price = np.zeros((len(tobs)+1, self.n_path))
        dt_arr = np.diff(np.atleast_1d(tobs), prepend=0)
        s_0 = self.sigma

        price[0, :] = s_0
        for k, dt in enumerate(dt_arr):
            s_t, u_t, v_t = self.cond_states(s_0, dt)

            xx = np.random.normal(int(self.n_path // 2))
            xx = np.array([xx, -xx]).flatten('F')

            price[k+1, :] = (self.intr - self.vov/2)*dt + self.rho*((s_t ** 2 - s_0 ** 2) / (2 * self.vov) \
                - (self.mr * self.theta / self.vov) * u_t + (self.mr / self.vov - 0.5 / self.rho) * v_t) \
                + np.sqrt((1-self.rho**2)*v_t) * xx
            s_0 = s_t

        np.exp(price, out=price)

        return price


    def price_variance_swap(self, tobs):
        p_path = self.price_paths(self, tobs)
