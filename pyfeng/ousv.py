import abc
import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv


class OusvSchobelZhu1998(sv.SvABC):
    """
    The implementation of Schobel & Zhu (1998)'s inverse FT pricing formula for European
    options the Ornstein-Uhlenbeck driven stochastic volatility process.

    References:
        - Schöbel, R., & Zhu, J. (1999). Stochastic Volatility With an Ornstein–Uhlenbeck Process: an Extension. Review of Finance, 3(1), 23–46. https://doi.org/10.1023/A:1009803506170

    Examples:
        >>> import pyfeng as pfex
        >>> model = pfex.OusvSchobelZhu1998(0.2, mr=4, vov=0.1, rho=-0.7, intr=0.09531)
        >>> model.price(100, 100, texp=np.array([1, 5, 10]))
        array([13.21493, 40.79773, 62.76312])
        >>> model = pfex.OusvSchobelZhu1998(0.25, mr=8, vov=0.3, rho=-0.6, intr=0.09531)
        >>> model.price(np.array([90, 100, 110]), 100, texp=1)
        array([21.41873, 15.16798, 10.17448])
    """

    var_process = False

    def D_B_C(self, s1, s2, s3, texp):
        # implement the formula for D(t,T), B(t,T), C(t,T) in paper appendix
        mr, theta, vov = self.mr, self.theta, self.vov

        gamma1 = np.sqrt(2 * vov**2 * s1 + mr**2)
        gamma2 = (mr - 2 * vov**2 * s3) / gamma1
        gamma3 = mr**2 * theta - s2 * vov**2
        sinh = np.sinh(gamma1 * texp)
        cosh = np.cosh(gamma1 * texp)
        sincos = sinh + gamma2 * cosh
        cossin = cosh + gamma2 * sinh
        ktg3 = mr * theta * gamma1 - gamma2 * gamma3
        s2g3 = vov**2 * gamma1**3

        D = (mr - gamma1 * sincos / cossin) / vov**2
        B = ((ktg3 + gamma3 * sincos) / cossin - mr * theta * gamma1) / (
            vov**2 * gamma1
        )
        C = (
            -0.5 * np.log(cossin)
            + 0.5 * mr * texp
            + ((mr * theta * gamma1)**2 - gamma3**2)
            / (2 * s2g3)
            * (sinh / cossin - gamma1 * texp)
            + ktg3 * gamma3 / s2g3 * ((cosh - 1) / cossin)
        )

        return D, B, C

    def f_1(self, phi, texp):
        # implement the formula (12)
        mr, theta, vov, rho = self.mr, self.theta, self.vov, self.rho

        tmp = 1 + 1j * phi
        s1 = 0.5 * tmp * (-tmp * (1 - rho**2) + (1 - 2 * mr * rho / vov))
        s2 = tmp * mr * theta * rho / vov
        s3 = 0.5 * tmp * rho / vov

        res = -0.5 * rho * tmp * (self.sigma**2 / vov + vov * texp)
        D, B, C = self.D_B_C(s1, s2, s3, texp)
        res += (D/2 * self.sigma + B) * self.sigma + C
        return np.exp(res)

    def f_2(self, phi, texp):
        # implement the formula (13)
        mr, theta, vov, rho = self.mr, self.theta, self.vov, self.rho

        s1 = 0.5 * phi * (phi * (1 - rho**2) + 1j * (1 - 2 * mr * rho / vov))
        s2 = 1j * phi * mr * theta * rho / vov
        s3 = 0.5 * 1j * phi * rho / vov

        res = -0.5 * 1j * phi * rho * (self.sigma**2 / vov + vov * texp)
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


class OusvMcABC(sv.SvABC, sv.CondMcBsmABC, abc.ABC):

    var_process = False

    @abc.abstractmethod
    def cond_states(self, sig_0, texp):
        """
        Final variance and integrated variance over dt given var_0
        The integrated variance is normalized by dt

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            (var_final, int_var_std)
        """
        pass

    def cond_spot_sigma(self, sig_0, texp):
        s_t, u_t_std, v_t_std = self.cond_states(sig_0, texp)

        fwd_cond = np.exp(
            self.rho * ((s_t**2 - self.sigma**2) / (2 * self.vov) - self.vov * texp / 2 \
            - (self.mr * self.theta / self.vov) * texp * u_t_std \
            + (self.mr / self.vov - self.rho / 2) * texp * v_t_std) \
        )

        if self.correct_fwd:
            fwd_err = np.mean(fwd_cond) - 1
            fwd_cond /= (1 + fwd_err)
        else:
            fwd_err = None

        sigma_cond = np.sqrt((1 - self.rho**2) * v_t_std) / self.sigma

        return fwd_cond, sigma_cond


class OusvMcTimeStep(OusvMcABC):
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

    def cond_states(self, sig_0, texp):
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        sigma_paths = self.vol_paths(tobs)
        s_t = sigma_paths[-1, :]
        u_t_std = scint.simps(sigma_paths, dx=1, axis=0) / n_dt
        v_t_std = scint.simps(sigma_paths**2, dx=1, axis=0) / n_dt

        return s_t, u_t_std, v_t_std


class OusvMcChoi2023(OusvMcABC):

    def set_mc_params(self, n_path=10000, n_sin=2, n_sin_max=None, rn_seed=None, antithetic=True):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step for Euler/Milstein steps
            rn_seed: random number seed
            antithetic: antithetic
        """
        assert n_sin % 2 == 0
        if n_sin_max is not None:
            assert n_sin_max % 2 == 0
        self.n_sin = n_sin
        self.n_sin_max = n_sin_max or n_sin

        super().set_mc_params(n_path, None, rn_seed, antithetic)

    @classmethod
    def _a2sum(cls, mr_t, ns=0, odd=None):
        if odd == 2:  # even
            rv = cls._a2sum(mr_t / 2) / 2**2
        elif odd == 1:  # odd
            rv = (mr_t / np.tanh(mr_t) - 1) / mr_t**2 - cls._a2sum(mr_t / 2) / 2**2
        else:  # all
            rv = (mr_t / np.tanh(mr_t) - 1) / mr_t**2

        if ns == 0:
            return rv

        n_pi_2 = (np.arange(1, ns + 1) * np.pi)**2
        a2 = 2 / (mr_t**2 + n_pi_2)

        if odd == 2:  # even
            rv -= np.sum(a2[1::2])
        elif odd == 1:  # odd
            rv -= np.sum(a2[::2])
        else:  # all
            rv -= np.sum(a2)
        return rv

    @classmethod
    def _a2overn2sum(cls, mr_t, ns=0, odd=None):
        if odd == 2:  # even
            rv = cls._a2overn2sum(mr_t / 2) / 2**4
        elif odd == 1:  # odd
            rv = (1/3 - (mr_t / np.tanh(mr_t) - 1) / mr_t**2) / mr_t**2 - cls._a2overn2sum(mr_t / 2) / 2**4
        else:  # all
            rv = (1/3 - (mr_t / np.tanh(mr_t) - 1) / mr_t**2) / mr_t**2

        if ns == 0:
            return rv

        n_pi_2 = (np.arange(1, ns + 1) * np.pi)**2
        a2overn2 = 2 / n_pi_2 / (mr_t**2 + n_pi_2)

        if odd == 2:  # even
            rv -= np.sum(a2overn2[1::2])
        elif odd == 1:  # odd
            rv -= np.sum(a2overn2[::2])
        else:  # all
            rv -= np.sum(a2overn2)
        return rv

    @classmethod
    def _a4sum(cls, mr_t, ns=0, odd=None):
        if odd == 2:  # even
            rv = cls._a4sum(mr_t / 2) / 2**4
        elif odd == 1:  # odd
            rv = (mr_t / np.tanh(mr_t) + mr_t**2 / np.sinh(mr_t)**2 - 2) / mr_t**4 - cls._a4sum(mr_t / 2) / 2**4
        else:  # all
            rv = (mr_t / np.tanh(mr_t) + mr_t**2 / np.sinh(mr_t)**2 - 2) / mr_t**4

        if ns == 0:
            return rv

        n_pi_2 = (np.arange(1, ns + 1) * np.pi)**2
        a4 = 4 / (mr_t**2 + n_pi_2)**2

        if odd == 2:  # even
            rv -= np.sum(a4[1::2])
        elif odd == 1:  # odd
            rv -= np.sum(a4[::2])
        else:  # all
            rv -= np.sum(a4)
        return rv

    @classmethod
    def _a6sum(cls, mr_t, ns=0, odd=None):
        if odd == 2:  # even
            rv = cls._a6sum(mr_t / 2) / 2**6
        elif odd == 1:  # odd
            rv = (3 * mr_t / np.tanh(mr_t) + (3 + 2 * mr_t / np.tanh(mr_t)) * mr_t**2 / np.sinh(mr_t)**2 - 8) / (
                        2 * mr_t**6) - cls._a6sum(mr_t / 2) / 2**6
        else:  # all
            rv = (3 * mr_t / np.tanh(mr_t) + (3 + 2 * mr_t / np.tanh(mr_t)) * mr_t**2 / np.sinh(mr_t)**2 - 8) / (
                        2 * mr_t**6)

        if ns == 0:
            return rv

        n_pi_2 = (np.arange(1, ns + 1) * np.pi)**2
        a6 = 8 / (mr_t**2 + n_pi_2)**3

        if odd == 2:  # even
            rv -= np.sum(a6[1::2])
        elif odd == 1:  # odd
            rv -= np.sum(a6[::2])
        else:  # all
            rv -= np.sum(a6)
        return rv

    @classmethod
    def _a6n2sum(cls, mr_t, ns=0, odd=None):
        if odd == 2:  # even
            rv = cls._a6n2sum(mr_t / 2) / 2**4
        elif odd == 1:  # odd
            rv = 2 * cls._a4sum(mr_t) - mr_t**2 * cls._a6sum(mr_t) - cls._a6n2sum(mr_t / 2) / 2**4
        else:  # all
            rv = 2 * cls._a4sum(mr_t) - mr_t**2 * cls._a6sum(mr_t)

        if ns == 0:
            return rv

        n_pi_2 = (np.arange(1, ns + 1) * np.pi)**2
        a6n2 = n_pi_2 * 8 / (mr_t**2 + n_pi_2)**3

        if odd == 2:  # even
            rv -= np.sum(a6n2[1::2])
        elif odd == 1:  # odd
            rv -= np.sum(a6n2[::2])
        else:  # all
            rv -= np.sum(a6n2)
        return rv


    def cond_states(self, sig_0, dt):
        n_sin_max = self.n_sin_max
        n_sin = self.n_sin
        n_path = self.n_path
        n_path_half = int(n_path//2)

        mr, vov, vinf = self.mr, self.vov, self.theta

        # random number for (time, path,
        zn = self.rng.standard_normal(size=(n_path_half, n_sin_max + 5))
        zn = np.stack([zn, -zn], axis=1).reshape((n_path, n_sin_max + 5))

        mr_t = mr * dt
        e_mr = np.exp(-mr_t)
        sinh = np.sinh(mr_t)
        cosh = np.cosh(mr_t)
        vovn = self.vov * np.sqrt(dt)  # normalized vov

        z_0 = zn[:, 0]
        z_g = zn[:, 1]
        z_p = zn[:, 2]
        z_q = zn[:, 3]
        z_r = zn[:, 4]
        z_sin = zn[:, 5:n_sin + 5]

        sighat = vov * np.sqrt(e_mr * sinh / mr) * z_0

        n_pi = np.pi * np.arange(1, n_sin + 1)
        an2 = 2 / (mr_t**2 + n_pi**2)  ## Careful: an contains texp in sqrt
        an = np.sqrt(an2)
        an3_n_pi = an2 * an * n_pi

        g_std = np.sqrt(self._a2overn2sum(mr_t, ns=n_sin, odd=1))
        p_std = np.sqrt(self._a6n2sum(mr_t, ns=n_sin, odd=1))
        q_std = np.sqrt(self._a6n2sum(mr_t, ns=n_sin, odd=2))  # even
        corr = self._a4sum(mr_t, ns=n_sin, odd=1) / (g_std * p_std)

        z_g = corr * z_p + np.sqrt(1 - corr**2) * z_g
        z_g *= g_std
        z_p *= p_std
        z_q *= q_std

        z_g += z_sin[:, ::2] @ (an[::2] / n_pi[::2])
        z_p += z_sin[:, ::2] @ an3_n_pi[::2]
        z_q += z_sin[:, 1::2] @ an3_n_pi[1::2]

        UT = vinf + (sig_0 - vinf) * (1 - e_mr) / mr_t  # * dt
        UT += (cosh - 1) / (mr_t * sinh) * sighat + 2 * vovn * z_g  # * dt

        VT = vinf * (2 * UT - vinf) + (sig_0 - vinf)**2 * (1 - e_mr**2) / (2 * mr_t)
        VT += sighat * (sig_0 - vinf) * (1 / sinh - e_mr / mr_t) + (sig_0 - vinf) * vovn * (
                    (1 + e_mr) * z_p + (1 - e_mr) * z_q)

        ## VT: even terms
        VT += (sinh * cosh - mr_t) / (2 * mr_t * sinh**2) * sighat**2 + sighat * vovn * (z_p - z_q)

        # LN variate (even term)
        m1 = self._a2sum(mr_t, ns=n_sin)
        var = 2 * self._a4sum(mr_t, ns=n_sin)
        ln_sig = np.sqrt(np.log(1 + var / m1**2))
        VT += 0.5 * vovn**2 * (z_sin**2 @ an**2 + m1 * np.exp(ln_sig * (z_r - 0.5 * ln_sig)))

        sighat += vinf + (sig_0 - vinf) * e_mr

        return sighat, UT, VT
