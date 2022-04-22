import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv
import math
import abc

class OusvSteinStein1991(sv.SvABC):
    """
    The implementation of Stein & Stein (1991)'s use of analytic techniques to derive an explicit closed-form solution
    for the case that volatility is driven by an OU process.

    References:
    Examples:
        >>> import pyfeng as pf
        >>> model = pf.OusvSteinStein1991(sigma=0.25, mr=16, vov=0.4, intr=0.0953)
        >>> model.price(100, 100, texp=0.5)
        8.35948
    """

    var_process = False

    def MGF(self, l, texp):
        #Implement the formula of the moment generation function
        A = - self.mr / np.power(self.vov, 2)
        B = self.theta * self.mr / np.power(self.vov, 2)
        C = - l / (np.power(self.vov, 2) * texp)
        a = np.sqrt(np.power(A, 2) - 2 * C)
        b = - A / a
        alpha = a * np.power(self.vov, 2) * texp
        if np.sinh(2 * alpha) != np.inf and np.cosh(2 * alpha) != np.inf:
            L = - A - a * (np.sinh(alpha) + b * np.cosh(alpha)) / (np.cosh(alpha) + b * np.sinh(alpha))
            M = B * ((b * np.sinh(alpha) + np.power(b, 2) * np.cosh(alpha) + 1 - np.power(b, 2)) / (np.cosh(alpha) + b * np.sinh(alpha)) - 1)
            N_1 = (a - A) * (np.power(a, 2) - A * np.power(B, 2) - a * np.power(B, 2)) * np.power(self.vov, 2) * texp / (2 * np.power(a, 2))
            N_2 = np.power(B, 2) * (np.power(A, 2) - np.power(a, 2)) / (2 * np.power(a, 3))
            N_3 = (2 * A + a + (2 * A - a) * np.exp(2 * alpha)) / (A + a + (A - a) * np.exp(2 * alpha))
            N_4 = 2 * A * np.power(B, 2) * (np.power(a, 2) - np.power(A, 2)) * np.exp(alpha) / (np.power(a, 3) * (A + a + (a - A) * np.exp(2 * alpha)))
            N_5 = 0.5 * np.log(0.5 * (A / a + 1) + 0.5 * (1 - A / a) * np.exp(2 * alpha))
            N = N_1 + N_2 * N_3 + N_4 - N_5
            return math.exp(L * np.power(self.sigma, 2) / 2 + M * self.sigma + N)
        else:
            return 0

    def density_func_original(self, price, spot, texp, cp=1):
        #implement formula (9) for the special case mu = 0
        mu = self.intr - self.divr

        def func_1(eta):
            return self.MGF((eta * eta + 0.25) * texp / 2, texp) * np.cos(np.log(price) * eta)

        f,err = scint.quad(func_1, -np.inf, np.inf)

        return f * np.power(price, -1.5) / (2 * np.pi)

    def density_func(self, price, spot, texp, cp=1):
        #implement the case mu != 0
        mu = self.intr - self.divr

        return self.density_func_original(price * np.exp(- mu * texp), spot, texp, cp=1) * np.exp(- mu * texp)

    def price(self, strike, spot, texp, cp=1):
        #implement the pricing formula (15)

        def func_2(P):
            return (P - strike) * self.density_func(P / spot, spot, texp) / spot

        price,err = scint.quad(func_2, strike, np.inf)
        price *= np.exp(-self.intr * texp)

        return price

class HestonUncorrBallRoma1994(sv.SvABC):
    """
    The implementation of Ball & Roma (1994)'s stochastic volatility pricing formula for European
    options when there is no correlation between innovations in security prices and volatility.

    References: Ball, C. A., & Roma, A. (1994). Stochastic Volatility Option Pricing. The Journal of Financial and Quantitative Analysis, 29(4), 589–607. https://doi.org/10.2307/2331111
    Examples:
        >>> import pyfeng as pf
        >>> model = pf.OusvBallRoma1994(sigma=0.3, mr=4, vov=0.4, intr=0)
        >>> model.price(100, 100, texp=0.5)
        8.35948
    """

    var_process = False

    def MGF(self, l, texp):
        v = l * self.sigma / texp

        gamma = np.sqrt(self.mr * self.mr + 2 * l * self.vov * self.vov / texp)

        g = 2 * gamma + (self.mr - gamma) * (1 - np.exp(- gamma * texp))

        m = (-2) * (1 - np.exp(- gamma * texp)) / g
        n = 2 * self.mr * self.theta * np.log(2 * gamma * np.exp((self.mr - gamma) * texp / 2) / g) /  (self.vov * self.vov)

        return np.exp(n + m * v)

    def first_derivative(self, texp):
        var0, mr, vov, theta = self.sigma, self.mr, self.vov, self.theta

        mr2 = mr * mr
        vov2 = vov * vov
        theta2 = theta * theta
        decay = np.exp(-mr * texp)

        #M_1 = (- self.sigma + np.exp(self.mr * texp) * self.sigma - self.mr * np.exp(self.mr * texp) * self.sigma * texp + self.theta - np.exp(self.mr * texp) * self.theta) / (self.mr * np.exp(self.mr * texp) * texp)
        M_1 = theta + (var0 - theta) * (1 - decay) / (mr * texp)
        return M_1

    def second_derivative(self, texp):
        var0, mr, vov, theta = self.sigma, self.mr, self.vov, self.theta

        mr2 = mr * mr
        vov2 = vov * vov
        theta2 = theta * theta
        decay = np.exp(-mr * texp)

        term1 = vov2 - mr
        term2 = vov2 - 2 * mr

        #M_21 = np.power(self.sigma, 2)
        #M_22 = (np.power(self.sigma, 2) / np.power(self.mr, 2) + np.power(self.sigma, 2) / (np.power(self.mr, 2) * np.exp(2 * self.mr * texp)) - 2 * np.power(self.sigma, 2) / (np.power(self.mr, 2) * np.exp(self.mr * texp)) - 2 * self.sigma * self.theta / np.power(self.mr, 2) - 2 * self.sigma * self.theta / (np.power(self.mr, 2) * np.exp(2 * self.mr * texp)) + 4 * self.sigma * self.theta / (np.power(self.mr, 2) * np.exp(self.mr * texp)) + np.power(self.theta, 2) / np.power(self.mr, 2)) / np.power(texp, 2)
        #M_23 = (np.power(self.theta, 2) / (np.power(self.mr, 2) * np.exp(2 * self.mr * texp)) - 2 * np.power(self.theta, 2) / (np.power(self.mr, 2) * np.exp(self.mr * texp)) - 5 * self.sigma * np.power(self.vov, 2) / (2 * np.power(self.mr, 3)) + self.sigma * np.power(self.vov, 2) / (2 * np.power(self.mr, 3) * np.exp(2 * self.mr * texp)) + 2 * self.sigma * np.power(self.vov, 2) / (np.power(self.mr, 3) * np.exp(self.mr * texp)) + self.theta * np.power(self.vov, 2) / np.power(self.mr, 3) - self.theta * np.power(self.vov, 2) / (np.power(self.mr, 3) * np.exp(2 * self.mr * texp))) / np.power(texp, 2)
        #M_24 = (-2 * np.power(self.sigma, 2) / self.mr + 2 * np.power(self.sigma, 2) / (self.mr * np.exp(self.mr * texp)) + 2 * self.sigma * self.theta / self.mr - 2 * self.sigma * self.theta / (self.mr * np.exp(self.mr * texp)) + self.sigma * np.power(self.vov, 2) / np.power(self.mr, 2) + 2 * self.sigma * np.power(self.vov, 2) / (np.power(self.mr, 2) * np.exp(self.mr * texp)) - 2 * self.theta * np.power(self.vov, 2) / (np.power(self.mr, 2) * np.exp(self.mr * texp))) / texp
        #M_2 = M_21 + M_22 + M_23 + M_24
        M2c_1 = - (decay * (var0 - theta)) ** 2
        M2c_2 = 2 * np.exp(term2 * texp) * (2 * mr * theta * (mr * theta + term2 * var0) + term1 * term2 * var0 ** 2)
        M2c_3 = -vov2 * (theta2 * (4 * mr * (3 - texp * mr) + (2 * texp * mr - 5) * vov2) + term2 * var0 * (
                    2 * theta + var0))

        M2c_4 = 2 * decay * vov2
        M2c_4 *= 2 * theta2 * (texp * mr2 - (1 + texp * mr) * vov2) \
                 + var0 * (2 * mr * theta * (1 + texp * term1) + term1 * var0)

        M_2 = M2c_1 / mr2 + M2c_2 / (term1 * term2) ** 2 + M2c_3 / mr2 / term2 ** 2 + M2c_4 / mr2 / term1 ** 2
        M_2 /= texp ** 2
        return M_2

    def density_func(self, price, spot, texp, cp=1):
        #implement the formula (15)
        mu = self.intr - self.divr

        #J, h = 201, 1  # need to take these as parameters
        #eta = (np.arange(J)[:, None] + 1) * h - 100  # shape=(J,1)

        def func_1(eta):
            return self.MGF((eta * eta + 0.25) * texp / 2, texp) * np.cos((np.log(price) - mu * texp) * eta)

        f,err = scint.quad(func_1, -np.inf, np.inf)

        return f * (np.exp(mu * texp / 2) / (2 * np.pi)) * np.power(price, -1.5)

    def price(self, strike, spot, texp, cp=1):
        #implement the formula (6)

        #J, h = 201, 0.5  # need to take these as parameters
        #P = (np.arange(J)[:, None] + 1) * h + strike  # shape=(J,1)

        def func_2(P):
            return (P - strike) * self.density_func(P / spot, spot, texp) / spot

        price,err = scint.quad(func_2, strike, np.inf)

        return price * np.exp(- self.intr * texp)

class OusvSchobelZhu1998(sv.SvABC):
    """
    The implementation of Schobel & Zhu (1998)'s inverse FT pricing formula for European
    options the Ornstein-Uhlenbeck driven stochastic volatility process.

    References: Schöbel, R., & Zhu, J. (1999). Stochastic Volatility With an Ornstein–Uhlenbeck Process: An Extension. Review of Finance, 3(1), 23–46. https://doi.org/10.1023/A:1009803506170

    Examples:
        >>> import pyfeng as pf
        >>> model = pf.OusvSchobelZhu1998(0.2, mr=4, vov=0.1, rho=-0.7, intr=0.09531)
        >>> model.price(100, 100, texp=np.array([1, 5, 10]))
        array([13.21493, 40.79773, 62.76312])
        >>> model = pf.OusvSchobelZhu1998(0.25, mr=8, vov=0.3, rho=-0.6, intr=0.09531)
        >>> model.price(np.array([90, 100, 110]), 100, texp=1)
        array([21.41873, 15.16798, 10.17448])
    """

    var_process = False

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


class OusvMcABC(sv.SvABC, sv.CondMcBsmABC, abc.ABC):

    model_type = "OUSV"
    var_process = False

    @abc.abstractmethod
    def cond_states(self, vol_0, texp):
        """
        Final variance and integrated variance over dt given var_0
        The integrated variance is normalized by dt
        Args:
            vol_0: initial volatility
            texp: time-to-expiry
        Returns:
            (var_final, var_mean, vol_mean)
        """
        return NotImplementedError

    def vol_step(self, vol_0, dt):
        """
        Stepping volatility according to OU process dynamics
        Args:
            vol_0: initial volatility
            dt: time step
        Returns:
            volatility after dt
        """
        mr_t = self.mr * dt
        e_mr = np.exp(-mr_t)
        sinh = np.sinh(mr_t)

        zz = self.rv_normal()
        vol_t = self.vov * np.sqrt(e_mr * sinh / self.mr) * zz
        vol_t += self.theta + (vol_0 - self.theta) * e_mr

        return vol_t

    def cond_spot_sigma(self, vol_0, texp):
        vol_texp, var_mean, vol_mean = self.cond_states(vol_0, texp)

        spot_cond = (vol_texp**2 - vol_0**2) / (2 * self.vov) - self.vov * texp / 2 \
            - (self.mr * self.theta / self.vov) * texp * vol_mean \
            + (self.mr / self.vov - self.rho / 2) * texp * var_mean
        np.exp(self.rho * spot_cond, out=spot_cond)

        sigma_cond = np.sqrt((1 - self.rho**2) * var_mean) / vol_0
        return spot_cond, sigma_cond


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

    def cond_states_full(self, sig_0, texp):
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        sigma_paths = self.vol_paths(tobs)
        s_t = sigma_paths[-1, :]
        u_t_std = scint.simps(sigma_paths, dx=1, axis=0) / n_dt
        v_t_std = scint.simps(sigma_paths**2, dx=1, axis=0) / n_dt

        return s_t, v_t_std, u_t_std

    def cond_states(self, vol_0, texp):
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        dt = np.diff(tobs, prepend=0)

        # precalculate the Simpson's rule weight
        weight = np.ones(n_dt + 1)
        weight[1:-1:2] = 4
        weight[2:-1:2] = 2
        weight /= weight.sum()

        vol_t = np.full(self.n_path, vol_0)
        mean_vol = weight[0] * vol_t
        mean_var = weight[0] * vol_t**2

        for i in range(n_dt):
            vol_t = self.vol_step(vol_t, dt[i])
            mean_vol += weight[i+1] * vol_t
            mean_var += weight[i+1] * vol_t**2

        return vol_t, mean_var, mean_vol


class OusvMcChoi2023(OusvMcABC):

    def set_mc_params(self, n_path=10000, dt=None, rn_seed=None, antithetic=True, n_sin=2, n_sin_max=None):
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

        super().set_mc_params(n_path, dt, rn_seed, antithetic)

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

    def cond_states(self, vol_0, texp):
        if self.dt is None:
            vol_t, var_mean, vol_mean = self.cond_states_step(vol_0, texp)
        else:
            tobs = self.tobs(texp)
            n_dt = len(tobs)
            dt = np.diff(tobs, prepend=0)

            vol_t = np.full(self.n_path, vol_0)
            vol_mean = np.zeros(self.n_path)
            var_mean = np.zeros(self.n_path)

            for i in range(n_dt):
                vol_t, d_v, d_u = self.cond_states_step(vol_t, dt[i], no_sin=True)
                vol_mean += d_u*dt[i]
                var_mean += d_v*dt[i]

            vol_mean /= texp
            var_mean /= texp
        return vol_t, var_mean, vol_mean

    def cond_states_step(self, vol_0, dt, no_sin=False):
        if no_sin:
            n_sin_max = n_sin = 0
        else:
            n_sin_max = self.n_sin_max
            n_sin = self.n_sin

        n_path = self.n_path
        n_path_half = int(n_path//2)

        mr, vov, theta = self.mr, self.vov, self.theta

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

        sighat = vov * np.sqrt(e_mr * sinh / mr) * z_0

        g_std = np.sqrt(self._a2overn2sum(mr_t, ns=n_sin, odd=1))
        p_std = np.sqrt(self._a6n2sum(mr_t, ns=n_sin, odd=1))
        q_std = np.sqrt(self._a6n2sum(mr_t, ns=n_sin, odd=2))  # even
        corr = self._a4sum(mr_t, ns=n_sin, odd=1) / (g_std * p_std)

        z_g = corr * z_p + np.sqrt(1 - corr**2) * z_g
        z_g *= g_std
        z_p *= p_std
        z_q *= q_std

        n_pi = np.pi * np.arange(1, n_sin + 1)
        an2 = 2 / (mr_t**2 + n_pi**2)  ## Careful: an contains texp in sqrt
        an = np.sqrt(an2)
        an3_n_pi = an2 * an * n_pi

        z_sin = zn[:, 5:n_sin + 5]
        z_g += z_sin[:, ::2] @ (an[::2] / n_pi[::2])
        z_p += z_sin[:, ::2] @ an3_n_pi[::2]
        z_q += z_sin[:, 1::2] @ an3_n_pi[1::2]

        UT = theta + (vol_0 - theta) * (1 - e_mr) / mr_t  # * dt
        UT += (cosh - 1) / (mr_t * sinh) * sighat + 2 * vovn * z_g  # * dt

        VT = theta * (2 * UT - theta) + (vol_0 - theta)**2 * (1 - e_mr**2) / (2 * mr_t)
        VT += sighat * (vol_0 - theta) * (1 / sinh - e_mr / mr_t) + (vol_0 - theta) * vovn * (
                    (1 + e_mr) * z_p + (1 - e_mr) * z_q)

        ## VT: even terms
        VT += (sinh * cosh - mr_t) / (2 * mr_t * sinh**2) * sighat**2 + sighat * vovn * (z_p - z_q)

        # LN variate (even term)
        m1 = self._a2sum(mr_t, ns=n_sin)
        var = 2 * self._a4sum(mr_t, ns=n_sin)
        ln_sig = np.sqrt(np.log(1 + var / m1**2))
        VT += 0.5 * vovn**2 * (z_sin**2 @ an**2 + m1 * np.exp(ln_sig * (z_r - 0.5 * ln_sig)))

        sighat += theta + (vol_0 - theta) * e_mr

        return sighat, VT, UT