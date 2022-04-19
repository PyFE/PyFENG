import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv
import math

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

class OusvBallRoma1994(sv.SvABC):
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
        M_1 = (- self.sigma + np.exp(self.mr * texp) * self.sigma - self.mr * np.exp(self.mr * texp) * self.sigma * texp + self.theta - np.exp(self.mr * texp) * self.theta) / (self.mr * np.exp(self.mr * texp) * texp)
        return M_1

    def second_derivative(self, texp):
        M_21 = np.power(self.sigma, 2)
        M_22 = (np.power(self.sigma, 2) / np.power(self.mr, 2) + np.power(self.sigma, 2) / (np.power(self.mr, 2) * np.exp(2 * self.mr * texp)) - 2 * np.power(self.sigma, 2) / (np.power(self.mr, 2) * np.exp(self.mr * texp)) - 2 * self.sigma * self.theta / np.power(self.mr, 2) - 2 * self.sigma * self.theta / (np.power(self.mr, 2) * np.exp(2 * self.mr * texp)) + 4 * self.sigma * self.theta / (np.power(self.mr, 2) * np.exp(self.mr * texp)) + np.power(self.theta, 2) / np.power(self.mr, 2)) / np.power(texp, 2)
        M_23 = (np.power(self.theta, 2) / (np.power(self.mr, 2) * np.exp(2 * self.mr * texp)) - 2 * np.power(self.theta, 2) / (np.power(self.mr, 2) * np.exp(self.mr * texp)) - 5 * self.sigma * np.power(self.vov, 2) / (2 * np.power(self.mr, 3)) + self.sigma * np.power(self.vov, 2) / (2 * np.power(self.mr, 3) * np.exp(2 * self.mr * texp)) + 2 * self.sigma * np.power(self.vov, 2) / (np.power(self.mr, 3) * np.exp(self.mr * texp)) + self.theta * np.power(self.vov, 2) / np.power(self.mr, 3) - self.theta * np.power(self.vov, 2) / (np.power(self.mr, 3) * np.exp(2 * self.mr * texp))) / np.power(texp, 2)
        M_24 = (-2 * np.power(self.sigma, 2) / self.mr + 2 * np.power(self.sigma, 2) / (self.mr * np.exp(self.mr * texp)) + 2 * self.sigma * self.theta / self.mr - 2 * self.sigma * self.theta / (self.mr * np.exp(self.mr * texp)) + self.sigma * np.power(self.vov, 2) / np.power(self.mr, 2) + 2 * self.sigma * np.power(self.vov, 2) / (np.power(self.mr, 2) * np.exp(self.mr * texp)) - 2 * self.theta * np.power(self.vov, 2) / (np.power(self.mr, 2) * np.exp(self.mr * texp))) / texp
        M_2 = M_21 + M_22 + M_23 + M_24
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


class OusvMcCond(sv.SvABC, sv.CondMcBsmABC):
    """
    OUSV model with conditional Monte-Carlo simulation
    The SDE of SV is: d sigma_t = mr (theta - sigma_t) dt + vov dB_T
    """

    var_process = False

    def vol_paths(self, tobs):
        # 2d array of (time, path) including t=0
        exp_tobs = np.exp(self.mr * tobs)

        bm_path = self._bm_incr(exp_tobs**2 - 1, cum=True)  # B_s (0 <= s <= 1)
        sigma_t = self.theta + (
            self.sigma - self.theta + self.vov / np.sqrt(2 * self.mr) * bm_path
        ) / exp_tobs[:, None]
        sigma_t = np.insert(sigma_t, 0, self.sigma, axis=0)
        return sigma_t

    def cond_spot_sigma(self, texp):

        rhoc = np.sqrt(1.0 - self.rho ** 2)
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        sigma_paths = self.vol_paths(tobs)
        sigma_final = sigma_paths[-1, :]
        int_sigma = scint.simps(sigma_paths, dx=1, axis=0) / n_dt
        int_var = scint.simps(sigma_paths ** 2, dx=1, axis=0) / n_dt

        spot_cond = np.exp(
            self.rho
            * (
                (sigma_final ** 2 - self.sigma ** 2) / (2 * self.vov)
                - self.vov * texp / 2
                - self.mr * self.theta / self.vov * int_sigma
                + (self.mr / self.vov - self.rho / 2) * int_var
            )
        )  # scaled by initial value

        # scaled by initial volatility
        sigma_cond = rhoc * np.sqrt(int_var) / self.sigma

        return spot_cond, sigma_cond
