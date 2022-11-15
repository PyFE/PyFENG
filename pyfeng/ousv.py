import abc
import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv
from . import bsm

#### Use of RN generation spawn:
# 0: simulation of sigma: vol_step()
# 1: truncated sine series: `z_gpqr`
# 2: sine series: `z_sin`
# 3: not used
# 4: not used
# 5: asset return


class OusvABC(sv.SvABC, abc.ABC):

    model_type = "OUSV"
    var_process = False

    def avgvol_mv(self, texp, vol0=None, nz_theta=True):
        """
        Mean and variance of the volatility sigma(t+dt) given sigma(t) = var_0
        (variance is not implemented yet)

        Args:
            texp: time step
            vol0: initial sigma
            nz_theta: non-zero theta. True by default. If False, assume theta=0 making computation simpler.

        Returns:
            mean, variance(=None)
        """

        if vol0 is None:
            vol0 = self.sigma

        if nz_theta:
            vol0 = vol0 - self.theta

        mr_t = self.mr * texp
        e_mr = np.exp(-mr_t)
        phi = (1 - e_mr)/mr_t
        m = vol0 * phi

        if nz_theta:
            m += self.theta

        return m, None

    def avgvar_mv(self, texp, vol0=None, nz_theta=True):
        """
        Mean and variance of the variance sigma^2(t+dt) given sigma(t) = var_0
        (variance is not implemented yet)

        Args:
            texp: time step
            vol0: initial sigma
            nz_theta: non-zero theta. True by default. If False, assume theta=0 making computation simpler.

        Returns:
            mean, variance(=None)
        """

        if vol0 is None:
            vol0 = self.sigma

        if nz_theta:
            vol0 = vol0 - self.theta

        mr_t = self.mr * texp
        e_mr = np.exp(-mr_t)
        phi = (1 - e_mr)/mr_t
        phi2 = (1 + e_mr)/2 * phi
        vv = vol0**2 * phi2 + 0.5*(self.vov**2/self.mr)*(1 - phi2)

        if nz_theta:
            vv += self.theta * (self.theta + 2*vol0 * phi)

        return vv, None

    def cond_avgvolvar_m(self, texp, vol_0, vol_t, nz_theta=True):
        """
        Conditional expectation of average vol and variance
        Args:
            texp: time step
            vol_0: initial vol
            vol_t: final vol
            nz_theta: non-zero theta. True by default. If False, assume theta=0 making computation simpler.

        Returns:
            avgvol mean, avgvar mean
        """
        mr_t = self.mr * texp
        e_mr = np.exp(-mr_t)
        phi = (1 - e_mr)/mr_t
        phi2 = (1 + e_mr)/2 * phi
        sinh = np.sinh(mr_t)
        cosh = np.cosh(mr_t)

        if nz_theta:
            vol_0 = vol_0 - self.theta  # Don't do `vol_0 -= self.theta`. Don't change `vol_0`
            vol_t = vol_t - self.theta

        volhat = vol_t - vol_0*e_mr

        vol_m = (vol_0 + volhat/(1+e_mr))*phi
        var_m = vol_0**2 * phi2 + 0.5*(self.vov**2/self.mr) * (cosh/sinh - 1/mr_t)
        var_m += volhat * (volhat * (sinh*cosh - mr_t)/(2*mr_t * sinh**2) + vol_0 * (e_mr/mr_t) * (1/phi2 - 1))

        if nz_theta:
            var_m += self.theta*(self.theta + 2*vol_m)
            vol_m += self.theta

        return vol_m, var_m

    def strike_var_swap_analytic(self, texp, dt):
        """
        Analytic fair strike of variance swap. Eq (17), (24), (25) in Bernard & Cui (2014)

        Args:
            texp: time to expiry
            dt: observation time step (e.g., dt=1/12 for monthly) For continuous monitoring, set dt=0

        Returns:
            Fair strike

        References:
            - Bernard C, Cui Z (2014) Prices and Asymptotics for Discrete Variance Swaps. Applied Mathematical Finance 21:140–173. https://doi.org/10.1080/1350486X.2013.820524

        """

        ### continuously monitored fair strike (same as mean of avgvar_mv)
        mrt = self.mr * texp
        e_mrt = np.exp(-mrt)
        phi = (e_mrt - 1) / mrt  # D / T
        x0 = self.sigma - self.theta
        strike, _ = self.avgvar_mv(texp, self.sigma)

        if not np.all(np.isclose(dt, 0.0)):
            mrt2 = mrt**2
            vov2t = self.vov**2 * texp

            sig = self.sigma
            sig2 = sig**2
            th = self.theta
            th2 = th**2

            E = 4*mrt2*(sig2**2 - th2**2) - 3*vov2t*(vov2t + 4*th2*mrt)  # E * T^2
            d2 = (vov2t + 2*mrt*th2) + ((2*mrt*(th2-sig2) + vov2t) + mrt*(vov2t/2 - mrt*x0**2)*phi)*phi

            # d1 / T
            d1 = sig2**2/4 - E*(1 + phi)/(16*mrt2)
            d1 += ((3*vov2t/4 - mrt*sig*x0/2)*sig2 + E/(32*mrt)) * phi**2
            d1 += ((2*th*sig/3 - sig2/6 - th2/2)*sig2*mrt2 - E/48 + (-mrt*sig*th + 3/4*sig2*mrt - vov2t/4)*vov2t) * phi**3
            d1 += (E/(8*mrt) - 3*vov2t*x0*th + 3*sig2*vov2t/2 - mrt*sig*x0*(2*th2 - th*sig + sig2)) * phi**4 * mrt2 / 8

            correction = self.intr * (self.intr - strike) + d1 - self.vov/(2*self.mr)*self.rho * d2/texp
            strike += correction * dt

        return strike


class OusvSchobelZhu1998(OusvABC):
    """
    The implementation of Schobel & Zhu (1998)'s inverse FT pricing formula for European
    options the Ornstein-Uhlenbeck driven stochastic volatility process.

    References:
        - Schöbel, R., & Zhu, J. (1999). Stochastic Volatility With an Ornstein–Uhlenbeck Process: an Extension. Review of Finance, 3(1), 23–46. https://doi.org/10.1023/A:1009803506170

    Examples:
        >>> import pyfeng as pf
        >>> model = pf.OusvSchobelZhu1998(0.2, mr=4, vov=0.1, rho=-0.7, intr=0.09531)
        >>> model.price(100, 100, texp=np.array([1, 5, 10]))
        array([13.21493, 40.79773, 62.76312])
        >>> model = pf.OusvSchobelZhu1998(0.25, mr=8, vov=0.3, rho=-0.6, intr=0.09531)
        >>> model.price(np.array([90, 100, 110]), 100, texp=1)
        array([21.41873, 15.16798, 10.17448])
    """

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
            / (2*s2g3)
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


class OusvUncorrBallRoma1994(OusvABC):
    """
    Ball & Roma (1994)'s approximation pricing formula for European options under uncorrelated (rho=0) OUSV model.
    Just 0th order (average variance) is implemented because higher order terms depends on numerical derivative of MGF

    See Also: HestonUncorrBallRoma1994, GarchUncorrBaroneAdesi2004
    """

    order = 0

    def price(self, strike, spot, texp, cp=1):

        if not np.isclose(self.rho, 0.0):
            print(f"Pricing ignores rho = {self.rho}.")

        avgvar, _ = self.avgvar_mv(texp, self.sigma)

        m_bs = bsm.Bsm(np.sqrt(avgvar), intr=self.intr, divr=self.divr)
        price = m_bs.price(strike, spot, texp, cp)

        if self.order > 0:
            raise ValueError(f"Not implemented for approx order: {self.order}")

        return price


class OusvMcABC(OusvABC, sv.CondMcBsmABC, abc.ABC):

    @abc.abstractmethod
    def cond_states_step(self, dt, vol_0, nz_theta=True):
        """
        Final volatility (sigma), average variance and volatility over dt given vol_0

        Args:
            dt: time-to-expiry
            vol_0: initial volatility
            nz_theta: non-zero theta. True by default. If False, assume theta=0 making computation simpler.

        Returns:
            (final vol, average var, average vol)
        """
        return NotImplementedError

    def vol_step(self, dt, vol_0, zn=None, nz_theta=True):
        """
        Stepping volatility according to OU process dynamics

        Args:
            vol_0: initial volatility
            dt: time step
            zn: specified normal rv to use (n_path, )
            nz_theta: non-zero theta. True by default. If False, assume theta=0 making computation simpler.

        Returns:
            volatility after dt
        """
        if nz_theta:
            vol_0 = vol_0 - self.theta  # Don't do `vol_0 -= self.theta`. Don't change `vol_0`

        e_mr = np.exp(-self.mr * dt)
        if zn is None:
            zn = self.rv_normal(spawn=0)

        vol_t = vol_0*e_mr + self.vov*np.sqrt((1 - e_mr**2)/(2*self.mr))*zn

        if nz_theta:
            vol_t += self.theta

        return vol_t

    def cond_spot_sigma(self, texp, vol_0):
        tobs = self.tobs(texp)
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        vol_t = np.full(self.n_path, vol_0 - self.theta)
        avgvar = np.zeros(self.n_path)
        avgvol = np.zeros(self.n_path)

        for i in range(n_dt):
            vol_t, avgvar_inc, avgvol_inc = self.cond_states_step(dt[i], vol_t, nz_theta=False)
            avgvar += avgvar_inc * dt[i]
            avgvol += avgvol_inc * dt[i]

        avgvar /= texp
        avgvol /= texp

        avgvar += self.theta * (self.theta + 2*avgvol)
        avgvol += self.theta
        vol_t += self.theta

        spot_cond = (vol_t**2 - vol_0**2 - self.vov**2*texp) +\
            texp*(-2*(self.mr * self.theta) * avgvol + (2*self.mr - self.rho*self.vov) * avgvar)
        np.exp(0.5*self.rho/self.vov * spot_cond, out=spot_cond)

        sigma_cond = np.sqrt((1 - self.rho**2) * np.fmax(avgvar, 1e-64)) / vol_0
        return spot_cond, sigma_cond

    def strike_var_swap_analytic(self, texp, dt=None):
        if dt is None:
            dt = self.dt
        rv = super().strike_var_swap_analytic(texp, dt)
        return rv

    def cond_log_return_var(self, dt, vol_0, vol_t, avgvar, avgvol):
        """
        Conditional log return variance expectation

            dt: time step
            vol_0: initial variance
            vol_t: final variance
            avgvar: average variance
            avgvol: average volatility

        Returns:
            expected log return

        """
        rho_vov = self.rho / self.vov
        ln_m = (self.intr - self.divr - self.rho*self.vov/2)*dt \
               + rho_vov * ((vol_t**2 - vol_0**2)/2 - self.mr*self.theta*dt*avgvol) \
               + (rho_vov*self.mr - 0.5)*dt*avgvar
        ln_sig2 = (1.0 - self.rho**2) * dt * avgvar
        return ln_m**2 + ln_sig2

    def draw_log_return(self, dt, vol_0, vol_t, avgvar, avgvol):
        """
        Samples log return, log(S_t/S_0)

        Args:
            dt: time step
            vol_0: initial variance
            vol_t: final variance
            avgvar: average variance
            avgvol: average volatility

        Returns:
            log return
        """
        rho_vov = self.rho / self.vov
        ln_m = (self.intr - self.divr - self.rho*self.vov/2)*dt\
               + rho_vov * ((vol_t**2 - vol_0**2)/2 - self.mr*self.theta*dt*avgvol)\
               + (rho_vov*self.mr - 0.5)*dt*avgvar
        ln_sig = np.sqrt((1.0 - self.rho**2) * dt * avgvar)
        zn = self.rv_normal(spawn=5)
        return ln_m + ln_sig * zn

    def return_var_realized(self, texp, cond=False):
        """
        Annualized realized return variance

        Args:
            texp: time to expiry
            cond: use conditional expectation without simulating price

        Returns:

        """
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        dt = np.diff(tobs, prepend=0)

        var_r = np.zeros(self.n_path)
        vol_0 = np.full(self.n_path, self.sigma)

        for i in range(n_dt):
            vol_t, avgvar_inc, avgvol_inc = self.cond_states_step(dt[i], vol_0)

            if cond:
                var_r += self.cond_log_return_var(dt[i], vol_0, vol_t, avgvar_inc, avgvol_inc)
            else:
                var_r += self.draw_log_return(dt[i], vol_0, vol_t, avgvar_inc, avgvol_inc)**2

            vol_0 = vol_t

        return var_r / texp  # annualized


class OusvMcTimeDisc(OusvMcABC):
    """
    OUSV model with conditional Monte-Carlo simulation
    The SDE of SV is: d sigma_t = mr (theta - sigma_t) dt + vov dB_T
    """
    scheme = 0  ## 0 for trapezoidal, 1 for mean

    def vol_paths(self, tobs):
        # 2d array of (time, path) including t=0
        exp_tobs = np.exp(self.mr * tobs)

        bm_path = self._bm_incr(exp_tobs**2 - 1, cum=True)  # B_s (0 <= s <= 1)
        sigma_t = self.theta + (
            self.sigma - self.theta + self.vov / np.sqrt(2 * self.mr) * bm_path
        ) / exp_tobs[:, None]
        sigma_t = np.insert(sigma_t, 0, self.sigma, axis=0)
        return sigma_t

    def cond_states_full(self, texp, sig_0):
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        sigma_paths = self.vol_paths(tobs)
        s_t = sigma_paths[-1, :]
        u_t_std = scint.simps(sigma_paths, dx=1, axis=0) / n_dt
        v_t_std = scint.simps(sigma_paths**2, dx=1, axis=0) / n_dt

        return s_t, v_t_std, u_t_std

    def cond_states_step(self, dt, vol_0, nz_theta=True):
        """
        Final volatility (sigma), average variance and volatilityu over dt given vol_0

        Args:
            dt: time-to-expiry
            vol_0: initial volatility
            nz_theta: non-zero theta. True by default. If False, assume theta=0 making computation simpler.

        Returns:
            (final vol, average var, average vol)
        """

        if nz_theta:
            vol_0 = vol_0 - self.theta  # Don't do `vol_0 -= self.theta`. Don't change `vol_0`

        vol_t = self.vol_step(dt, vol_0, nz_theta=False)

        if self.scheme == 0:
            avgvol = (vol_0 + vol_t) / 2
            avgvar = (vol_0**2 + vol_t**2) / 2
        elif self.scheme == 1:
            avgvol, avgvar = self.cond_avgvolvar_m(dt, vol_0, vol_t, nz_theta=False)

        if nz_theta:
            avgvar += self.theta * (self.theta + 2*avgvol)
            avgvol += self.theta
            vol_t += self.theta

        return vol_t, avgvar, avgvol


class OusvMcChoi2023KL(OusvMcABC):

    n_sin = 2

    def set_num_params(self, n_path=10000, dt=None, rn_seed=None, antithetic=True, n_sin=2):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step for Euler/Milstein steps
            rn_seed: random number seed
            antithetic: antithetic
        """
        assert n_sin % 2 == 0
        self.n_sin = n_sin

        super().set_num_params(n_path, dt, rn_seed, antithetic)

    @classmethod
    def _a2sum(cls, mr_t, ns=0, odd=None):
        """
        sum_{n=ns+1}^\infty a_n^2  where  a_n = sqrt(2 / (mr_t^2 + (n*pi)^2))

        Args:
            mr_t: mean reversion * time step
            ns: number of truncated terms. Must be an even number
            odd: sum all terms if None (default), odd terms only if odd=1, or even terms only if odd=2.

        Returns:
            sum
        """
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
        """
        sum_{n=ns+1}^\infty a_n^2 / (n pi)^2  where  a_n = sqrt(2 / (mr_t^2 + (n*pi)^2))

        Args:
            mr_t: mean reversion * time step
            ns: number of truncated terms. Must be an even number
            odd: sum all terms if None (default), odd terms only if odd=1, or even terms only if odd=2.

        Returns:
            sum
        """

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
        """
        sum_{n=ns+1}^\infty a_n^4  where  a_n = sqrt(2 / (mr_t^2 + (n*pi)^2))

        Args:
            mr_t: mean reversion * time step
            ns: number of truncated terms. Must be an even number
            odd: sum all terms if None (default), odd terms only if odd=1, or even terms only if odd=2.

        Returns:
            sum
        """

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
        """
        sum_{n=ns+1}^\infty a_n^6  where  a_n = sqrt(2 / (mr_t^2 + (n*pi)^2))

        Args:
            mr_t: mean reversion * time step
            ns: number of truncated terms. Must be an even number
            odd: sum all terms if None (default), odd terms only if odd=1, or even terms only if odd=2.

        Returns:
            sum
        """

        if odd == 2:  # even
            rv = cls._a6sum(mr_t / 2) / 2**6
        elif odd == 1:  # odd
            rv = (3*mr_t / np.tanh(mr_t) + (3 + 2*mr_t / np.tanh(mr_t))*mr_t**2 / np.sinh(mr_t)**2 - 8)/(2*mr_t**6) \
                 - cls._a6sum(mr_t / 2) / 2**6
        else:  # all
            rv = (3*mr_t / np.tanh(mr_t) + (3 + 2*mr_t / np.tanh(mr_t))*mr_t**2 / np.sinh(mr_t)**2 - 8)/(2*mr_t**6)

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
        """
        sum_{n=ns+1}^\infty (n pi)^2 a_n^6  where  a_n = sqrt(2 / (mr_t^2 + (n*pi)^2))

        Args:
            mr_t: mean reversion * time step
            ns: number of truncated terms. Must be an even number
            odd: sum all terms if None (default), odd terms only if odd=1, or even terms only if odd=2.

        Returns:
            sum
        """

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

    def cond_states_step(self, dt, vol_0, nz_theta=True, zn=None):
        """
        Final volatility (sigma), average variance and volatilityu over dt given vol_0

        Args:
            dt: time-to-expiry
            vol_0: initial volatility
            zn: normal RVs to specify in the (1+n_sin, n_path) format. None by default.
            nz_theta: non-zero theta. True by default. If False, assume theta=0 making computation simpler.

        Returns:
            (final vol, average var, average vol)
        """

        mr_t = self.mr * dt
        vovn = self.vov * np.sqrt(dt)  # normalized vov

        if nz_theta:
            vol_0 = vol_0 - self.theta  # Don't do `vol_0 -= self.theta`. Don't change `vol_0`

        if zn is None:
            vol_t = self.vol_step(dt, vol_0, nz_theta=False)
        else:
            vol_t = self.vol_step(dt, vol_0, zn=zn[0, :], nz_theta=False)

        if zn is None:
            n_sin = self.n_sin
            n_path = self.n_path

            if self.antithetic:
                n_path_half = int(n_path//2)
                z_sin = self.rng_spawn[2].standard_normal(size=(n_path_half, n_sin)).T
                z_sin = np.stack([z_sin, -z_sin], axis=-1).reshape((n_sin, n_path))

                z_gpqr = self.rng_spawn[1].standard_normal(size=(n_path_half, 4)).T
                z_gpqr = np.stack([z_gpqr, -z_gpqr], axis=-1).reshape((4, n_path))
            else:
                z_sin = self.rng_spawn[2].standard_normal(size=(n_path, n_sin)).T
                z_gpqr = self.rng_spawn[1].standard_normal(size=(n_path, 4)).T

            # Create views to array rows
            z_g = z_gpqr[0, :]
            z_p = z_gpqr[1, :]
            z_q = z_gpqr[2, :]
            z_r = z_gpqr[3, :]

            g_std = np.sqrt(self._a2overn2sum(mr_t, ns=n_sin, odd=1))
            p_std = np.sqrt(self._a6n2sum(mr_t, ns=n_sin, odd=1))
            q_std = np.sqrt(self._a6n2sum(mr_t, ns=n_sin, odd=2))  # even
            corr = self._a4sum(mr_t, ns=n_sin, odd=1)/(g_std*p_std)
            corr = np.clip(corr, -1.0, 1.0)

            z_g[:] = (corr*z_p + np.sqrt(1.0 - corr**2)*z_g) * g_std
            z_p *= p_std
            z_q *= q_std

            r_var = self._a4sum(mr_t, ns=n_sin)
            z_r[:] = np.sqrt(r_var)*(z_r**2 - 1)
        else:
            n_sin = zn.shape[0] - 1
            r_m = self._a2sum(mr_t, ns=n_sin)
            z_sin = zn[1:, :]
            z_g, z_p, z_q, z_r = 0.0, 0.0, 0.0, -r_m
            #### -r_m is the correction to include only `an2 @ z_sin**2`

        if n_sin > 0:
            n_pi = np.pi * np.arange(1, n_sin + 1)
            an2 = 2 / (mr_t**2 + n_pi**2)
            an = np.sqrt(an2)
            an3_n_pi = an2 * an * n_pi

            z_g += (an[::2] / n_pi[::2]) @ z_sin[::2, :]  # odd terms
            z_p += an3_n_pi[::2] @ z_sin[::2, :]  # odd terms
            z_q += an3_n_pi[1::2] @ z_sin[1::2, :]  # even terms
            z_r += an2 @ (z_sin**2 - 1)

        uu_t, vv_t = self.cond_avgvolvar_m(dt, vol_0, vol_t, nz_theta=False)
        uu_t += 2 * vovn * z_g  # * dt
        vv_t += vovn * ((vol_0*(z_p + z_q) + vol_t*(z_p - z_q)) + 0.5*vovn*z_r)

        if nz_theta:
            vv_t += (2*uu_t + self.theta) * self.theta
            uu_t += self.theta
            vol_t += self.theta

        return vol_t, vv_t, uu_t

    def unexplained_var_ratio(self, mr_t, ns=None):

        if ns is None:
            ns = self.n_sin
        rv = self._a4sum(mr_t, ns=ns) / self._a4sum(mr_t)
        return rv

    def strike_var_swap_analytic(self, texp, dt=None):
        if dt is None:
            dt = self.dt
        rv = super().strike_var_swap_analytic(texp, dt)
        return rv

    def vol_path_sin(self, tobs, zn=None):
        """
        vol path composed of sin terms
        Args:
            tobs: observation time (n_time, )
            zn: specified normal rvs to use (n_sin + 1, n_path). None by default

        Returns:
            vol path (n_time, n_path)
        """
        dt = tobs[-1]
        mr_t = self.mr * dt
        e_mr = np.exp(-mr_t)
        e_mr_tobs = np.exp(-self.mr*tobs[:, None])

        vol_0 = self.sigma - self.theta

        if zn is None:
            vol_t = self.vol_step(dt, vol_0, nz_theta=False)
            zn = self.rng_spawn[2].standard_normal(size=(self.n_sin, self.n_path))
            n_sin, n_path = self.n_sin, self.n_path
        else:
            vol_t = self.vol_step(dt, vol_0, zn[0, :], nz_theta=False)
            n_sin = zn.shape[0] - 1

        volhat = vol_t - vol_0 * e_mr

        n_pi = np.pi*np.arange(1, n_sin + 1)
        an = np.sqrt(2/(mr_t**2 + n_pi**2))
        sin = np.sin(n_pi*tobs[:, None]/dt)
        sigma_path = self.theta + vol_0 * e_mr_tobs \
                     + 0.5*(1/e_mr_tobs - e_mr_tobs)/np.sinh(mr_t) * volhat \
                     + self.vov * np.sqrt(dt) * (an*sin) @ zn[1:, :]

        return sigma_path
