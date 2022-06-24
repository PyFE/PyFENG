import abc
import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv
from . import bsm


class OusvABC(sv.SvABC, abc.ABC):

    model_type = "OUSV"

    def avgvar_mv(self, var0, texp):
        """
        Mean and variance of the variance V(t+dt) given V(0) = var_0
        (variance is not implemented yet)

        Args:
            var0: initial variance
            texp: time step

        Returns:
            mean, variance(=None)
        """

        mr_t = self.mr * texp
        e_mr = np.exp(-mr_t)
        x0 = var0 - self.theta
        vv = self.vov**2/2/self.mr + self.theta**2 + \
             ((x0**2 - self.vov**2/2/self.mr)*(1 + e_mr) + 4*self.theta * x0)*(1 - e_mr)/(2*self.mr*texp)
        return vv, None


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

        avgvar, _ = self.avgvar_mv(self.sigma, texp)

        m_bs = bsm.Bsm(np.sqrt(avgvar), intr=self.intr, divr=self.divr)
        price = m_bs.price(strike, spot, texp, cp)

        if self.order > 0:
            raise ValueError(f"Not implemented for approx order: {self.order}")

        return price


class OusvMcABC(OusvABC, sv.CondMcBsmABC, abc.ABC):

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

    def vol_step(self, vol_0, dt, zn=None):
        """
        Stepping volatility according to OU process dynamics

        Args:
            vol_0: initial volatility
            dt: time step
            zn: specified normal rv to use (n_path, )

        Returns:
            volatility after dt
        """
        e_mr = np.exp(-self.mr * dt)
        if zn is None:
            zn = self.rv_normal(spawn=0)

        vol_t = self.theta + (vol_0 - self.theta)*e_mr + self.vov*np.sqrt((1 - e_mr**2)/(2*self.mr))*zn
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
                vol_t, d_v, d_u = self.cond_states_step(vol_t, dt[i])
                vol_mean += d_u * dt[i]
                var_mean += d_v * dt[i]

            vol_mean /= texp
            var_mean /= texp
        return vol_t, var_mean, vol_mean

    def cond_states_step(self, vol0, dt, zn=None):
        """
        Incremental conditional states

        Args:
            vol0: initial volatility
            dt: time step
            zn: specified normal rvs to use. (n_sin + 1, n_path)

        Returns:
            final volatility, average volatility, average variance. (n_path, ) each.
        """

        mr, vov = self.mr, self.vov

        mr_t = mr * dt
        e_mr = np.exp(-mr_t)
        sinh = np.sinh(mr_t)
        cosh = np.cosh(mr_t)
        vovn = self.vov * np.sqrt(dt)  # normalized vov

        x_0 = vol0 - self.theta
        if zn is None:
            x_t = self.vol_step(vol0, dt) - self.theta
        else:
            x_t = self.vol_step(vol0, dt, zn=zn[0, :]) - self.theta
        sighat = x_t - x_0 * e_mr

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

            z_g[:] = (corr*z_p + np.sqrt(1 - corr**2)*z_g) * g_std
            z_p *= p_std
            z_q *= q_std

            r_m = self._a2sum(mr_t, ns=n_sin)
            r_var = self._a4sum(mr_t, ns=n_sin)
            z_r[:] = np.sqrt(r_var)*(z_r**2 - 1) + r_m
        else:
            n_sin = zn.shape[0] - 1
            z_sin = zn[1:, :]
            z_g, z_p, z_q, z_r = 0.0, 0.0, 0.0, 0.0

        n_pi = np.pi * np.arange(1, n_sin + 1)
        an2 = 2 / (mr_t**2 + n_pi**2)
        an = np.sqrt(an2)
        an3_n_pi = an2 * an * n_pi

        z_g += (an[::2] / n_pi[::2]) @ z_sin[::2, :]  # odd terms
        z_p += an3_n_pi[::2] @ z_sin[::2, :]  # odd terms
        z_q += an3_n_pi[1::2] @ z_sin[1::2, :]  # even terms
        z_r += an2 @ z_sin**2

        uu_t = x_0 * (1 - e_mr) / mr_t + (cosh - 1) / (mr_t * sinh) * sighat + 2 * vovn * z_g  # * dt

        vv_t = (1 - e_mr**2) / (2 * mr_t) * x_0**2 + (sinh * cosh - mr_t) / (2 * mr_t * sinh**2) * sighat**2
        vv_t += sighat * x_0 * (1/sinh - e_mr / mr_t) + vovn * (x_0 * (z_p + z_q) + x_t * (z_p - z_q))
        vv_t += 0.5 * vovn**2 * z_r

        ### sigma_t = x_t + theta
        vv_t += (2*uu_t + self.theta) * self.theta
        uu_t += self.theta
        x_t += self.theta

        return x_t, vv_t, uu_t

    def unexplained_var_ratio(self, mr_t, ns):
        ns = ns or self.n_sin
        rv = self._a4sum(mr_t, ns=ns) / self._a4sum(mr_t)
        return rv

    def vol_path_sin(self, tobs, zn=None):
        """
        vol path composed of sin terms
        Args:
            vol_t: terminal volatility (n_path, )
            tobs: observation time (n_time, )
            zn: specified normal rvs to use (n_sin + 1, n_path). None by default

        Returns:
            vol path (n_time, n_path)
        """
        dt = tobs[-1]
        mr_t = self.mr * dt
        e_mr = np.exp(-mr_t)
        e_mr_tobs = np.exp(-self.mr*tobs[:, None])

        x_0 = self.sigma - self.theta

        if zn is None:
            vol_t = self.vol_step(self.sigma, dt)
            zn = self.rng_spawn[1].standard_normal(size=(self.n_sin, self.n_path))
            n_sin, n_path = self.n_sin, self.n_path
        else:
            vol_t = self.vol_step(self.sigma, dt, zn[0, :])
            n_sin = zn.shape[0] - 1

        sighat = vol_t - self.theta - x_0 * e_mr

        n_pi = np.pi*np.arange(1, n_sin + 1)
        an = np.sqrt(2/(mr_t**2 + n_pi**2))
        sin = np.sin(n_pi*tobs[:, None]/dt)
        sigma_path = self.theta + x_0 * e_mr_tobs \
                     + 0.5*(1/e_mr_tobs - e_mr_tobs)/np.sinh(mr_t) * sighat \
                     + self.vov * np.sqrt(dt) * (an*sin) @ zn[1:,:]

        return sigma_path

    def price_var_option(self, strike, texp, cp=1):
        """
        Price of variance option

        Args:
            strike:
            texp:
            cp:

        Returns:

        """
        df = np.exp(-self.intr * texp)
        vol_t, vv_t, uu_t = self.cond_states(self.sigma, texp)
        # vv_t is the average variance
        price = df * np.fmax(np.sign(cp)*(vv_t[:, None] - strike), 0).mean(axis=0)
        return price