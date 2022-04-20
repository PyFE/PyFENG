import numpy as np
import scipy.stats as spst
import abc
from . import sv_abc as sv
from . import bsm
import scipy.integrate as spint


class GarchCondMcABC(sv.SvABC, sv.CondMcBsmABC, abc.ABC):
    var_process = True
    model_type = "GarchDiff"
    scheme = None

    @abc.abstractmethod
    def cond_states(self, var_0, dt):
        """
        Final variance and integrated variance over dt given var_0
        The int_var is normalized by dt

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            (vol_final, var_mean, vol_mean, inv_vol_mean)
        """
        return NotImplementedError

    # @staticmethod
    def cond_spot_sigma(self, sigma_0, texp):
        var_0 = sigma_0**2
        vol_final, var_mean, vol_mean, inv_vol_mean = self.cond_states(var_0, texp)

        spot_cond = 2 * (vol_final - np.sqrt(var_0)) / self.vov \
            - self.mr * self.theta * inv_vol_mean * texp / self.vov \
            + (self.mr / self.vov + self.vov / 4) * vol_mean * texp \
            - self.rho * var_mean * texp / 2
        np.exp(self.rho * spot_cond, out=spot_cond)

        sigma_cond = np.sqrt((1.0 - self.rho**2)/var_0*var_mean)

        return spot_cond, sigma_cond



class GarchMcTimeStep(GarchCondMcABC):
    """
    Garch model with conditional Monte-Carlo simulation
    The SDE of SV is: dv_t = mr * (theta - v_t) dt + vov * v_t dB_T
    """
    scheme = 1  # Milstein

    def set_mc_params(self, n_path=10000, dt=0.05, rn_seed=None, antithetic=True, scheme=1):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step
            rn_seed: random number seed
            antithetic: antithetic
            scheme: 0 for Euler, 1 for Milstein (default), 2 for log variance

        References:
            - Andersen L (2008) Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11:1–42. https://doi.org/10.21314/JCF.2008.189
        """
        super().set_mc_params(n_path, dt, rn_seed, antithetic)
        self.scheme = scheme

    def var_step_euler(self, var_0, dt, milstein=True):
        """
        Euler/Milstein Schemes:
        v_(t+dt) = v_t + mr * (theta - v_t) * dt + vov * v_t Z * sqrt(dt) + (vov^2/2) v_t (Z^2-1) dt

        Args:
            var_0: initial variance
            dt: delta t, time step

        Returns: Variance path (time, path) including the value at t=0
        """

        zz = self.rv_normal()

        var_t = var_0 + self.mr * (self.theta - var_0) * dt + self.vov * var_0 * np.sqrt(dt) * zz
        if milstein:
            var_t += (self.vov**2 / 2) * var_0 * dt * (zz**2 - 1)

        # although rare, floor at zero
        var_t[var_t < 0] = 0.0

        return var_t

    def var_step_log(self, log_var_0, dt):
        """
        Euler schemes on w_t = log(v_t):
        w_(t+dt) = w_t + (mr * theta * exp(-w_t) - mr - vov^2 / 2) * dt + vov * Z * sqrt(dt)

        Args:
            log_var_0: initial log variance
            dt: time step

        Returns: Variance path (time, path) including the value at t=0
        """

        zz = self.rv_normal()

        log_var_t = log_var_0 + (self.mr * self.theta * np.exp(-log_var_0) - self.mr - self.vov**2 / 2) * dt \
                + self.vov * np.sqrt(dt) * zz

        return log_var_t

    def cond_states(self, var_0, texp):
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        dt = np.diff(tobs, prepend=0)

        # precalculate the Simpson's rule weight
        weight = np.ones(n_dt + 1)
        weight[1:-1:2] = 4
        weight[2:-1:2] = 2
        weight /= weight.sum()

        var_t = np.full(self.n_path, var_0)

        mean_var = weight[0] * var_t
        mean_vol = weight[0] * np.sqrt(var_t)
        mean_inv_vol = weight[0] / np.sqrt(var_t)

        if self.scheme < 2:
            milstein = (self.scheme == 1)

            for i in range(n_dt):
                var_t = self.var_step_euler(var_t, dt[i], milstein=milstein)
                mean_var += weight[i+1] * var_t
                vol_t = np.sqrt(var_t)
                mean_vol += weight[i+1] * vol_t
                mean_inv_vol += weight[i+1] / vol_t

        elif self.scheme == 2:
            log_var_t = np.full(self.n_path, np.log(var_0))
            for i in range(n_dt):
                # Euler scheme on Log(var)
                log_var_t = self.var_step_log(log_var_t, dt[i])
                vol_t = np.exp(log_var_t / 2)
                mean_var += weight[i+1] * vol_t**2
                mean_vol += weight[i+1] * vol_t
                mean_inv_vol += weight[i+1] / vol_t
        else:
            raise ValueError(f'Invalid scheme: {self.scheme}')

        return vol_t, mean_var, mean_vol, mean_inv_vol


class GarchUncorrBaroneAdesi2004(sv.SvABC):
    """
    The implementation of Barone-Adesi et al (2004)'s approximation pricing formula for European
    options under uncorrelated (rho=0) GARCH diffusion model.

    References:
        - Barone-Adesi G, Rasmussen H, Ravanelli C (2005) An option pricing formula for the GARCH diffusion model. Computational Statistics & Data Analysis 49:287–310. https://doi.org/10.1016/j.csda.2004.05.014

    This method is only used to compare with the method GarchCondMC.
    """
    model_type = "GarchDiff"

    def price(self, strike, spot, texp, cp=1):

        if not np.isclose(self.rho, 0.0):
            print(f"Pricing ignores rho = {self.rho}.")

        var0, mr, vov, theta = self.sigma, self.mr, self.vov, self.theta

        mr2 = mr * mr
        vov2 = vov * vov
        theta2 = theta*theta
        decay = np.exp(-mr * texp)

        # Eq (12) of Barone-Adesi et al. (2005)
        M1 = theta + (var0 - theta) * (1 - decay) / (mr * texp)

        term1 = vov2 - mr
        term2 = vov2 - 2*mr

        # Eq (13)
        M2c_1 = - (decay * (var0 - theta))**2
        M2c_2 = 2*np.exp(term2 * texp) * (2*mr * theta * (mr * theta + term2 * var0) + term1 * term2 * var0**2)
        M2c_3 = -vov2 * (theta2 * (4*mr * (3 - texp * mr) + (2*texp * mr - 5) * vov2) + term2 * var0 * (2*theta + var0))

        M2c_4 = 2*decay * vov2
        M2c_4 *= 2*theta2 * (texp * mr2 - (1 + texp * mr) * vov2) \
                 + var0 * (2*mr * theta * (1 + texp * term1) + term1 * var0)

        M2c = M2c_1 / mr2 + M2c_2 / (term1 * term2)**2 + M2c_3 / mr2 / term2**2 + M2c_4 / mr2 / term1**2
        M2c /= texp**2
        # M3c=None
        # M4c=None

        # Eq. (11)
        logk = np.log(spot / strike)
        sigma_std = np.sqrt(self.sigma * texp)

        m = logk + (self.intr - self.divr) * texp
        d1 = logk / sigma_std + sigma_std / 2

        m_bs = bsm.Bsm(np.sqrt(M1), intr=self.intr, divr=self.divr)
        c_bs = m_bs.price(strike, spot, texp, cp)

        c_bs_p1 = np.exp(-self.divr * texp) * strike * np.sqrt(texp / 4 / M1) * spst.norm.pdf(d1)
        c_bs_p2 = c_bs_p1 * texp / 2 * ((m / M1 / texp)**2 - 1 / M1 / texp - 1 / 4)

        # C_bs_p3=C_bs_p1*(m**4/(4*(M1*texp)**4)-m**2*(12+M1*texp)/(8*(M1*texp)**3)+(48+8*M1*texp+(M1*texp)**2)/(64*(M1*texp)**2))*texp**2
        # C_bs_p4=C_bs_p1*(m**6/(8*(M1*texp)**6)-3*m**4*(20+M1*texp)/(32*(M1*texp)**5)+3*m**2*(240+24*M1*texp+(M1*texp)**2)/(128*(M1*texp)**4)-(960+144*M1*texp+12*(M1*texp)**2+(M1*texp)**3)/(512*(M1*texp)**3))*texp**3

        c_ga_2 = c_bs + (M2c / 2) * c_bs_p2
        # C_ga_3=C_ga_2+(M3c/6)*C_bs_p3
        # C_ga_4=C_ga_3+(M4c/24)*C_bs_p4

        return c_ga_2


class GarchCapriotti2018(GarchCondMcABC, abc.ABC):
    def __init__(self, sigma_0, vov, rho, mr, theta):
        super().__init__(sigma=sigma_0, vov=vov, rho=rho, mr=mr, theta=theta)
        self.set_mc_params()
        self.rhoc = np.sqrt(1-rho**2)

    def _transition_density_logvar(self, x, x0, t, n: int = 3):
        """
        Transition density of log(var_t).
        Args:
            x: log(var_t)
            x0: log(var_0)
            t: time to expiry
            n: orders for approx

        Returns:
            pdf of log(var_t)
        """
        pdf_x = (
                1/np.sqrt(2*np.pi*self.sigma**2*t) *
                np.exp(-(x-x0)**2 / (2*self.sigma**2*t) - w_for_density(self.mr, self.theta, self.sigma, x, x0, t, n))
        )
        return pdf_x

    def transition_density_var(self, var_t, t, var_0, t0=0, n: int = 3):
        """
        Get transition desity of var at time t.
        At time t, pdf(Var_t) is
        f(y) = 1/y * ( 1/sqrt(2*pi) * exp(-(x-x0)^2/(2*sima^2*t) - W(x,x0)) )
        where, x = ln(y)
        Args:
            var_t: var at time t
            var_0: var at last time t-1
            t: current time
            t0: last time
            n: orders

        Returns:
            pdf of var
        """
        x = np.log(var_t)
        x0 = np.log(var_0)
        pdf_x = self._transition_density_logvar(x, x0, t-t0, n)
        pdf_y = pdf_x / var_t
        return pdf_y

    def _get_c(self, t, var_0, t0=0, num: int = 2000):
        var_0_mean = var_0.mean()
        yvals = np.r_[np.linspace(0.001, var_0_mean-0.002, num), np.linspace(var_0_mean+0.002, 1, num)]
        self.yvals = yvals
        xvals = np.log(yvals)
        wvals = w_for_density(self.mr, self.theta, self.vov, xvals, np.log(var_0_mean), t - t0)
        c = np.exp(-np.min(wvals)) + 1
        return c

    def rv(self, t, var_0, t0=0):
        """
        Sampling in terms of the other normal distribution.
        Args:
            t: current time t
            var_0: last variance
            t0: last time

        Returns:
            y_target: random variable sigma^2_t
        """
        # Identify the constant C
        c = self._get_c(t, var_0, t0)
        # Generate one RN
        x0 = np.log(var_0)
        if np.isscalar(x0):
            while True:
                # 1. first step: draw z, u independently
                (z, u) = (np.random.normal(x0, self.vov*np.sqrt(t-t0)), np.random.uniform(0, 1))
                # 2. second step: accept Z' = Z, if z and u satisfy the condition
                if u * c <= np.exp(-w_for_density(self.mr, self.theta, self.vov, z, x0, t - t0)):
                    break
            y_target = np.exp(z)
            return y_target
        # Generate n RNs
        else:
            num = 80
            z = np.random.randn(x0.size, num)*self.vov * np.sqrt(t - t0) + x0[:, None]
            u = np.random.uniform(0, 1, size=(x0.size, num))
            fgu = np.exp(-w_for_density(self.mr, self.theta, self.vov, z, x0[:, None], t - t0)) / u
            z_temp = fgu >= c
            while ~z_temp.any(axis=1).all():
                shape_retry = z[~z_temp.any(axis=1), :].shape
                zi = (np.random.randn(shape_retry[0], shape_retry[1])*self.vov * np.sqrt(t-t0) +
                      x0[~z_temp.any(axis=1), None])
                ui = np.random.uniform(0, 1, size=shape_retry)
                fgui = np.exp(-w_for_density(self.mr, self.theta, self.vov, zi, x0[~z_temp.any(axis=1), None],
                                             t - t0)) / ui
                z[~z_temp.any(axis=1), :] = zi
                z_temp[~z_temp.any(axis=1), :] = (zi-x0[~z_temp.any(axis=1), None] > 1e-3) & (fgui >= c)
            z_rows = z[np.arange(x0.size), z_temp.argmax(axis=1)]
            y_target = np.exp(z_rows)

            return y_target

    def cond_states(self, var_0, texp):
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        dt = self.dt

        varvals = np.zeros((n_dt+1, self.n_path))
        varvals[0, :] = var_0

        for i in range(n_dt):
            varvals[i+1, :] = self.rv(t=dt, var_0=varvals[i, :], t0=0)

        volvals = np.sqrt(varvals)
        vol_final = volvals[-1, :]
        var_mean = spint.simps(varvals, dx=dt, axis=0)
        vol_mean = spint.simps(volvals, dx=dt, axis=0)
        inn_vol_mean = spint.simps(1 / volvals, dx=dt, axis=0)
        return vol_final, var_mean, vol_mean, inn_vol_mean


def w_for_density(mr, theta, vov, x, x0, t, n: int = 3):
    assert (n > 0) and (n < 4)
    if np.isscalar(x):
        if abs(x-x0) < 1e-3:
            return w_at_x0(mr, theta, vov, x, t, n)
        else:
            return w_ex_x0(mr, theta, vov, x, x0, t, n)
    else:
        idx_x0 = abs(x-x0) < 1e-3
        w_res = np.zeros(x.shape)
        if idx_x0.any():
            x0 = x0 * np.ones(x.shape)
            w_res[idx_x0] = w_at_x0(mr, theta, vov, x[idx_x0], t, n)
            w_res[~idx_x0] = w_ex_x0(mr, theta, vov, x[~idx_x0], x0[~idx_x0], t, n)
        else:
            w_res = w_ex_x0(mr, theta, vov, x, x0, t, n)

    return w_res


def w_at_x0(mr, theta, vov, x, t, n: int = 3):
    assert n > 0

    a = mr
    b = theta
    sigma = vov

    w_res = 0
    if n > 0:
        w_res = (
                ((sigma**2/2 - a*np.exp(-x)*(b - np.exp(x)))*(sigma**2/2 + a - a*b*np.exp(-x)))/sigma**2 -
                (sigma**2/2 + a - a*b*np.exp(-x))**2/(2*sigma**2) - a +
                (a*b*np.exp(-x))/2 - a*np.exp(-x)*(b - np.exp(x))
        )*t
    if n > 1:
        w_res += (
                         ((sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x))) * (a + ((a + a * np.exp(-x) * (b - np.exp(x))) * (
                            sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 - (a * b * np.exp(-x)) / 2 + a * np.exp(
                    -x) * (b - np.exp(x)) - (a * b * np.exp(-x) * (
                            sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 + (a * b * np.exp(-x) * (
                            sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2)) / 2 - (
                            (sigma ** 2 / 2 + a - a * b * np.exp(-x)) * (a + ((a + a * np.exp(-x) * (b - np.exp(x))) * (
                                sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 - (
                                                                                 a * b * np.exp(-x)) / 2 + a * np.exp(
                        -x) * (b - np.exp(x)) - (a * b * np.exp(-x) * (
                                sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 + (a * b * np.exp(-x) * (
                                sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2)) / 2 - (sigma ** 2 * (
                    a + ((a + a * np.exp(-x) * (b - np.exp(x))) * (
                        sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 - (a * b * np.exp(-x)) / 2 + a * np.exp(
                -x) * (b - np.exp(x)) + (a ** 2 * b ** 2 * np.exp(-2 * x)) / sigma ** 2 - (
                                a * b * np.exp(-x) * (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 + (
                                a * b * np.exp(-x) * (sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2 - (
                                2 * a * b * np.exp(-x) * (a + a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2)) / 4

        ) * t**2
    if n > 2:
        w_res += (
                         ((sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x))) * (
                                 ((sigma ** 2 / 2 + a - a * b * np.exp(-x)) *
                                  (a + ((a + a * np.exp(-x) * (b - np.exp(x))) * (sigma ** 2 / 2 + a - a * b * np.exp(-x))) /
                                   sigma ** 2 - (a * b * np.exp(-x)) / 2 + a * np.exp(-x) * (b - np.exp(x)) +
                                   (a ** 2 * b ** 2 * np.exp(-2 * x)) / sigma ** 2 -
                                   (a * b * np.exp(-x) * (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 +
                                   (a * b * np.exp(-x) * (sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2 -
                                   (2 * a * b * np.exp(-x) * (a + a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2)) / 2 -
                                 ((sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x))) *
                                  (a + ((a + a * np.exp(-x) * (b - np.exp(x))) * (sigma ** 2 / 2 + a - a * b * np.exp(-x))) /
                                   sigma ** 2 - (a * b * np.exp(-x)) / 2 + a * np.exp(-x) * (b - np.exp(x)) +
                                   (a ** 2 * b ** 2 * np.exp(-2 * x)) / sigma ** 2 -
                                   (a * b * np.exp(-x) * (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 +
                                   (a * b * np.exp(-x) * (sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2 -
                                   (2 * a * b * np.exp(-x) * (a + a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2)) / 2 +
                                 (sigma ** 2 * (a + ((a + a * np.exp(-x) * (b - np.exp(x))) *
                                                    (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 -
                                               (a * b * np.exp(-x)) / 2 + a * np.exp(-x) * (b - np.exp(x)) +
                                               (3 * a ** 2 * b ** 2 * np.exp(-2 * x)) / sigma ** 2 -
                                               (a * b * np.exp(-x) * (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 +
                                               (a * b * np.exp(-x) * (sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x)))) /
                                               sigma ** 2 - (6 * a * b * np.exp(-x) * (a + a * np.exp(-x) * (b - np.exp(x)))) /
                                               sigma ** 2)) / 4 + (
                                         (a + a * np.exp(-x) * (b - np.exp(x))) *
                                         (a + ((a + a * np.exp(-x) * (b - np.exp(x))) *
                                               (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 -
                                          (a * b * np.exp(-x)) / 2 + a * np.exp(-x) * (b - np.exp(x)) -
                                          (a * b * np.exp(-x) * (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 +
                                          (a * b * np.exp(-x) * (sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x)))) /
                                          sigma ** 2)) / 2 -
                                 (a * b * np.exp(-x) * (a + ((a + a * np.exp(-x) * (b - np.exp(x))) *
                                                          (sigma ** 2 / 2 + a - a * b * np.exp(-x))) /
                                                     sigma ** 2 - (a * b * np.exp(-x)) / 2 +
                                                     a * np.exp(-x) * (b - np.exp(x)) -
                                                     (a * b * np.exp(-x) *
                                                      (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 +
                                                     (a * b * np.exp(-x) * (sigma ** 2 / 2 -
                                                                         a * np.exp(-x) * (b - np.exp(x)))) /
                                                     sigma ** 2)) / 2)) / 3 -
                         (sigma ** 2 * (((sigma ** 2 / 2 + a - a * b * np.exp(-x)) *
                                        (a + ((a + a * np.exp(-x) * (b - np.exp(x))) *
                                              (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 -
                                         (a * b * np.exp(-x)) / 2 + a * np.exp(-x) * (b - np.exp(x)) +
                                         (3 * a ** 2 * b ** 2 * np.exp(-2 * x)) / sigma ** 2 -
                                         (a * b * np.exp(-x) * (sigma ** 2 / 2 + a - a * b * np.exp(-x))) /
                                         sigma ** 2 + (a * b * np.exp(-x) *
                                                      (sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x)))) /
                                         sigma ** 2 - (6 * a * b * np.exp(-x) * (a + a * np.exp(-x) * (b - np.exp(x)))) /
                                         sigma ** 2)) / 2 -
                                       ((sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x))) *
                                        (a + ((a + a * np.exp(-x) * (b - np.exp(x))) *
                                              (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 -
                                         (a * b * np.exp(-x)) / 2 + a * np.exp(-x) * (b - np.exp(x)) + (3 * a ** 2 * b ** 2 * np.exp(
                                                          -2 * x)) / sigma ** 2 - (a * b * np.exp(-x) * (
                                                                  sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 + (
                                                                                                                          a * b * np.exp(
                                                                                                                      -x) * (
                                                                                                                                      sigma ** 2 / 2 - a * np.exp(
                                                                                                                                  -x) * (
                                                                                                                                                  b - np.exp(
                                                                                                                                              x)))) / sigma ** 2 - (
                                                                                                                          6 * a * b * np.exp(
                                                                                                                      -x) * (
                                                                                                                                      a + a * np.exp(
                                                                                                                                  -x) * (
                                                                                                                                                  b - np.exp(
                                                                                                                                              x)))) / sigma ** 2)) / 2 + (
                                                              a + a * np.exp(-x) * (b - np.exp(x))) * (a + (
                                         (a + a * np.exp(-x) * (b - np.exp(x))) * (
                                             sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 - (a * b * np.exp(
                                 -x)) / 2 + a * np.exp(-x) * (b - np.exp(x)) + (a ** 2 * b ** 2 * np.exp(-2 * x)) / sigma ** 2 - (
                                                                                                             a * b * np.exp(
                                                                                                         -x) * (
                                                                                                                         sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                     -x))) / sigma ** 2 + (
                                                                                                             a * b * np.exp(
                                                                                                         -x) * (
                                                                                                                         sigma ** 2 / 2 - a * np.exp(
                                                                                                                     -x) * (
                                                                                                                                     b - np.exp(
                                                                                                                                 x)))) / sigma ** 2 - (
                                                                                                             2 * a * b * np.exp(
                                                                                                         -x) * (
                                                                                                                         a + a * np.exp(
                                                                                                                     -x) * (
                                                                                                                                     b - np.exp(
                                                                                                                                 x)))) / sigma ** 2) + (
                                                              sigma ** 2 * (a + ((a + a * np.exp(-x) * (b - np.exp(x))) * (
                                                                  sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 - (
                                                                                       a * b * np.exp(-x)) / 2 + a * np.exp(
                                                          -x) * (b - np.exp(x)) + (7 * a ** 2 * b ** 2 * np.exp(
                                                          -2 * x)) / sigma ** 2 - (a * b * np.exp(-x) * (
                                                                  sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 + (
                                                                                       a * b * np.exp(-x) * (
                                                                                           sigma ** 2 / 2 - a * np.exp(
                                                                                       -x) * (b - np.exp(
                                                                                       x)))) / sigma ** 2 - (
                                                                                       14 * a * b * np.exp(-x) * (
                                                                                           a + a * np.exp(-x) * (b - np.exp(
                                                                                       x)))) / sigma ** 2)) / 4 + (
                                                              (a + a * np.exp(-x) * (b - np.exp(x))) * (a + (
                                                                  (a + a * np.exp(-x) * (b - np.exp(x))) * (
                                                                      sigma ** 2 / 2 + a - a * b * np.exp(
                                                                  -x))) / sigma ** 2 - (a * b * np.exp(-x)) / 2 + a * np.exp(
                                                          -x) * (b - np.exp(x)) - (a * b * np.exp(-x) *
                                                                                (sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 + (a * b * np.exp(-x) * (sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2)) / 2 - a * b * np.exp(
                                 -x) * (a + ((a + a * np.exp(-x) * (b - np.exp(x))) * (
                                         sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 - (
                                                    a * b * np.exp(-x)) / 2 + a * np.exp(-x) * (b - np.exp(x)) + (
                                                    a ** 2 * b ** 2 * np.exp(-2 * x)) / sigma ** 2 - (a * b * np.exp(-x) * (
                                         sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 + (a * b * np.exp(-x) * (
                                         sigma ** 2 / 2 - a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2 - (
                                                    2 * a * b * np.exp(-x) * (
                                                        a + a * np.exp(-x) * (b - np.exp(x)))) / sigma ** 2) - (
                                                              a * b * np.exp(-x) * (a + (
                                                                  (a + a * np.exp(-x) * (b - np.exp(x))) * (
                                                                      sigma ** 2 / 2 + a - a * b * np.exp(
                                                                  -x))) / sigma ** 2 - (a * b * np.exp(-x)) / 2 + a * np.exp(
                                                          -x) * (b - np.exp(x)) - (a * b * np.exp(-x) * (
                                                                  sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 + (
                                                                                             a * b * np.exp(-x) * (
                                                                                                 sigma ** 2 / 2 - a * np.exp(
                                                                                             -x) * (b - np.exp(
                                                                                             x)))) / sigma ** 2)) / 2)) / 6 - (
                                     sigma ** 2 * ((a + ((a + a * np.exp(-x) * (b - np.exp(x))) * (
                                         sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 - (
                                                               a * b * np.exp(-x)) / 2 + a * np.exp(-x) * (b - np.exp(x)) - (
                                                               a * b * np.exp(-x) * (
                                                                   sigma ** 2 / 2 + a - a * b * np.exp(-x))) / sigma ** 2 + (
                                                               a * b * np.exp(-x) * (sigma ** 2 / 2 - a * np.exp(-x) * (
                                                                   b - np.exp(x)))) / sigma ** 2) ** 2 + (
                                                              2 * (sigma ** 2 / 2 + a - a * b * np.exp(-x)) * (((sigma ** 2 / 2 + a - a * b * np.exp(-x)) * (a + ((
                                                                                                                                        a + a * np.exp(
                                                                                                                                    -x) * (
                                                                                                                                                    b - np.exp(
                                                                                                                                                x))) * (
                                                                                                                                        sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                                    -x))) / sigma ** 2 - (
                                                                                                                                    a * b * np.exp(
                                                                                                                                -x)) / 2 + a * np.exp(
                                                                                                                    -x) * (
                                                                                                                                    b - np.exp(
                                                                                                                                x)) + (
                                                                                                                                    a ** 2 * b ** 2 * np.exp(
                                                                                                                                -2 * x)) / sigma ** 2 - (
                                                                                                                                    a * b * np.exp(
                                                                                                                                -x) * (
                                                                                                                                                sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                                            -x))) / sigma ** 2 + (
                                                                                                                                    a * b * np.exp(
                                                                                                                                -x) * (
                                                                                                                                                sigma ** 2 / 2 - a * np.exp(
                                                                                                                                            -x) * (
                                                                                                                                                            b - np.exp(
                                                                                                                                                        x)))) / sigma ** 2 - (
                                                                                                                                    2 * a * b * np.exp(
                                                                                                                                -x) * (
                                                                                                                                                a + a * np.exp(
                                                                                                                                            -x) * (
                                                                                                                                                            b - np.exp(
                                                                                                                                                        x)))) / sigma ** 2)) / 2 - (
                                                                                                                       (
                                                                                                                                   sigma ** 2 / 2 - a * np.exp(
                                                                                                                               -x) * (
                                                                                                                                               b - np.exp(
                                                                                                                                           x))) * (
                                                                                                                                   a + (
                                                                                                                                       (
                                                                                                                                                   a + a * np.exp(
                                                                                                                                               -x) * (
                                                                                                                                                               b - np.exp(
                                                                                                                                                           x))) * (
                                                                                                                                                   sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                                               -x))) / sigma ** 2 - (
                                                                                                                                               a * b * np.exp(
                                                                                                                                           -x)) / 2 + a * np.exp(
                                                                                                                               -x) * (
                                                                                                                                               b - np.exp(
                                                                                                                                           x)) + (
                                                                                                                                               a ** 2 * b ** 2 * np.exp(
                                                                                                                                           -2 * x)) / sigma ** 2 - (
                                                                                                                                               a * b * np.exp(
                                                                                                                                           -x) * (
                                                                                                                                                           sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                                                       -x))) / sigma ** 2 + (
                                                                                                                                               a * b * np.exp(
                                                                                                                                           -x) * (
                                                                                                                                                           sigma ** 2 / 2 - a * np.exp(
                                                                                                                                                       -x) * (
                                                                                                                                                                       b - np.exp(
                                                                                                                                                                   x)))) / sigma ** 2 - (
                                                                                                                                               2 * a * b * np.exp(
                                                                                                                                           -x) * (
                                                                                                                                                           a + a * np.exp(
                                                                                                                                                       -x) * (
                                                                                                                                                                       b - np.exp(
                                                                                                                                                                   x)))) / sigma ** 2)) / 2 + (
                                                                                                                       sigma ** 2 * (
                                                                                                                           a + (
                                                                                                                               (
                                                                                                                                           a + a * np.exp(
                                                                                                                                       -x) * (
                                                                                                                                                       b - np.exp(
                                                                                                                                                   x))) * (
                                                                                                                                           sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                                       -x))) / sigma ** 2 - (
                                                                                                                                       a * b * np.exp(
                                                                                                                                   -x)) / 2 + a * np.exp(
                                                                                                                       -x) * (
                                                                                                                                       b - np.exp(
                                                                                                                                   x)) + (
                                                                                                                                       3 * a ** 2 * b ** 2 * np.exp(
                                                                                                                                   -2 * x)) / sigma ** 2 - (
                                                                                                                                       a * b * np.exp(
                                                                                                                                   -x) * (
                                                                                                                                                   sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                                               -x))) / sigma ** 2 + (
                                                                                                                                       a * b * np.exp(
                                                                                                                                   -x) * (
                                                                                                                                                   sigma ** 2 / 2 - a * np.exp(
                                                                                                                                               -x) * (
                                                                                                                                                               b - np.exp(
                                                                                                                                                           x)))) / sigma ** 2 - (
                                                                                                                                       6 * a * b * np.exp(
                                                                                                                                   -x) * (
                                                                                                                                                   a + a * np.exp(
                                                                                                                                               -x) * (
                                                                                                                                                               b - np.exp(
                                                                                                                                                           x)))) / sigma ** 2)) / 4 + (
                                                                                                                       (
                                                                                                                                   a + a * np.exp(
                                                                                                                               -x) * (
                                                                                                                                               b - np.exp(
                                                                                                                                           x))) * (
                                                                                                                                   a + (
                                                                                                                                       (
                                                                                                                                                   a + a * np.exp(
                                                                                                                                               -x) * (
                                                                                                                                                               b - np.exp(
                                                                                                                                                           x))) * (
                                                                                                                                                   sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                                               -x))) / sigma ** 2 - (
                                                                                                                                               a * b * np.exp(
                                                                                                                                           -x)) / 2 + a * np.exp(
                                                                                                                               -x) * (
                                                                                                                                               b - np.exp(
                                                                                                                                           x)) - (
                                                                                                                                               a * b * np.exp(
                                                                                                                                           -x) * (
                                                                                                                                                           sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                                                       -x))) / sigma ** 2 + (
                                                                                                                                               a * b * np.exp(
                                                                                                                                           -x) * (
                                                                                                                                                           sigma ** 2 / 2 - a * np.exp(
                                                                                                                                                       -x) * (
                                                                                                                                                                       b - np.exp(
                                                                                                                                                                   x)))) / sigma ** 2)) / 2 - (
                                                                                                                       a * b * np.exp(
                                                                                                                   -x) * (
                                                                                                                                   a + (
                                                                                                                                       (
                                                                                                                                                   a + a * np.exp(
                                                                                                                                               -x) * (
                                                                                                                                                               b - np.exp(
                                                                                                                                                           x))) * (
                                                                                                                                                   sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                                               -x))) / sigma ** 2 - (
                                                                                                                                               a * b * np.exp(
                                                                                                                                           -x)) / 2 + a * np.exp(
                                                                                                                               -x) * (
                                                                                                                                               b - np.exp(
                                                                                                                                           x)) - (
                                                                                                                                               a * b * np.exp(
                                                                                                                                           -x) * (
                                                                                                                                                           sigma ** 2 / 2 + a - a * b * np.exp(
                                                                                                                                                       -x))) / sigma ** 2 + (
                                                                                                                                               a * b * np.exp(
                                                                                                                                           -x) * (
                                                                                                                                                           sigma ** 2 / 2 - a * np.exp(
                                                                                                                                                       -x) * (
                                                                                                                                                                       b - np.exp(
                                                                                                                                                                   x)))) / sigma ** 2)) / 2)) / sigma ** 2)) / 6

                 ) * t**3

    return w_res


def w_ex_x0(mr, theta, vov, x, x0, t, n: int = 3):
    """
    Exponent Expansion on the nth order
    to approximately get W(x, x0, t) of transition density.

    Parameters
    ----------
    mr : speed
    theta : long-term stable vol
    sigma : initial vol
    x : float
        fractile point in terms of transformation, xt = ln(yt).
    x0 : float
        initial point of x.
    t : float
        time to expiry.
    n : int
        the highest order for Exponent Expansion.
    Returns
    -------
    w_res : float
        w = symsum(wn*t**n, 0, inf)
    """
    a = mr
    b = theta
    sigma = vov

    w0 = ((x - x0)*(sigma**2/2 + a) + a*b*(np.exp(-x) - np.exp(-x0)))/sigma**2
    if n > 0.5:
        w1 = (
            a/2 + sigma**2/8 + a**2/(2*sigma**2) - (a**2*b**2*np.exp(-2*x))/(4*(sigma**2*x - sigma**2*x0)) +
            (a**2*b**2*np.exp(-2*x0))/(4*(sigma**2*x - sigma**2*x0)) + (a*b*np.exp(-x))/(x - x0) -
            (a*b*np.exp(-x0))/(x - x0) + (a**2*b*np.exp(-x))/(sigma**2*x - sigma**2*x0) -
            (a**2*b*np.exp(-x0))/(sigma**2*x - sigma**2*x0)
        )
        w_res = w0 + w1 * t
    if n > 1:
        w2 = (
            (a**2*b**2*sigma**2*(np.exp(-2*x)/(2*sigma**2*(x - x0)**2) - np.exp(-2*x0)/(2*sigma**2*(x - x0)**2)))/2 -
            (a**2*b*sigma**2*(np.exp(-x)/(sigma**2*(x - x0)**2) - np.exp(-x0)/(sigma**2*(x - x0)**2)))/2 -
            (a*b*sigma**2*(np.exp(-x) - np.exp(-x0)))/(2*(x - x0)**2) -
            (a*b*np.exp(-2*x0)*(np.exp(2*x0 - x)*(4*sigma**2 + 4*a) - np.exp(x0)*(4*a + 4*a*x0 + 4*sigma**2*x0 + 4*sigma**2) +
                             2*a*b*x0 + 2*a*b*np.exp(x0 - x)*(np.exp(x - x0)/2 - np.exp(x0 - x)/2)))/(4*(x - x0)**3) -
            (a*b*x*np.exp(-2*x0)*(4*np.exp(x0)*sigma**2 - 2*a*b + 4*a*np.exp(x0)))/(4*(x - x0)**3)
        )
        w_res += w2 * t ** 2
    if n > 2:
        w3 = (
            np.exp(4 * x0 - 4 * x) * ((a ** 4 * b ** 4 * np.exp(-4 * x0)) / (32 * sigma ** 2 * (x - x0) ** 3) +
                                   (a ** 4 * b ** 4 * np.exp(-4 * x0)) / (32 * sigma ** 2 * (x - x0) ** 4)) +
            np.exp(2 * x0 - 2 * x) * ((a ** 2 * b ** 2 * np.exp(-4 * x0) *
                                    (- a ** 2 * b ** 2 + 4 * np.exp(x0) * a ** 2 * b + 8 * np.exp(2 * x0) * a ** 2 +
                                     4 * np.exp(x0) * a * b * sigma ** 2 + 16 * np.exp(2 * x0) * a * sigma ** 2 +
                                     8 * np.exp(2 * x0) * sigma ** 4)) / (16 * sigma ** 2 * (x - x0) ** 4) +
                                   (a ** 2 * b ** 2 * np.exp(-2 * x0) * (sigma ** 2 + a) ** 2) /
                                   (4 * sigma ** 2 * (x - x0) ** 3)) - np.exp(3 * x0 - 3 * x) *
            ((a ** 3 * b ** 3 * np.exp(-3 * x0) * (sigma ** 2 + a)) / (6 * sigma ** 2 * (x - x0) ** 3) +
             (a ** 3 * b ** 3 * np.exp(-3 * x0) * (sigma ** 2 + a)) / (4 * sigma ** 2 * (x - x0) ** 4)) +
            (a ** 4 * b ** 4 - 8 * np.exp(x0) * a ** 4 * b ** 3 + 16 * np.exp(2 * x0) * a ** 4 * b ** 2 -
             8 * np.exp(x0) * a ** 3 * b ** 3 * sigma ** 2 + 32 * np.exp(2 * x0) * a ** 3 * b ** 2 * sigma ** 2 +
             16 * np.exp(2 * x0) * a ** 2 * b ** 2 * sigma ** 4) /
            (32 * sigma ** 2 * x ** 4 * np.exp(4 * x0) + 32 * sigma ** 2 * x0 ** 4 * np.exp(4 * x0) +
             192 * sigma ** 2 * x ** 2 * x0 ** 2 * np.exp(4 * x0) - 128 * sigma ** 2 * x * x0 ** 3 * np.exp(4 * x0) -
             128 * sigma ** 2 * x ** 3 * x0 * np.exp(4 * x0)) +
            ((a * b * sigma ** 2 * np.exp(-2 * x0) *
              (np.exp(x0) * (6 * a * x - 6 * a * x0 + 6 * sigma ** 2 * x - 6 * sigma ** 2 * x0) +
               np.exp(2 * x0 - x) * (6 * a * x - 6 * a * x0 + 6 * sigma ** 2 * x - 6 * sigma ** 2 * x0) -
               6 * a * b * x * np.cosh(x - x0) * np.exp(x0 - x) + 6 * a * b * x0 * np.cosh(x - x0) * np.exp(x0 - x))) / 4 +
             (a * b * sigma ** 2 * np.exp(-2 * x0) * (np.exp(2 * x0 - x) * (12 * sigma ** 2 + 12 * a) - np.exp(x0) * (
                12 * sigma ** 2 + 12 * a) + 6 * a * b * np.sinh(x - x0) * np.exp(x0 - x))) / 4) / (x - x0) ** 5 + (
                        a * b * sigma ** 4 * (np.exp(-x) - np.exp(-x0))) / (4 * (x - x0) ** 3) - (
                        a ** 2 * b ** 2 * np.exp(-4 * x0) * (
                            3 * a ** 2 * b ** 2 - 16 * np.exp(x0) * a ** 2 * b + 24 * np.exp(2 * x0) * a ** 2 - 16 * np.exp(
                        x0) * a * b * sigma ** 2 + 48 * np.exp(2 * x0) * a * sigma ** 2 + 24 * np.exp(
                        2 * x0) * sigma ** 4)) / (96 * sigma ** 2 * (x - x0) ** 3) + (
                        a * b * sigma ** 2 * np.exp(-2 * x0) * (
                            a * x ** 2 * np.exp(2 * x0 - x) + a * x0 ** 2 * np.exp(2 * x0 - x) - 2 * a * x * x0 * np.exp(
                        2 * x0 - x) - a * b * x ** 2 * np.exp(2 * x0 - 2 * x) - a * b * x0 ** 2 * np.exp(
                        2 * x0 - 2 * x) + 2 * a * b * x * x0 * np.exp(2 * x0 - 2 * x))) / (4 * (x - x0) ** 5) + (
                        a ** 2 * b * sigma ** 2 * np.exp(-2 * x0) * (b - np.exp(x0))) / (4 * (x - x0) ** 3) - (
                        a ** 2 * b ** 2 * np.exp(- x - 2 * x0) * (sigma ** 2 + a) * (
                            4 * np.exp(x0) * sigma ** 2 - a * b + 4 * a * np.exp(x0))) / (4 * sigma ** 2 * (x - x0) ** 4)

        )
        w_res += w3 * t ** 3
    if n > 3:
        w4 = (
            (
                        4 * a ** 4 * b ** 4 * x0 - 4 * a ** 4 * b ** 4 * x + 5 * a ** 4 * b ** 4 * x ** 2 + 5 * a ** 4 * b ** 4 * x0 ** 2 + 240 * a * b * sigma ** 6 * np.exp(
                    3 * x0) + 32 * a ** 4 * b ** 3 * x * np.exp(x0) - 32 * a ** 4 * b ** 3 * x0 * np.exp(
                    x0) + 240 * a ** 2 * b * sigma ** 4 * np.exp(3 * x0) - 64 * a ** 4 * b ** 2 * x * np.exp(
                    2 * x0) + 64 * a ** 4 * b ** 2 * x0 * np.exp(2 * x0) - 28 * a ** 4 * b ** 3 * x ** 2 * np.exp(
                    x0) - 28 * a ** 4 * b ** 3 * x0 ** 2 * np.exp(
                    x0) - 10 * a ** 4 * b ** 4 * x * x0 - 60 * a ** 2 * b ** 2 * sigma ** 4 * np.exp(
                    2 * x0) + 40 * a ** 4 * b ** 2 * x ** 2 * np.exp(2 * x0) + 40 * a ** 4 * b ** 2 * x0 ** 2 * np.exp(
                    2 * x0) - 120 * a ** 2 * b * sigma ** 4 * x * np.exp(
                    3 * x0) + 120 * a ** 2 * b * sigma ** 4 * x0 * np.exp(3 * x0) + 24 * a * b * sigma ** 6 * x ** 2 * np.exp(
                    3 * x0) + 24 * a * b * sigma ** 6 * x0 ** 2 * np.exp(
                    3 * x0) + 32 * a ** 3 * b ** 3 * sigma ** 2 * x * np.exp(
                    x0) - 32 * a ** 3 * b ** 3 * sigma ** 2 * x0 * np.exp(x0) - 80 * a ** 4 * b ** 2 * x * x0 * np.exp(
                    2 * x0) - 128 * a ** 3 * b ** 2 * sigma ** 2 * x * np.exp(
                    2 * x0) + 128 * a ** 3 * b ** 2 * sigma ** 2 * x0 * np.exp(
                    2 * x0) - 4 * a ** 2 * b ** 2 * sigma ** 4 * x * np.exp(
                    2 * x0) + 4 * a ** 2 * b ** 2 * sigma ** 4 * x0 * np.exp(
                    2 * x0) + 24 * a ** 2 * b * sigma ** 4 * x ** 2 * np.exp(
                    3 * x0) + 24 * a ** 2 * b * sigma ** 4 * x0 ** 2 * np.exp(
                    3 * x0) - 28 * a ** 3 * b ** 3 * sigma ** 2 * x ** 2 * np.exp(
                    x0) - 28 * a ** 3 * b ** 3 * sigma ** 2 * x0 ** 2 * np.exp(x0) - 120 * a * b * sigma ** 6 * x * np.exp(
                    3 * x0) + 120 * a * b * sigma ** 6 * x0 * np.exp(3 * x0) + 56 * a ** 4 * b ** 3 * x * x0 * np.exp(
                    x0) + 80 * a ** 3 * b ** 2 * sigma ** 2 * x ** 2 * np.exp(
                    2 * x0) + 80 * a ** 3 * b ** 2 * sigma ** 2 * x0 ** 2 * np.exp(
                    2 * x0) + 16 * a ** 2 * b ** 2 * sigma ** 4 * x ** 2 * np.exp(
                    2 * x0) + 16 * a ** 2 * b ** 2 * sigma ** 4 * x0 ** 2 * np.exp(
                    2 * x0) - 48 * a * b * sigma ** 6 * x * x0 * np.exp(
                    3 * x0) - 48 * a ** 2 * b * sigma ** 4 * x * x0 * np.exp(
                    3 * x0) + 56 * a ** 3 * b ** 3 * sigma ** 2 * x * x0 * np.exp(
                    x0) - 160 * a ** 3 * b ** 2 * sigma ** 2 * x * x0 * np.exp(
                    2 * x0) - 32 * a ** 2 * b ** 2 * sigma ** 4 * x * x0 * np.exp(2 * x0)) / (
                        16 * x ** 7 * np.exp(4 * x0) - 16 * x0 ** 7 * np.exp(4 * x0) - 336 * x ** 2 * x0 ** 5 * np.exp(
                    4 * x0) + 560 * x ** 3 * x0 ** 4 * np.exp(4 * x0) - 560 * x ** 4 * x0 ** 3 * np.exp(
                    4 * x0) + 336 * x ** 5 * x0 ** 2 * np.exp(4 * x0) + 112 * x * x0 ** 6 * np.exp(
                    4 * x0) - 112 * x ** 6 * x0 * np.exp(4 * x0)) - np.exp(x0 - x) * ((a * b * np.exp(-3 * x0) * (
                sigma ** 2 + a) * (- a ** 2 * b ** 2 + 6 * np.exp(2 * x0) * sigma ** 4)) / (4 * (x - x0) ** 5) + (
                                                                                           a * b * sigma ** 4 * np.exp(
                                                                                       -x0) * (sigma ** 2 + a)) / (
                                                                                           8 * (x - x0) ** 4) + (
                                                                                           15 * a * b * sigma ** 4 * np.exp(
                                                                                       -x0) * (sigma ** 2 + a)) / (
                                                                                           x - x0) ** 7 + (
                                                                                           a * b * np.exp(-3 * x0) * (
                                                                                               sigma ** 2 + a) * (
                                                                                                       4 * a ** 2 * b ** 2 - 16 * np.exp(
                                                                                                   x0) * a ** 2 * b - 16 * np.exp(
                                                                                                   x0) * a * b * sigma ** 2 + 15 * np.exp(
                                                                                                   2 * x0) * sigma ** 4)) / (
                                                                                           2 * (x - x0) ** 6)) - np.exp(
        2 * x0 - 2 * x) * ((a ** 2 * b ** 2 * np.exp(-2 * x0) * (2 * a ** 2 + 4 * a * sigma ** 2 + sigma ** 4)) / (
                4 * (x - x0) ** 4) + (a ** 2 * b ** 2 * np.exp(-4 * x0) * (
                - 2 * a ** 2 * b ** 2 + 8 * np.exp(x0) * a ** 2 * b + 16 * np.exp(2 * x0) * a ** 2 + 8 * np.exp(
            x0) * a * b * sigma ** 2 + 32 * np.exp(2 * x0) * a * sigma ** 2 + np.exp(2 * x0) * sigma ** 4)) / (
                                       4 * (x - x0) ** 6) + (a ** 2 * b ** 2 * np.exp(-3 * x0) * (
                4 * sigma ** 4 * np.exp(x0) + a ** 2 * b + 10 * a ** 2 * np.exp(x0) + 20 * a * sigma ** 2 * np.exp(
            x0) + a * b * sigma ** 2)) / (4 * (x - x0) ** 5) - (15 * a ** 2 * b ** 2 * sigma ** 4 * np.exp(-2 * x0)) / (
                                       4 * (x - x0) ** 7)) + np.exp(3 * x0 - 3 * x) * (
                        (a ** 3 * b ** 3 * np.exp(-3 * x0) * (sigma ** 2 + a)) / (2 * (x - x0) ** 4) + (
                            7 * a ** 3 * b ** 3 * np.exp(-3 * x0) * (sigma ** 2 + a)) / (4 * (x - x0) ** 5) + (
                                    2 * a ** 3 * b ** 3 * np.exp(-3 * x0) * (sigma ** 2 + a)) / (x - x0) ** 6) - np.exp(
        4 * x0 - 4 * x) * (
                        (a ** 4 * b ** 4 * np.exp(-4 * x0)) / (8 * (x - x0) ** 4) + (5 * a ** 4 * b ** 4 * np.exp(-4 * x0)) / (
                            16 * (x - x0) ** 5) + (a ** 4 * b ** 4 * np.exp(-4 * x0)) / (4 * (x - x0) ** 6)) - (
                        a * b * np.exp(-4 * x0) * (
                            a ** 3 * b ** 3 - 4 * np.exp(x0) * a ** 3 * b ** 2 + 4 * np.exp(2 * x0) * a ** 3 * b - 4 * np.exp(
                        x0) * a ** 2 * b ** 2 * sigma ** 2 + 8 * np.exp(2 * x0) * a ** 2 * b * sigma ** 2 + 2 * np.exp(
                        2 * x0) * a * b * sigma ** 4 + np.exp(3 * x0) * a * sigma ** 4 + np.exp(3 * x0) * sigma ** 6)) / (
                        8 * (x - x0) ** 4)

        )
        w_res += w4*t**4

    return w_res