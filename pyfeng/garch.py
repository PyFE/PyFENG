import numpy as np
import scipy.stats as spst
from . import sv_abc as sv
from . import bsm


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


class GarchMcTimeStep(sv.SvABC, sv.CondMcBsmABC):
    """
    Garch model with conditional Monte-Carlo simulation
    The SDE of SV is: dv_t = mr * (theta - v_t) dt + vov * v_t dB_T
    """

    model_type = "GarchDiff"
    var_process = True
    scheme = 1  #

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

    def cond_spot_sigma(self, var_0, texp):

        vol_final, mean_var, mean_vol, mean_inv_vol = self.cond_states(var_0, texp)

        spot_cond = 2 * (vol_final - np.sqrt(var_0)) / self.vov \
            - self.mr * self.theta * mean_inv_vol * texp / self.vov \
            + (self.mr / self.vov + self.vov / 4) * mean_vol * texp \
            - self.rho * mean_var * texp / 2
        np.exp(self.rho * spot_cond, out=spot_cond)

        sigma_cond = np.sqrt((1.0 - self.rho**2)/var_0*mean_var)

        return spot_cond, sigma_cond
