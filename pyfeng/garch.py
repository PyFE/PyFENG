import numpy as np
from . import sv_abc as sv
from . import bsm

#### Use of RN generation spawn:
# 0: simulation of variance (gamma/ncx2/normal)

class GarchUncorrBaroneAdesi2004(sv.SvABC):
    """
    Barone-Adesi et al. (2004)'s approximation pricing formula for European options under uncorrelated (rho=0) GARCH diffusion model.
    Up to 2nd order is implemented.

    References:
        - Barone-Adesi G, Rasmussen H, Ravanelli C (2005) An option pricing formula for the GARCH diffusion model. Computational Statistics & Data Analysis 49:287–310. https://doi.org/10.1016/j.csda.2004.05.014

    See Also: OusvUncorrBallRoma1994, HestonUncorrBallRoma1994
    """

    model_type = "GarchDiff"
    var_process = True
    order = 2

    def avgvar_mv(self, texp, var0):
        """
        Mean and variance of the average variance given V(0) = var0.
        Eqs. (12)-(13) in Barone-Adesi et al. (2005)

        Args:
            texp: time step
            var0: initial variance

        Returns:
            mean, variance
        """

        mr, vov, theta = self.mr, self.vov, self.theta

        mr2 = mr*mr
        vov2 = vov*vov
        theta2 = theta*theta
        mr_t = self.mr*texp
        e_mr = np.exp(-mr_t)

        # Eq (12) of Barone-Adesi et al. (2005)
        M1 = theta + (var0 - theta)*(1 - e_mr)/mr_t

        term1 = vov2 - mr
        term2 = vov2 - 2*mr

        # Eq (13)
        M2c_1 = -(e_mr*(var0 - theta))**2
        M2c_2 = 2*np.exp(term2*texp)*(2*mr*theta*(mr*theta + term2*var0) + term1*term2*var0**2)
        M2c_3 = -vov2*(theta2*(4*mr*(3 - mr_t) + (2*mr_t - 5)*vov2) + term2*var0*(2*theta + var0))

        M2c_4 = 2*e_mr*vov2
        M2c_4 *= 2*theta2*(mr_t*mr - (1 + mr_t)*vov2) + var0*(2*mr*theta*(1 + texp*term1) + term1*var0)

        M2c = M2c_1/mr2 + M2c_2/(term1*term2)**2 + M2c_3/mr2/term2**2 + M2c_4/mr2/term1**2
        M2c /= texp**2

        return M1, M2c

    def price(self, strike, spot, texp, cp=1):

        if not np.isclose(self.rho, 0.0):
            print(f"Pricing ignores rho = {self.rho}.")

        avgvar, var = self.avgvar_mv(texp, self.sigma)

        m_bs = bsm.Bsm(np.sqrt(avgvar), intr=self.intr, divr=self.divr)
        price = m_bs.price(strike, spot, texp, cp)

        if self.order == 2:
            price += 0.5*var*m_bs.d2_var(strike, spot, texp, cp)
        elif self.order > 2:
            raise ValueError(f"Not implemented for approx order: {self.order}")

        return price


class GarchMcTimeDisc(sv.SvABC, sv.CondMcBsmABC):
    """
    Garch model with conditional Monte-Carlo simulation
    The SDE of SV is: dv_t = mr * (theta - v_t) dt + vov * v_t dB_T
    """

    model_type = "GarchDiff"
    var_process = True
    scheme = 1  #

    def set_num_params(self, n_path=10000, dt=0.05, rn_seed=None, antithetic=True, scheme=1):
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
        super().set_num_params(n_path, dt, rn_seed, antithetic)
        self.scheme = scheme

    def vol_step_euler(self, dt, var_0, milstein=True):
        """
        Euler/Milstein Schemes:
        v_(t+dt) = v_t + mr * (theta - v_t) * dt + vov * v_t Z * sqrt(dt) + (vov^2/2) v_t (Z^2-1) dt

        Args:
            var_0: initial variance
            dt: delta t, time step

        Returns: Variance path (time, path) including the value at t=0
        """

        zz = self.rv_normal(spawn=0)
        var_t = var_0 + self.mr*(self.theta - var_0)*dt + self.vov*var_0*np.sqrt(dt)*zz
        if milstein:
            var_t += (self.vov**2/2)*var_0*dt*(zz**2 - 1)

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

        zz = self.rv_normal(spawn=0)

        log_var_t = log_var_0 + (self.mr*self.theta*np.exp(-log_var_0) - self.mr - self.vov**2/2)*dt \
                    + self.vov*np.sqrt(dt)*zz

        return log_var_t

    def cond_states_step(self, dt, var_0):

        if self.scheme < 2:
            milstein = (self.scheme == 1)
            var_t = self.vol_step_euler(dt, var_0, milstein=milstein)

            mean_var = (var_0 + var_t)/2
            vol_0 = np.sqrt(var_0)
            vol_t = np.sqrt(var_t)
            mean_vol = (vol_0 + vol_t)/2
            mean_inv_vol = (1/vol_0 + 1/vol_t)/2

        elif self.scheme == 2:

            log_var_t = self.var_step_log(np.log(var_0), dt)

            vol_0 = np.sqrt(var_0)
            vol_t = np.exp(log_var_t/2)
            var_t = vol_t**2

            mean_var = (var_0 + var_t)/2
            mean_vol = (vol_0 + vol_t)/2
            mean_inv_vol = (1/vol_0 + 1/vol_t)/2
        else:
            raise ValueError(f'Invalid scheme: {self.scheme}')

        return var_t, mean_var, mean_vol, mean_inv_vol

    def cond_spot_sigma(self, texp, var_0):
        tobs = self.tobs(texp)
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        var_t = np.full(self.n_path, var_0)
        avgvar = np.zeros(self.n_path)
        avgvol = np.zeros(self.n_path)
        avgivol = np.zeros(self.n_path)

        for i in range(n_dt):
            var_t, avgvar_inc, avgvol_inc, avgivol_inc = self.cond_states_step(dt[i], var_t)
            avgvar += avgvar_inc * dt[i]
            avgvol += avgvol_inc * dt[i]
            avgivol += avgivol_inc * dt[i]

        avgvar /= texp
        avgvol /= texp
        avgivol /= texp

        spot_cond = 2 * (np.sqrt(var_t) - np.sqrt(var_0)) / self.vov + \
            (-self.mr * self.theta * avgivol / self.vov \
            + (self.mr/self.vov + self.vov/4) * avgvol - self.rho * avgvar / 2) * texp
        np.exp(self.rho*spot_cond, out=spot_cond)

        cond_sigma = np.sqrt((1.0 - self.rho**2)/var_0*avgvar)

        return spot_cond, cond_sigma

    def return_var_realized(self, texp, cond):
        return None