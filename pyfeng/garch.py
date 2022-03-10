import numpy as np
import scipy.stats as spst
import scipy.integrate as scint
from . import sv_abc as sv
from . import bsm


class GarchApproxUncor(sv.SvABC):
    """
    The implementation of Barone-Adesi et al (2004)'s approximation pricing formula for European
    options under uncorrelated (rho=0) GARCH diffusion model.

    References:
        - Barone-Adesi G, Rasmussen H, Ravanelli C (2005) An option pricing formula for the GARCH diffusion model. Computational Statistics & Data Analysis 49:287–310. https://doi.org/10.1016/j.csda.2004.05.014

    This method is only used to compare with the method GarchCondMC.
    """

    def price(self, strike, spot, texp, cp=1):
        V0, mr, vov, theta = self.sigma, self.mr, self.vov, self.theta

        mr2 = mr * mr
        vov2 = vov * vov
        theta2 = theta*theta
        decay = np.exp(-mr * texp)

        # Eq (12) of Barone-Adesi et al. (2005)
        M1 = theta + (V0 - theta) * (1 - decay) / (mr * texp)

        term1 = vov2 - mr
        term2 = vov2 - 2 * mr

        # Eq (13)
        M2c_1 = - (decay * (V0 - theta)) ** 2
        M2c_2 = 2 * np.exp(term2 * texp) * (2 * mr * theta * (mr * theta + term2 * V0) + term1 * term2 * V0 ** 2)
        M2c_3 = -vov2 * (theta2 * (4 * mr * (3 - texp * mr) + (2 * texp * mr - 5) * vov2)
                               + 2 * theta * term2 * V0 + term2 * V0 ** 2)

        M2c_4 = 2 * decay * vov2 * mr2
        M2c_4 *= 2 * theta2 * (texp * mr2 - (1 + texp * mr) * vov2) \
                 + 2 * mr * theta * (1 + texp * term1) * V0 \
                 + term1 * V0 ** 2

        M2c = M2c_1 / mr2 + M2c_2 / (term1 * term2) ** 2 + M2c_3 / mr2 / (term2) ** 2 \
              + M2c_4 / (mr2 * term1) ** 2

        M2c /= texp ** 2
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
        c_bs_p2 = c_bs_p1 * texp / 2 * ((m / M1 / texp) ** 2 - 1 / M1 / texp - 1 / 4)

        # C_bs_p3=C_bs_p1*(m**4/(4*(M1*texp)**4)-m**2*(12+M1*texp)/(8*(M1*texp)**3)+(48+8*M1*texp+(M1*texp)**2)/(64*(M1*texp)**2))*texp**2
        # C_bs_p4=C_bs_p1*(m**6/(8*(M1*texp)**6)-3*m**4*(20+M1*texp)/(32*(M1*texp)**5)+3*m**2*(240+24*M1*texp+(M1*texp)**2)/(128*(M1*texp)**4)-(960+144*M1*texp+12*(M1*texp)**2+(M1*texp)**3)/(512*(M1*texp)**3))*texp**3

        c_ga_2 = c_bs + (M2c / 2) * c_bs_p2
        # C_ga_3=C_ga_2+(M3c/6)*C_bs_p3
        # C_ga_4=C_ga_3+(M4c/24)*C_bs_p4

        return c_ga_2


class GarchCondMC(sv.SvABC, sv.CondMcBsmABC):
    """
    Garch model with conditional Monte-Carlo simulation
    The SDE of SV is: dv_t = mr * (theta - v_t) dt + vov * v_t dB_T
    """

    var_process = True

    def vol_paths(self, tobs):
        """
        Milstein Schemes:
        w_(t+dt) = w_t + (mr * theta * exp(-w_t) - mr - vov^2 / 2) * dt + vov * Z * sqrt(dt)
        v_t = exp(w_t)
        Args:
            mr: coefficient of dt
            theta: the long term average
            Z : std normal distributed RN
            dt : delta t, time step

        Returns: Variance path (time, path) including the value at t=0
        """
        n_dt = len(tobs)
        n_path = self.n_path
        rn_norm = self._bm_incr(tobs=np.arange(1, n_dt + 0.1), cum=False)

        w_t = np.zeros((n_dt + 1, int(n_path)))
        w_t[0, :] = np.log(self.sigma)

        for i in range(1, n_dt + 1):
            w_t[i, :] = (
                w_t[i - 1, :]
                + (
                    self.mr * self.theta * np.exp(-w_t[i - 1, :])
                    - self.mr
                    - self.vov ** 2 / 2
                )
                * self.dt
                + self.vov * np.sqrt(self.dt) * rn_norm[i - 1, :]
            )

        return np.exp(w_t)

    def cond_spot_sigma(self, texp):

        rhoc = np.sqrt(1.0 - self.rho ** 2)
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        var_paths = self.vol_paths(tobs)
        sigma_paths = np.sqrt(var_paths)
        sigma_final = sigma_paths[-1, :]
        int_sigma = scint.simps(sigma_paths, dx=1, axis=0) * texp/n_dt
        int_var = scint.simps(var_paths, dx=1, axis=0) * texp/n_dt
        int_sigma_inv = scint.simps(1/sigma_paths, dx=1, axis=0) * texp/n_dt

        fwd_cond = np.exp(
            self.rho
            * (
                2 * (np.sqrt(sigma_final) - np.sqrt(self.sigma)) / self.vov
                - self.mr * self.theta * int_sigma_inv / self.vov
                + (self.mr / self.vov + self.vov / 4) * int_sigma
                - self.rho * int_var / 2
            )
        )  # scaled by initial value

        sigma_cond = rhoc * np.sqrt(int_var/texp)

        return fwd_cond, sigma_cond
