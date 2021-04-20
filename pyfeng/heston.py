import numpy as np
import scipy.integrate as scint
from . import sv


class HestonCondMc(sv.SvABC, sv.CondMcBsmABC):
    """
    Heston model with conditional Monte-Carlo simulation
    """

    def vol_paths(self, tobs):

        np.random.seed(self.rn_seed)

        dt = np.diff(tobs)
        n_steps = len(dt)
        vov_sqrt_dt = self.vov * np.sqrt(dt)

        # Antithetic
        if self.antithetic:
            zz = self.rng.normal(size=(n_steps, self.n_path//2))
            zz = np.concatenate((zz, -zz), axis=1)
        else:
            zz = self.rng.normal(size=(n_steps, self.n_path))

        vv_path = np.empty([n_steps+1, self.n_path])  # variance series: V0, V1,...,VT
        vv = self.sigma**2
        vv_path[0, :] = vv
        for i in range(n_steps):
            vv = vv + self.mr * (self.sig_inf**2 - vv)*dt[i] + np.sqrt(vv)*vov_sqrt_dt[i]*zz[i, :]  # Euler method
            vv = vv + 0.25 * vov_sqrt_dt[i]**2 * (zz[i, :]**2 - 1)  # Milstein method
            vv[vv < 0] = 0  # variance should be larger than zero
            vv_path[i+1, :] = vv

        # return normalized sigma, e.g., sigma(0) = 1
        return np.sqrt(vv_path)/self.sigma

    def cond_fwd_vol(self, texp):

        tobs = self.tobs(texp)
        n_steps = len(tobs) - 1
        sigma_paths = self.vol_paths(tobs)
        vv0 = self.sigma**2
        vv_ratio = sigma_paths[-1, :]
        int_var_std = scint.simps(sigma_paths**2, dx=1, axis=0) / n_steps

        int_sig_dw = ((vv_ratio - 1)*vv0 - self.mr * texp * (self.sig_inf**2 - int_var_std*vv0)) / self.vov
        fwd_cond = np.exp(self.rho * int_sig_dw - 0.5*self.rho**2 * int_var_std * vv0 * texp)
        vol_cond = np.sqrt((1 - self.rho**2) * int_var_std)

        # return normalized forward and volatility
        return fwd_cond, vol_cond
