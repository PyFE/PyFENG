import numpy as np
import scipy.integrate as scint
from . import sv


class HestonCondMc(sv.SvABC, sv.CondMcBsmABC):
    """
    Heston model with conditional Monte-Carlo simulation
    """

    def vol_paths(self, tobs):
        dt = np.diff(tobs)
        n_dt = len(dt)

        dB_t = self._bm_incr(tobs, cum=False)  # B_t (0 <= s <= 1)
        vv_path = np.empty([n_dt+1, self.n_path])  # variance series: V0, V1,...,VT
        vv = self.sigma**2
        vv_path[0, :] = vv
        for i in range(n_dt):
            vv = vv + self.mr * (self.sig_inf**2 - vv)*dt[i] + np.sqrt(vv)*self.vov*dB_t[i, :]  # Euler method
            vv = vv + 0.25 * self.vov**2 * (dB_t[i, :]**2 - dt[i])  # Milstein method
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
