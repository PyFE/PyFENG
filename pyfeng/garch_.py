import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv


class GarchApproxUncor(sv.SvABC):
    """
    The implementation of Barone-Adesi et al. (2004)'s approximation pricing formula for European
    options under uncorrelated GARCH diffusion model.

    References: Barone-Adesi, G., Rasmussen, H., Ravanelli, C., 2005. An option pricing formula for the GARCH diffusion model. Computational Statistics & Data Analysis, 2nd CSDA Special Issue on Computational Econometrics 49, 287–310. https://doi.org/10.1016/j.csda.2004.05.014

    This method is only used to compare with the method GarchCondMC.
    """


class GarchCondMC(sv.SvABC, sv.CondMcBsmABC):
    """
        Garch model with conditional Monte-Carlo simulation
    """

    def vol_paths(self, tobs, n_path=None):
        """
            sigma_t = np.exp(-mr * tobs) * (sigma0 - theta * mr + vov / np.sqrt(2 * mr) * bm) + theta * mr
            Args:
                tobs: observation time (array)
                mr: coefficient of dt
                theta: the long term average
                mu: rn-derivative

            Returns: volatility path (time, path) including the value at t=0
        """
        dt = self.dt
        n_dt = len(tobs)
        n_path = n_path or self.n_path
        w_path = np.zeros((n_dt+1, n_path))
        w0 = 2 * np.log(self.sigma)
        w_path[0, :] = w0
        var_t = np.zeros((n_dt+1, n_path))
        var_t[0, :] = self.sigma ** 2
        for i in range(n_dt):
            z1 = np.random.standard_normal(n_path)
            w_path[i + 1, :] = w_path[i, :] + (self.mr*self.theta*np.exp(-w_path[i, :]) - self.mr - self.vov**2 / 2) * dt + \
                               self.vov * z1 * np.sqrt(dt)
            var_t[i + 1, :] = var_t[i, :] * np.exp(w_path[i+1, :] - w_path[i, :])
        return var_t

    def cond_fwd_vol(self, texp):
        """
            Returns new forward and volatility conditional on volatility path (e.g., sigma_T, integrated variance)
            The forward and volatility are standardized in the sense that F_0 = 1 and sigma_0 = 1
            Therefore, they should be scaled by the original F_0 and sigma_0 values

            Args:
                mr: coefficient of dt
                texp: time-to-expiry
                w0: initial value of w0
                theta: the long term average
            Returns: (forward, volatility)
        """
        rhoc = np.sqrt(1.0 - self.rho ** 2)
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        var_paths = self.vol_paths(tobs)
        sigma_final = np.sqrt(var_paths[-1, :])

        int_sigma = scint.simps(np.sqrt(var_paths), dx=1, axis=0) / n_dt
        int_sigma_inverse = scint.simps(1/np.sqrt(var_paths), dx=1, axis=0) / n_dt
        int_var = scint.simps(var_paths, dx=1, axis=0) / n_dt

        fwd_cond = np.exp(self.rho * (2/self.vov*(sigma_final - self.sigma) - self.mr*self.theta/self.vov*int_sigma_inverse
                                      + (self.mr/self.vov + self.vov/4)*int_sigma - self.rho * 0.5 * int_var))  # scaled by initial value

        vol_cond = rhoc * np.sqrt(int_var / texp)

        return fwd_cond, vol_cond

    def price(self, strike, spot, texp, cp=1):
        """
            Calculate option price based on BSM
            Args:
                strike: strike price
                spot: spot price
                texp: time to maturity
                mr: coefficient of dt
                theta: the long term average
                w0: initial value of w
                cp: cp=1 if call option else put option

            Returns: price
        """
        price = []
        texp = [texp] if isinstance(texp, (int, float)) else texp
        for t in texp:
            fwd = self.forward(spot, t)
            kk = strike / fwd
            scalar_output = len(kk)
            kk = np.atleast_1d(kk)

            fwd_cond, vol_cond = self.cond_fwd_vol(t)

            base_model = self.base_model(vol_cond)
            price_grid = base_model.price(kk[:, None], fwd_cond, texp=t, cp=cp)

            price.append(fwd * np.mean(price_grid, axis=1))  # in cond_fwd_vol, S_0 = 1

        return price[:, 0] if scalar_output == 1 else price
























