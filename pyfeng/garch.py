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
    The SDE of SV is: dv_t = mr * (theta - v_t) dt + vov * v_t dB_T
    """

    def vol_paths(self, tobs, n_path=None):
        """
        Milstein Schemes:
        w_(t+dt) = w_t + (mr * theta * exp(-w_t) - mr - vov^2 / 2) * dt + vov * Z * sqrt(dt)
        v_t = exp(w_t)
        Args:
            mr: coefficient of dt
            theta: the long term average
            Z : std normal distributed RN
            dt : delta t, time step

        Returns: volatility path (time, path) including the value at t=0
        """
        n_dt = len(tobs)
        n_path = n_path or self.n_path
        rn_norm = self.rng.normal(size=(n_dt, int(n_path)))

        w_t = np.zeros((n_dt + 1, int(n_path)))
        w_t[0, :] = 2 * np.log(self.sigma)

        for i in range(1, n_dt + 1):
            w_t[i, :] = w_t[i - 1, :] + (self.mr * self.theta * np.exp(-w_t[i - 1, :]) - self.mr - self.vov ** 2 / 2) * \
                        self.dt + self.vov * np.sqrt(self.dt) * rn_norm[i - 1, :]

        v_t = np.exp(w_t)

        return v_t

    def cond_fwd_vol(self, texp):
        """
            Returns new forward and volatility conditional on volatility path (e.g., sigma_T, integrated variance)
            The forward and volatility are standardized in the sense that F_0 = 1 and sigma_0 = 1
            Therefore, they should be scaled by the original F_0 and sigma_0 values

            Args:
                theta: the long term average
                mr: coefficient of dt
                texp: time-to-expiry

            Returns: (forward, volatility)
        """
        rhoc = np.sqrt(1.0 - self.rho ** 2)
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        v_paths = self.vol_paths(tobs)
        sigma_paths = np.sqrt(v_paths)
        sigma_final = sigma_paths[-1, :]
        int_sigma = scint.simps(sigma_paths, dx=1, axis=0) / n_dt
        int_var = scint.simps(sigma_paths ** 2, dx=1, axis=0) / n_dt
        int_sigma_inv = scint.simps(1 / sigma_paths, dx=1, axis=0) / n_dt

        fwd_cond = np.exp(
            self.rho * (2 * (sigma_final - self.sigma) / self.vov - self.mr * self.theta * int_sigma_inv / self.vov
                        + (
                                    self.mr / self.vov + self.vov / 4) * int_sigma - self.rho * int_var / 2))  # scaled by initial value

        vol_cond = rhoc * np.sqrt(int_var / texp)

        return fwd_cond, vol_cond

    def price(self, strike, spot, texp, cp=1):
        """
        Calculate option price based on BSM
        Args:
            strike: strike price
            spot: spot price
            texp: time to maturity
            cp: cp=1 if call option else put option

        Returns: price
        """
        price = []
        texp = [texp] if isinstance(texp, (int, float)) else texp
        for t in texp:
            fwd = self.forward(spot, t)
            kk = strike / fwd
            kk = np.atleast_1d(kk)

            fwd_cond, vol_cond = self.cond_fwd_vol(t)

            base_model = self.base_model(vol_cond)
            price_grid = base_model.price(kk[:, None], fwd_cond, texp=t, cp=cp)

            price.append(fwd * np.mean(price_grid, axis=1))  # in cond_fwd_vol, S_0 = 1

        return price
