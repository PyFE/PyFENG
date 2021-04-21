import numpy as np
import scipy.integrate as scint
from . import sabr
from . import sv
from . import cev


class SabrCondMc(sabr.SabrABC, sv.CondMcBsmABC):
    """
    Conditional MC for SABR model (beta=0,1 or rho=0) with conditional Monte-Carlo simulation

    """

    def vol_paths(self, tobs, mu=0):
        """
        exp(vov_std B_s - 0.5*vov_std^2 * s)  where s = 0, ..., 1, vov_std = vov*sqrt(T)

        Args:
            tobs: observation time (array)
            mu: rn-derivative

        Returns: volatility path (time, path)
        """
        assert tobs[0] == 0

        texp = tobs[-1]
        tobs = tobs/texp  # normalized time: s
        vov_std = self.vov * np.sqrt(texp)

        log_sig_s = self._bm_incr(tobs, cum=True)  # B_s (0 <= s <= 1)
        log_rn_deriv = 0.0 if mu == 0 else -mu*(log_sig_s[-1, :] + 0.5*mu)

        log_sig_s = np.insert(log_sig_s, 0, np.zeros(log_sig_s.shape[1]), axis=0)
        log_sig_s = vov_std*(log_sig_s + (mu - 0.5*vov_std) * tobs[:, None])
        return np.exp(log_sig_s), log_rn_deriv

    def cond_fwd_vol(self, texp, mu=0):
        rhoc = np.sqrt(1.0 - self.rho**2)
        rho_sigma = self.rho*self.sigma

        tobs = self.tobs(texp)
        n_steps = len(tobs)-1
        sigma_paths, log_rn_deriv = self.vol_paths(tobs, mu=mu)
        sigma_final = sigma_paths[-1, :]
        int_var = scint.simps(sigma_paths**2, dx=1, axis=0) / n_steps
        vol_cond = rhoc*np.sqrt(int_var)
        if np.isclose(self.beta, 0):
            fwd_cond = rho_sigma/self.vov*(sigma_final - 1)
        else:
            fwd_cond = np.exp(rho_sigma*(1.0/self.vov*(sigma_final - 1) - 0.5*rho_sigma*int_var*texp))

        return fwd_cond, vol_cond, log_rn_deriv

    def price(self, strike, spot, texp, cp=1):
        fwd = self.forward(spot, texp)
        fwd_cond, vol_cond, log_rn_deriv = self.cond_fwd_vol(texp)
        if np.isclose(self.beta, 0):
            base_model = self._m_base(self.sigma*vol_cond, is_fwd=True)
            price_grid = base_model.price(strike[:, None], fwd + fwd_cond, texp, cp=cp)
            price = np.mean(price_grid*np.exp(log_rn_deriv), axis=1)
        else:
            alpha = self.sigma / np.power(spot, 1.0 - self.beta)
            kk = strike / fwd

            base_model = self._m_base(alpha*vol_cond, is_fwd=True)
            price_grid = base_model.price(kk[:, None], fwd_cond, texp, cp=cp)
            price = fwd * np.mean(price_grid*np.exp(log_rn_deriv), axis=1)

        return price

    def mass_zero(self, spot, texp, log=False, mu=0):
        assert 0 < self.beta < 1
        assert self.rho == 0

        eta = self.vov*np.power(spot, 1.0-self.beta)/(self.sigma*(1.0-self.beta))
        vov_std = self.vov * np.sqrt(texp)

        if mu is None:
            mu = 0.5*(vov_std + np.log(1+eta**2)/vov_std)
            #print(f'mu = {mu}')

        fwd_cond, vol_cond, log_rn_deriv = self.cond_fwd_vol(texp, mu=mu)
        base_model = cev.Cev(sigma=self.sigma*vol_cond, beta=self.beta)
        if log:
            log_mass_grid = base_model.mass_zero(spot, texp, log=True) + log_rn_deriv
            log_mass_max = np.amax(log_mass_grid)
            log_mass_grid -= log_mass_max
            log_mass = log_mass_max + np.log(np.mean(np.exp(log_mass_grid)))
            return log_mass
        else:
            mass_grid = base_model.mass_zero(spot, texp, log=False)*np.exp(log_rn_deriv)
            mass = np.mean(mass_grid)
            return mass
