import abc
import numpy as np
from . import sv_abc as sv
from . import heston_mc
import scipy.optimize as spop
import scipy.special as spsp


class Sv32McABC(sv.SvABC, sv.CondMcBsmABC, abc.ABC):
    var_process = True
    model_type = "3/2"
    scheme = None

    def _m_heston(self):
        m = heston_mc.HestonMcAndersen2008(1/self.sigma, self.vov, self.rho, self.mr, self.theta)
        return m

    @abc.abstractmethod
    def cond_states(self, var_0, dt):
        """
        Final variance and integrated variance over dt given var_0
        The int_var is normalized by dt

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            (var_final, var_mean)
        """
        return NotImplementedError

    def cond_spot_sigma(self, var_0, texp):
        var_final, var_mean = self.cond_states(var_0, texp)

        spot_cond = (np.log(var_final/var_0) - self.mr * texp *\
            (self.theta - (1 + self.vov**2/2/self.mr)*var_mean))/self.vov\
            - 0.5 * self.rho * var_mean * texp
        np.exp(self.rho * spot_cond, out=spot_cond)
        sigma_cond = np.sqrt((1.0 - self.rho**2) * var_mean / var_0 )  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond


class Sv32McTimeStep(Sv32McABC):
    '''
    CONDITIONAL SIMULATION OF THE 3/2 MODEL

    '''
    scheme = 1  # Milstein

    def set_mc_params(self, n_path=10000, dt=0.05, rn_seed=None, antithetic=True, scheme=1):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step for Euler/Milstein steps
            rn_seed: random number seed
            antithetic: antithetic
            scheme: 0 for Euler, 1 for Milstein, 2 for NCX2, 3 for NCX2 with Poisson, 4 for 2 for Andersen (2008)'s QE scheme

        References:
            - Andersen L (2008) Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11:1–42. https://doi.org/10.21314/JCF.2008.189
        """
        super().set_mc_params(n_path, dt, rn_seed, antithetic)
        self.scheme = scheme

    def var_step_euler(self, var_0, dt, milstein=True):
        """
        Simulate final variance with Euler/Milstein schemes (scheme = 0, 1)

        Args:
            var_0: initial variance
            dt: time step
            milstein: True or False (default)

        Returns:
            final variance (at t=T)
        """
        zz = self.rv_normal(spawn=0)

        # Euler scheme
        var_t = 1.0 + self.mr * (self.theta - var_0) * dt + self.vov * np.sqrt(var_0 * dt) * zz \
        # Extra-term for Milstein scheme
        if milstein:
            var_t += 0.75 * self.vov**2 * var_0 * (zz**2 - 1.0) * dt

        var_t *= var_0
        var_t[var_t < 0] = 0  # variance should be larger than zero

        return var_t

    def cond_states(self, var_0, texp):

        tobs = self.tobs(texp)
        n_dt = len(tobs)
        dt = np.diff(tobs, prepend=0)

        # precalculate the Simpson's rule weight
        weight = np.ones(n_dt + 1)
        weight[1:-1] = 2
        weight /= weight.sum()

        var_t = np.full(self.n_path, var_0)
        var_mean = weight[0] * var_t

        milstein = (self.scheme == 1)
        for i in range(n_dt):
            # Euler (or Milstein) scheme
            var_t = self.var_step_euler(var_t, dt[i], milstein=milstein)
            var_mean += weight[i + 1] * var_t

        return var_t, var_mean  # * texp
