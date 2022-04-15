import abc
import numpy as np
import mpmath as mp
from . import sv_abc as sv
from . import heston_mc
import scipy.optimize as spop
import scipy.special as spsp


class Sv32McABC(sv.SvABC, sv.CondMcBsmABC, abc.ABC):
    var_process = True
    model_type = "3/2"
    scheme = None
    _m_heston = None

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
            scheme: 0 for Euler, 1 for Milstein, 2 for NCX2, 3 for NCX2 with Poisson

        References:
            - Andersen L (2008) Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11:1–42. https://doi.org/10.21314/JCF.2008.189
        """
        super().set_mc_params(n_path, dt, rn_seed, antithetic)

        self.scheme = scheme
        mr = self.mr * self.theta
        theta = (self.mr + self.vov**2)/mr
        self._m_heston = heston_mc.HestonMcAndersen2008(1/self.sigma, self.vov, self.rho, mr, theta)
        self._m_heston.set_mc_params(n_path, dt, rn_seed, antithetic, scheme=scheme)

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

        weight = np.ones(n_dt + 1)
        weight[1:-1] = 2
        weight /= weight.sum()

        var_t = np.full(self.n_path, var_0)
        var_mean = weight[0] * var_t

        if self.scheme < 2:
            milstein = (self.scheme == 1)
            for i in range(n_dt):
                # Euler (or Milstein) scheme
                var_t = self.var_step_euler(var_t, dt[i], milstein=milstein)
                var_mean += weight[i + 1] * var_t
        elif self.scheme == 2:
            for i in range(n_dt):
                # Euler (or Milstein) scheme
                var_t = 1/self._m_heston.var_step_ncx2(1/var_t, dt[i])
                var_mean += weight[i + 1] * var_t
        elif self.scheme == 3:
            for i in range(n_dt):
                # Euler (or Milstein) scheme
                var_t, _ = self._m_heston.var_step_ncx2_eta(1/var_t, dt[i])
                var_t = 1/var_t
                var_mean += weight[i + 1] * var_t
        else:
            raise ValueError(f'Invalid scheme: {self.scheme}')

        return var_t, var_mean  # * texp


class Sv32McExactBaldeaux2012(Sv32McABC):
    """
    EXACT SIMULATION OF THE 3/2 MODEL

    Parameters:
        sigma: float, initial volatility
        vov, mr, rho, theta: float, parameters of the 3/2 model, similar to Heston model where
            vov is the volatility of the variance process
            mr is the rate at which the variance reverts toward its long-term mean
            rho is correlation between asset price and volatility
            theta is the mean long-term variance
        intr, divr: float, interest rate and dividend yield
        is_fwd: Bool, true if asset price is forward
    """

    error = 1e-5

    def set_mc_params(self, n_path=10000, dt=0.05, rn_seed=None, antithetic=True, scheme=1):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step for Euler/Milstein steps
            rn_seed: random number seed
            antithetic: antithetic
            scheme: 0 for Euler, 1 for Milstein, 2 for NCX2, 3 for NCX2 with Poisson

        References:
            - Andersen L (2008) Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11:1–42. https://doi.org/10.21314/JCF.2008.189
        """
        super().set_mc_params(n_path, dt, rn_seed, antithetic)

        self.scheme = scheme
        mr = self.mr * self.theta
        theta = (self.mr + self.vov**2)/mr
        self._m_heston = heston_mc.HestonMcAndersen2008(1/self.sigma, self.vov, self.rho, mr, theta)
        self._m_heston.set_mc_params(n_path, dt, rn_seed, antithetic, scheme=scheme)


    def cond_states(self, var_0, texp):
        '''
        Sample variance at maturity and conditional integrated variance

        Args:
            texp: float, time to maturity
        Returns:
            tuple, variance at maturity and conditional integrated variance
        '''

        x_0 = 1 / var_0
        mr_new = self.mr * self.theta
        x_t = self._m_heston.var_step_ncx2(x_0, texp)

        nu = self._m_heston.chi_dim() / 2 - 1
        phi, _ = self._m_heston.phi_exp(texp)
        zz = np.sqrt(x_0 * x_t) * phi
        base_val = spsp.iv(nu, zz)

        # Determine h, the grid size associated with the trapezoidal rule and N, the number of terms in the summation
        Laplacefun_plus = spsp.iv(np.sqrt(nu**2 + 8e-5 / self.vov**2), zz) / base_val
        Laplacefun_minus = spsp.iv(np.sqrt(nu**2 - 8e-5 / self.vov**2), zz) / base_val
        moment1 = (Laplacefun_minus - Laplacefun_plus) / 2e-5
        moment2 = (Laplacefun_plus + Laplacefun_minus - 2) / 1e-10
        u_error = moment1 + 5 * np.sqrt(np.fmax(moment2 - moment1**2, 0))
        h = np.pi / u_error

        mp.dps = 8
        besseli_ufun = np.frompyfunc(mp.besseli, 2, 1)
        #N = np.ones(self.n_path)
        #for i in range(self.n_path):
        #    Nfun = lambda _N: mp.fabs(
        #        besseli_ufun(np.sqrt(nu**2 - 8j * h[i] * _N / self.vov**2), z[i]) / base_val[i]) \
        #                      - np.pi * self.error * _N / 2
        #    N[i] = int(spop.brentq(Nfun, 0, 1000)) + 1
        #N = N.max()
        #print(N)
        N = 60

        # Store the value of characteristic function for each term in the summation when approximating the CDF
        order = np.sqrt(nu**2 - 8j * np.arange(1, N + 1) * h[:, None] / self.vov**2)
        phimat = besseli_ufun(order, zz[:, None]) / base_val[:, None]
        phimat = phimat.astype('complex').real

        # Sample the conditional integrated variance by inverse transform sampling
        integ_var = np.ones(self.n_path)
        uniformRN = self.rv_uniform()
        for i in range(self.n_path):
            conCDF = lambda _x: h[i] * _x / np.pi + 2 / np.pi * (
                        np.sin(h[i] * _x * np.arange(1, N + 1)) * phimat[i, :] / np.arange(1, N + 1)).sum() - uniformRN[
                                    i]
            integ_var[i] = spop.brentq(conCDF, 0, 2 * u_error[i])

        return 1 / x_t, integ_var/texp
