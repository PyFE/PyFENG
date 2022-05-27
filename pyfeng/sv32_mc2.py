import abc
import numpy as np
from . import sv_abc as sv
from . import heston_mc
import scipy.optimize as spop
import scipy.special as spsp
import scipy.stats as spst
from scipy.misc import derivative


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
            (var_final, var_avg)
        """
        return NotImplementedError

    @staticmethod
    def ivc(nu, zz):
        p0 = np.power(0.5 * zz, nu) / spsp.gamma(nu + 1)
        iv = p0.copy()
        for kk in np.arange(1, 64):
            p0 *= zz**2 / (4 * kk * (kk + nu))
            iv += p0
        return iv

    def cond_spot_sigma(self, var_0, texp):
        var_t, var_avg = self.cond_states(var_0, texp)

        spot_cond = (np.log(var_t/var_0) - texp * (self.mr * self.theta - (self.mr + self.vov**2/2)*var_avg)) / self.vov\
            - self.rho * var_avg * texp / 2
        np.exp(self.rho * spot_cond, out=spot_cond)
        sigma_cond = np.sqrt((1.0 - self.rho**2) * var_avg / var_0)  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond


class Sv32McTimeStep(Sv32McABC):
    """
    CONDITIONAL SIMULATION OF THE 3/2 MODEL

    """
    scheme = 1  # Milstein

    def set_num_params(self, n_path=10000, dt=0.05, rn_seed=None, antithetic=True, scheme=1):
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
        super().set_num_params(n_path, dt, rn_seed, antithetic)

        self.scheme = scheme
        mr = self.mr * self.theta
        theta = (self.mr + self.vov**2)/mr
        self._m_heston = heston_mc.HestonMcAndersen2008(1/self.sigma, self.vov, self.rho, mr, theta)
        self._m_heston.set_num_params(n_path, dt, rn_seed, antithetic, scheme=scheme)

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
        var_t = 1.0 + self.mr * (self.theta - var_0) * dt + self.vov * np.sqrt(var_0 * dt) * zz
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
        var_avg = weight[0] * var_t

        if self.scheme < 2:
            milstein = (self.scheme == 1)
            for i in range(n_dt):
                # Euler (or Milstein) scheme
                var_t = self.var_step_euler(var_t, dt[i], milstein=milstein)
                var_avg += weight[i + 1] * var_t
        elif self.scheme == 2:
            for i in range(n_dt):
                # Euler (or Milstein) scheme
                var_t = 1/self._m_heston.var_step_ncx2(1/var_t, dt[i])
                var_avg += weight[i + 1] * var_t
        elif self.scheme == 3:
            for i in range(n_dt):
                # Euler (or Milstein) scheme
                var_t, _ = self._m_heston.var_step_ncx2_eta(1/var_t, dt[i])
                var_t = 1/var_t
                var_avg += weight[i + 1] * var_t
        else:
            raise ValueError(f'Invalid scheme: {self.scheme}')

        return var_t, var_avg  # * texp


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

    def set_num_params(self, n_path=10000, dt=None, rn_seed=None, antithetic=True, scheme=1):
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
        super().set_num_params(n_path, dt, rn_seed, antithetic)

        self.scheme = scheme
        mr = self.mr * self.theta
        theta = (self.mr + self.vov**2)/mr
        self._m_heston = heston_mc.HestonMcAndersen2008(1/self.sigma, self.vov, self.rho, mr, theta)
        self._m_heston.set_num_params(n_path, dt, rn_seed, antithetic, scheme=scheme)

    def laplace(self, bb, var_0, var_t, dt):
        phi, _ = self._m_heston.phi_exp(dt)
        nu = self._m_heston.chi_dim()/2 - 1
        nu_bb = np.sqrt(nu**2 + 8*bb/self.vov**2)
        zz = phi / np.sqrt(var_0 * var_t)
        ret = self.ivc(nu_bb, zz) / spsp.iv(nu, zz)
        return ret

    def cond_states(self, var_0, texp):
        """
        Sample variance at maturity and conditional integrated variance

        Args:
            texp: float, time to maturity
        Returns:
            tuple, variance at maturity and conditional integrated variance
        """

        x_t, _ = self._m_heston.var_step_ncx2_eta(1/var_0, texp)

        def laplace_cond(bb):
            return self.laplace(bb, var_0, 1/x_t, texp)

        eps = 1e-5
        val_up = laplace_cond(eps)
        val_dn = laplace_cond(-eps)
        m1 = (val_dn - val_up) / (2*eps)
        var = (val_dn + val_up - 2.0)/eps**2 - m1**2
        ln_sig = np.sqrt(np.log(1+var/m1**2))

        u_error = m1 + 5 * np.sqrt(np.fmax(var, 0))
        h = np.pi / u_error

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
        jj = np.arange(1, N + 1)[:, None]
        phimat = laplace_cond(-1j * jj * h).real

        # Sample the conditional integrated variance by inverse transform sampling
        zz = self.rv_normal()
        uu = spst.norm.cdf(zz)

        def root(xx):
            h_xx = h * xx
            rv = h_xx + 2*(phimat * np.sin(h_xx * jj) / jj).sum(axis=0) - uu * np.pi
            return rv

        guess = m1 * np.exp(ln_sig*(zz - ln_sig/2))
        int_var = spop.newton(root, guess)
        return 1/x_t, int_var/texp


class Sv32McExactChoiKwok2023(Sv32McExactBaldeaux2012):

    dist = 'ig'

    def laplace(self, bb, var_0, var_t, dt, eta=None):
        phi, _ = self._m_heston.phi_exp(dt)
        nu = self._m_heston.chi_dim()/2 - 1
        nu_bb = np.sqrt(nu**2 + 8*bb/self.vov**2)
        zz = phi / np.sqrt(var_0 * var_t)

        if eta is None:
            ret = self.ivc(nu_bb, zz) / spsp.iv(nu, zz)
        else:
            nu_diff = 8*bb / self.vov**2 / (nu_bb + nu)
            ret = spsp.gamma(eta + nu + 1) / spsp.gamma(eta + nu_bb + 1) * np.power(zz/2, nu_diff)

        return ret

    def cond_avgvar_mv(self, var_0, var_t, dt, eta=None):
        """
        Mean and variance of the integrated variance conditional on initial var, final var, and eta

        Args:
            var_0: initial variance
            var_t: final variance
            eta: Poisson RV
            dt: time step

        Returns:
            (integarted variance / dt)
        """
        phi, _ = self._m_heston.phi_exp(dt)
        nu = self._m_heston.chi_dim()/2 - 1

        d1_nu_bb = 4 / self.vov**2 / nu
        d2_nu_bb = -(4 / self.vov**2)**2 / nu**3

        zz = phi / np.sqrt(var_0 * var_t)
        d1 = - spsp.digamma(eta + nu + 1) + np.log(zz/2)
        var = d2_nu_bb * d1 - d1_nu_bb**2 * spsp.polygamma(1, eta + nu + 1)
        d1 *= -d1_nu_bb
        return d1, var

    def cond_avgvar_mv_numeric(self, var_0, var_t, dt):
        """
        Mean and variance of the average variance conditional on initial var, final var.
        It is computed from the numerical derivatives of the conditional Laplace transform.

        Args:
            var_0: initial variance
            var_t: final variance
            dt: time step

        Returns:
            mean, variance

        See Also:
            cond_avgvar_mv
        """
        # conditional Cumulant Generating Fuction
        def cumgenfunc_cond(bb):
            return np.log(self.laplace(-bb, var_0, var_t, dt))

        m1 = derivative(cumgenfunc_cond, 0, n=1, dx=1e-4)
        var = derivative(cumgenfunc_cond, 0, n=2, dx=1e-4)
        return m1/dt, var/dt**2

    def cond_states_invlap(self, var_0, texp):
        """
        Sample variance at maturity and conditional integrated variance

        Args:
            texp: float, time to maturity
        Returns:
            tuple, variance at maturity and conditional integrated variance
        """

        x_t, eta = self._m_heston.var_step_ncx2_eta(1/var_0, texp)
        # print('eta', eta.min(), eta.mean(), eta.max())

        def laplace_cond(bb):
            return self.laplace(bb, var_0, 1/x_t, texp, eta)

        eps = 1e-5
        val_up = laplace_cond(eps)
        val_dn = laplace_cond(-eps)
        m1 = (val_dn - val_up) / (2*eps)
        var = (val_dn + val_up - 2.0)/eps**2 - m1**2
        # print('m1', np.amin(m1), np.amax(m1))
        # print('var', np.amin(var), np.amax(var), (var<0).mean())
        std = np.sqrt(np.fmax(var, 0))
        u_error = np.fmax(m1, 1e-6) + 5 * std
        h = np.pi / u_error
        # print('h', (h<0).sum())
        N = 60

        # Store the value of characteristic function for each term in the summation when approximating the CDF
        jj = np.arange(1, N + 1)[:, None]
        phimat = laplace_cond(-1j * jj * h).real

        # Sample the conditional integrated variance by inverse transform sampling
        zz = self.rv_normal()
        uu = spst.norm.cdf(zz)

        def root(xx):
            h_xx = h * xx
            rv = h_xx + 2*(phimat * np.sin(h_xx * jj) / jj).sum(axis=0) - uu * np.pi
            return rv

        int_var = spop.newton(root, m1)
        return 1 / x_t, int_var/texp

    def cond_states(self, inv_var_0, texp):

        tobs = self.tobs(texp)
        n_dt = len(tobs)
        dt = np.diff(tobs, prepend=0)

        inv_var_0 = np.full(self.n_path, inv_var_0)
        var_avg = np.zeros_like(inv_var_0)

        for i in range(n_dt):

            inv_var_t, _ = self._m_heston.var_step_ncx2_eta(inv_var_0, dt[i])
            m1, var = self.cond_avgvar_mv_numeric(1/inv_var_0, 1/inv_var_t, dt[i])
            inv_var_0 = inv_var_t

            if self.dist.lower() == 'ig':
                # mu and lambda defined in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
                # RNG.wald takes the same parameters
                lam = m1**3 / var
                var_avg += self.rng_spawn[1].wald(mean=m1, scale=lam)
            elif self.dist.lower() == 'ga':
                scale = var / m1
                shape = m1 / scale
                var_avg += scale * self.rng_spawn[1].standard_gamma(shape=shape)
            elif self.dist.lower() == 'ln':
                scale = np.sqrt(np.log(1 + var/m1**2))
                var_avg += m1 * np.exp(scale * (self.rv_normal(spawn=1) - scale/2))
            else:
                raise ValueError(f"Incorrect distribution: {self.dist}.")

        var_avg /= n_dt

        return 1/inv_var_t, var_avg
