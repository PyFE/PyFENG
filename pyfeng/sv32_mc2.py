import abc
import numpy as np
from . import sv_abc as sv
from . import heston_mc
import scipy.optimize as spop
import scipy.special as spsp
import scipy.stats as spst
from scipy.misc import derivative


class Sv32McABC(sv.SvABC, sv.CondMcBsmABC, abc.ABC):
    model_type = "3/2"
    var_process = True
    scheme = None
    _m_heston = None

    def set_num_params(self, n_path=10000, dt=None, rn_seed=None, antithetic=True):
        super().set_num_params(n_path, dt, rn_seed, antithetic)

        mr = self.mr * self.theta
        theta = (self.mr + self.vov ** 2) / mr
        self._m_heston = heston_mc.HestonMcAndersen2008(1/self.sigma, self.vov, self.rho, mr, theta)
        self._m_heston.set_num_params(n_path, dt, rn_seed, antithetic)

    @abc.abstractmethod
    def cond_states_step(self, dt, var_0):
        """
        Final variance and integrated variance over dt given var_0
        The int_var is normalized by dt

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            (var_t, avgvar)
        """
        return NotImplementedError

    @staticmethod
    def iv_complex(nu, zz):
        """
        Modified Bessel function of the first kind with complex argument

        Args:
            nu: index
            zz: value

        Returns:

        """
        p0 = np.power(0.5 * zz, nu) / spsp.gamma(nu + 1)
        iv = p0.copy()
        zzh2 = (zz/2)**2
        for kk in np.arange(1, 64):
            p0 *= zzh2 / (kk * (kk + nu))
            iv += p0
        return iv

    @staticmethod
    def iv_d12(nu, zz):
        """
        The 1st and 2nd derivative of modified Bessel function of the first kind w.r.t. the index nu

        Args:
            nu: index
            zz: value

        Returns:

        References:
            * https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/20/ShowAll.html
        """
        print(nu, zz)
        p0 = np.power(zz/2, nu) / spsp.gamma(nu + 1)
        #psi_1 = np.full_like(zz, spsp.polygamma(1, nu + 1), dtype=float)
        psi_1 = spsp.polygamma(1, nu + 1)
        log_m_psi0 = np.log(zz/2) - spsp.digamma(nu + 1)
        iv1 = log_m_psi0 * p0
        iv2 = (log_m_psi0**2 - psi_1) * p0

        kk_max = max(64, int((np.mean(zz) + 2*np.std(zz)) * 20))
        zzh2 = (zz/2)**2
        #print(f'kk: {kk_max}')
        for kk in np.arange(1, kk_max):
            p0 *= zzh2 / (kk * (kk + nu))
            log_m_psi0 -= 1/(nu + kk)
            psi_1 -= 1/(nu + kk)**2
            iv1 += log_m_psi0 * p0
            iv2 += (log_m_psi0**2 - psi_1) * p0
        return iv1, iv2

    def cond_avgvar_mv(self, dt, var_0, var_t, eta=None):
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

        vov2dt = self.vov**2 * dt
        d1_nu_bb = 4 / vov2dt / nu
        d2_nu_bb = -(4 / vov2dt)**2 / nu**3
        zz = phi / np.sqrt(var_0 * var_t)

        # print(f'phi: {phi}, nu: {nu}, zz: {zz.mean()}')
        if eta is None:
            iv = spsp.iv(nu, zz)
            iv_d1, iv_d2 = self.iv_d12(nu, zz)
            d1 = - (iv_d1 * d1_nu_bb) / iv
            var = (iv_d1 * d2_nu_bb + iv_d2 * d1_nu_bb**2) / iv - d1**2
        else:
            d1 = - spsp.digamma(eta + nu + 1) + np.log(zz/2)
            var = d2_nu_bb * d1 - d1_nu_bb**2 * spsp.polygamma(1, eta + nu + 1)
            d1 *= -d1_nu_bb

        var[var < 0] = 1e-64
        return d1, var

    def cond_spot_sigma(self, texp, var_0):
        tobs = self.tobs(texp)
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        var_t = np.full(self.n_path, var_0)
        avgvar = np.zeros(self.n_path)

        for i in range(n_dt):
            var_t, avgvar_inc = self.cond_states_step(dt[i], var_t)
            avgvar += avgvar_inc * dt[i]

        avgvar /= texp
        spot_cond = (np.log(var_t/var_0) - texp * (self.mr * self.theta - (self.mr + self.vov**2/2)*avgvar)) / self.vov\
            - self.rho * avgvar * texp / 2
        np.exp(self.rho * spot_cond, out=spot_cond)
        sigma_cond = np.sqrt((1.0 - self.rho**2) * avgvar / var_0)  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond

    def return_var_realized(self, texp, cond):
        return None

class Sv32McTimeStep(Sv32McABC):
    """
    CONDITIONAL SIMULATION OF THE 3/2 MODEL

    """
    scheme = 1  # Milstein

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

    def cond_states_step(self, dt, var_0):

        if self.scheme < 2:
            milstein = (self.scheme == 1)
            # Euler (or Milstein) scheme
            var_t = self.var_step_euler(var_0, dt, milstein=milstein)
        elif self.scheme == 2:
            # Euler (or Milstein) scheme
            var_t = self._m_heston.var_step_ncx2(dt, 1 / var_0)
            np.divide(1.0, var_t, out=var_t)
        elif self.scheme == 3:
            # Euler (or Milstein) scheme
            var_t, _ = self._m_heston.var_step_pois_gamma(dt, 1 / var_0)
            np.divide(1.0, var_t, out=var_t)
        else:
            raise ValueError(f'Invalid scheme: {self.scheme}')

        # Trapezoidal rule
        avgvar = (var_0 + var_t)/2

        return var_t, avgvar


class Sv32McBaldeaux2012Exact(Sv32McABC):
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

    def cond_avgvar_laplace(self, bb, dt, var_0, var_t, eta=None):
        vov2dt = self.vov**2 * dt
        phi, _ = self._m_heston.phi_exp(dt)
        nu = self._m_heston.chi_dim()/2 - 1
        nu_bb = np.sqrt(nu**2 + 8*bb/vov2dt)
        zz = phi / np.sqrt(var_0 * var_t)

        if eta is None:
            ret = self.iv_complex(nu_bb, zz) / spsp.iv(nu, zz)
        else:
            nu_diff = 8*bb / self.vov**2 / (nu_bb + nu)
            ret = spsp.gamma(eta + nu + 1) / spsp.gamma(eta + nu_bb + 1) * np.power(zz/2, nu_diff)

        return ret

    def draw_cond_avgvar(self, dt, var_0, var_t):
        def laplace_cond(bb):
            return self.cond_avgvar_laplace(bb, dt, var_0, var_t)

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
        zz = self.rv_normal(spawn=0)
        uu = spst.norm.cdf(zz)

        def root(xx):
            h_xx = h * xx
            rv = h_xx + 2*(phimat * np.sin(h_xx * jj) / jj).sum(axis=0) - uu * np.pi
            return rv

        guess = m1 * np.exp(ln_sig*(zz - ln_sig/2))
        avgvar = spop.newton(root, guess)

        return avgvar

    def cond_states_step(self, dt, var_0):
        """
        Sample variance at maturity and conditional integrated variance

        Args:
            dt: float, time to maturity
        Returns:
            tuple, variance at maturity and conditional integrated variance
        """

        var_t = self._m_heston.var_step_ncx2(dt, 1 / var_0)
        np.divide(1.0, var_t, out=var_t)

        avgvar = self.draw_cond_avgvar(dt, var_0, var_t)

        return var_t, avgvar


class Sv32McChoiKwok2023Ig(Sv32McBaldeaux2012Exact):

    dist = 'ig'

    def draw_from_mv(self, mean, var, dist):
        """
        Draw RNs from distributions with mean and variance matched
        Args:
            mean: mean (1d array)
            var: variance (1d array)
            dist: distribution. 'ig' for IG, 'ga' for Gamma, 'ln' for log-normal

        Returns:
            RNs with size of mean/variance
        """
        idx = (mean > np.finfo(np.float).eps)
        avgvar = np.zeros_like(mean)
        mean = mean[idx]
        var = var[idx]

        if dist.lower() == 'ig':
            # mu and lambda defined in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
            # RNG.wald takes the same parameters
            lam = mean ** 3 / var
            avgvar[idx] = self.rng_spawn[1].wald(mean=mean, scale=lam)
        elif dist.lower() == 'ga':
            scale = var / mean
            shape = mean / scale
            avgvar[idx] = scale * self.rng_spawn[1].standard_gamma(shape=shape)
        elif dist.lower() == 'ln':
            scale = np.sqrt(np.log(1 + var / mean ** 2))
            avgvar[idx] = mean * np.exp(scale * (self.rv_normal(spawn=1) - scale / 2))
        else:
            raise ValueError(f"Incorrect distribution: {dist}.")

        return avgvar

    def cond_avgvar_mv_numeric(self, dt, var_0, var_t):
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
            return np.log(self.cond_avgvar_laplace(-bb, dt, var_0, var_t))

        m1 = derivative(cumgenfunc_cond, 0, n=1, dx=1e-3)
        var = derivative(cumgenfunc_cond, 0, n=2, dx=1e-3)
        return m1, var

    def cond_states_step_invlap(self, var_0, texp):
        """
        Sample variance at maturity and conditional integrated variance

        Args:
            texp: float, time to maturity
        Returns:
            tuple, variance at maturity and conditional integrated variance
        """

        var_t, eta = self._m_heston.var_step_pois_gamma(texp, 1 / var_0)
        # var_t = self._m_heston.var_step_ncx2(1/var_0, dt)
        np.divide(1.0, var_t, out=var_t)
        # print('eta', eta.min(), eta.mean(), eta.max())

        def laplace_cond(bb):
            return self.cond_avgvar_laplace(bb, texp, var_0, var_t, eta)

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

        avgvar = spop.newton(root, m1)
        return var_t, avgvar

    def cond_states_step(self, dt, var_0):
        """
        Final variance and integrated variance over dt given var_0
        The int_var is normalized by dt

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            (var_t, avgvar)
        """

        # var_t, _ = self._m_heston.var_step_pois_gamma(1/var_0, dt)
        var_t = self._m_heston.var_step_ncx2(dt, 1/var_0)
        np.divide(1.0, var_t, out=var_t)
        m1, var = self.cond_avgvar_mv(dt, var_0, var_t, eta=None)
        avgvar = self.draw_from_mv(m1, var, self.dist)

        return var_t, avgvar
