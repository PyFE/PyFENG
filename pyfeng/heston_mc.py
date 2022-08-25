import abc
import numpy as np
import scipy.stats as spst
import scipy.interpolate as spinterp
from scipy import special as spsp
from scipy.misc import derivative
import functools
from . import sv_abc as sv
from . import heston


class HestonMcABC(heston.HestonABC, sv.CondMcBsmABC, abc.ABC):
    var_process = True
    scheme = None

    def chi_dim(self):
        """
        Noncentral Chi-square (NCX) distribution's degree of freedom

        Returns:
            degree of freedom (scalar)
        """
        chi_dim = 4 * self.theta * self.mr / self.vov**2
        return chi_dim

    def chi_lambda(self, dt):
        """
        Noncentral Chi-square (NCX) distribution's noncentrality parameter

        Returns:
            noncentrality parameter (scalar)
        """
        chi_lambda = 4 * self.sigma * self.mr / self.vov**2
        chi_lambda /= np.exp(self.mr*dt) - 1
        return chi_lambda

    def phi_exp(self, texp):
        exp = np.exp(-self.mr*texp/2)
        phi = 4*self.mr / self.vov**2 / (1/exp - exp)
        return phi, exp

    def var_step_euler(self, var_0, dt, milstein=False):
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

        # Euler (or Milstein) scheme
        var_t = var_0 + self.mr * (self.theta - var_0) * dt + np.sqrt(var_0) * self.vov * zz
        # Extra-term for Milstein scheme
        if milstein:
            var_t += 0.25 * self.vov**2 * (zz**2 - dt)

        var_t[var_t < 0] = 0  # variance should be larger than zero
        return var_t

    def var_step_ncx2(self, var_0, dt):
        """
        Draw final variance from NCX2 distribution (scheme = 0)

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            final variance (at t=T)
        """
        chi_df = self.chi_dim()
        phi, exp = self.phi_exp(dt)
        chi_nonc = var_0 * exp * phi
        var_t = (exp / phi) * self.rng_spawn[0].noncentral_chisquare(df=chi_df, nonc=chi_nonc, size=self.n_path)
        return var_t

    def var_step_ncx2_eta(self, var_0, dt):
        """
        Draw final variance from NCX2 distribution with Poisson (scheme = 1)

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            final variance and eta (at t=T)
        """
        chi_df = self.chi_dim()
        phi, exp = self.phi_exp(dt)
        chi_nonc = var_0 * exp * phi
        pois = self.rng_spawn[0].poisson(chi_nonc / 2, size=self.n_path)
        var_t = (exp / phi) * 2 * self.rng_spawn[0].standard_gamma(shape=chi_df / 2 + pois, size=self.n_path)
        return var_t, pois

    @abc.abstractmethod
    def cond_states_step(self, var_0, dt):
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

    def cond_states(self, vol_0, texp):

        tobs = self.tobs(texp)
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        var_t = np.full(self.n_path, vol_0)
        var_avg = np.zeros(self.n_path)

        for i in range(n_dt):
            var_t, d_v = self.cond_states_step(var_t, dt[i])
            var_avg += d_v * dt[i]

        var_avg /= texp

        return var_t, var_avg

    def cond_spot_sigma(self, var_0, texp):
        var_final, var_avg = self.cond_states(var_0, texp)

        avgvar_m, avgvar_v = self.avgvar_mv(texp, var_0)
        self.result = {**self.result,
                       'avgvar mean': avgvar_m,
                       'avgvar mean error': var_avg.mean()/avgvar_m - 1,
                       'avgvar var': avgvar_v,
                       'avgvar var error': np.square(var_avg - avgvar_m).mean()/avgvar_v - 1
                       }

        spot_cond = ((var_final - var_0) + self.mr * texp * (var_avg - self.theta)) / self.vov \
             - 0.5 * self.rho * var_avg * texp
        np.exp(self.rho * spot_cond, out=spot_cond)
        sigma_cond = np.sqrt((1.0 - self.rho**2) / var_0 * var_avg)  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond

    def log_return(self, var_0, var_t, var_avg, dt):
        """
        Samples log return, log(S_t/S_0)

        Args:
            var_0: initial variance
            var_t: final variance
            var_avg: average variance
            dt: time step

        Returns:
            log return
        """
        mean_ln = self.rho/self.vov * ((var_t - var_0) + self.mr * dt * (var_avg - self.theta))  \
            + (self.intr - 0.5 * var_avg) * dt
        sigma_ln = np.sqrt((1.0 - self.rho**2) * var_avg * dt)
        zn = self.rv_normal(spawn=4)
        return mean_ln + sigma_ln * zn

    def avgvar_realized(self, texp):
        """

        Args:
            texp:

        Returns:

        """
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        dt = np.diff(tobs, prepend=0)

        var = np.zeros(self.n_path)
        var_0 = np.full(self.n_path, self.sigma)

        for i in range(n_dt):
            var_t, var_avg = self.cond_states_step(var_0, dt[i])
            r_ln = self.log_return(var_0, var_t, var_avg, dt[i])
            var_0 = var_t
            var += r_ln ** 2

        return var

    def gamma_lambda(self, dt, kk=0):
        """
        gamma_n and lambda_n below Eq. (2.8) in Glasserman & Kim (2011).
        gamma_n is the original value * dt to make x1, x2, and x3 the average variance.

        Args:
            dt: time step
            kk: number of terms

        Returns:
            gamma_n, lambda_n
        """

        mrt2 = (self.mr * dt)**2
        vov2dt = self.vov**2 * dt

        n_2pi_2 = (np.arange(1, kk + 1) * 2 * np.pi)**2
        gamma_n = (mrt2 + n_2pi_2) / (2 * vov2dt)  # / dt
        lambda_n = 4 * n_2pi_2 / vov2dt / (mrt2 + n_2pi_2)

        return gamma_n, lambda_n

    def x1star_avgvar_mv(self, dt, kk=0):
        """
        Mean and variance of the truncated terms of (X1^*/dt) in Glasserman & Kim (2011).
        (v_0 + v_t) need to be multiplied to mean and variance afterwards.

        Args:
            dt: time step
            kk: number of gamma expansion terms

        References:
            - p 281-282 in Glasserman & Kim (2011)
            - Proposition 3.1 in Tse & Wan (2013)

        Returns:
            mean, variance
        """

        mrt_h = self.mr * dt / 2
        vov2dt = self.vov**2 * dt
        csch = 1 / np.sinh(mrt_h)
        coth = np.cosh(mrt_h) * csch

        mean = (coth/mrt_h - csch**2) / 2
        var = vov2dt * (coth / mrt_h**3 + csch**2 / mrt_h**2 - 2 * coth*csch**2 / mrt_h) / 8

        if kk > 0:
            gamma_n, lambda_n = self.gamma_lambda(dt, kk=kk)
            mean -= np.sum(lambda_n/gamma_n)
            var -= 2*np.sum(lambda_n/gamma_n**2)

        return mean, var

    def x2star_avgvar_mv(self, dt, kk=0):
        """
        Mean and variance of the truncated terms of X2/dt (with shape=1 or delta=2) in Glasserman & Kim (2011)

            X2 = sum_{n=1}^kk standard_gamma(1) / gamma_n

        Args:
            dt: time step
            kk: number of gamma expansion terms

        References:
            - p 284 in Glasserman & Kim (2011)
            - Proposition 3.1 in Tse & Wan (2013)

        Returns:
            mean, variance
        """

        mrt_h = self.mr * dt / 2
        vov2dt = self.vov**2 * dt

        csch = 1 / np.sinh(mrt_h)
        coth = np.cosh(mrt_h) * csch

        mean = vov2dt * (mrt_h * coth - 1) / (4 * mrt_h**2)
        var = vov2dt**2 * (mrt_h * coth + mrt_h**2 * csch**2 - 2) / (16 * mrt_h**4)

        if kk > 0:
            gamma_n, _ = self.gamma_lambda(dt, kk)
            mean -= np.sum(1/gamma_n)
            var -= np.sum(1/gamma_n**2)

        return mean, var

    def eta_mv(self, var_0, var_t, dt):
        """
        The mean and variance of eta RV.

        Args:
            var_0: initial variance
            var_t: final variance
            dt: time step

        Returns:
            eta (n_path, 1)

        References:
            Proposition 3.1 in Tse & Wan (2013)
        """
        phi, exp = self.phi_exp(dt)
        zz = np.sqrt(var_0 * var_t) * phi

        iv_index = 0.5 * self.chi_dim() - 1
        iv0 = spsp.iv(iv_index, zz)
        iv1 = spsp.iv(iv_index + 1, zz)
        iv2 = spsp.iv(iv_index + 2, zz)

        mean = (zz/2) * (iv1/iv0)
        var = (zz/2)**2 * (iv2/iv0) + mean - mean**2

        return mean, var

    def cond_avgvar_mv(self, var_0, var_t, dt, eta=None, kk=0):
        """
        Mean and variance of the average variance conditional on initial var, final var, and eta

        Args:
            var_0: initial variance
            var_t: final variance
            eta: Poisson RV
            dt: time step
            kk: number of gamma expansion terms

        Returns:
            mean, variance

        References:
            Proposition 3.1 in Tse & Wan (2013)
        """

        # x = np.arange(1, 10) * 0.02
        # y1 = 1 / x / np.tanh(x) - 1 / np.sinh(x)**2
        # y2 = 2 / 3 - (4 / 45) * x**2 + (4 / 315) * x**4 - (8 / 4725) * x**6
        # y2 - y1
        # y1 = (x / np.tanh(x) - 1) / x**2
        # y2 = 1 / 3 - (1 / 45) * x**2 + (2 / 945) * x**4
        # y2 - y1

        if eta is None:
            eta_mean, eta_var = self.eta_mv(var_0, var_t, dt)
        else:
            eta_mean, eta_var = eta, 0.0

        x1_mean, x1_var = self.x1star_avgvar_mv(dt, kk=kk)
        x1_mean *= (var_0 + var_t)
        x1_var *= (var_0 + var_t)

        x2_mean, x2_var = self.x2star_avgvar_mv(dt, kk=kk)
        x23_mean = (2*eta_mean + self.chi_dim()/2) * x2_mean
        x23_var = (2*eta_mean + self.chi_dim()/2) * x2_var
        x23_var += eta_var * (2*x2_mean)**2

        return x1_mean + x23_mean, x1_var + x23_var


class HestonMcGlassermanKim2011(HestonMcABC):
    """
    Exact simulation using the gamma series based on Glasserman & Kim (2011)

    References:
        - Glasserman P, Kim K-K (2011) Gamma expansion of the Heston stochastic volatility model. Finance Stoch 15:267–296. https://doi.org/10.1007/s00780-009-0115-y
    """

    kk = 1  # K for series truncation.
    tabulate_x2_z = False

    def set_num_params(self, n_path=10000, dt=None, rn_seed=None, antithetic=True, kk=1):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step
            rn_seed: random number seed
            kk: truncation index

        """
        super().set_num_params(n_path, dt, rn_seed, antithetic)
        self.kk = kk

    def cond_avgvar_mgf(self, aa, var_0, var_t, dt):
        """
        MGF of the average variance conditional on the initial and final variance.

        Args:
            aa: dummy variable in the transformation
            var_0: initial variance
            var_t: final variance
            dt: time step

        Returns:
            Conditional MGF at dummy variable aa
        """

        vov2dt = self.vov**2 * dt
        mrt = self.mr * dt
        iv_index = 0.5 * self.chi_dim() - 1

        # Note that dt term is additionally multiplied to aa
        # to make it Laplace transform of average variance, not integrated variance
        gamma = np.sqrt(mrt**2 - 2 * vov2dt * aa)

        var_mean = np.sqrt(var_0 * var_t)
        phi_mr, _ = self.phi_exp(dt)
        cosh_mr = np.cosh(mrt/2)

        exph_gamma = np.exp(-gamma/2)
        phi_gamma = 4 * gamma / vov2dt * exph_gamma / (1 - exph_gamma**2)
        cosh_gamma = (exph_gamma + 1/exph_gamma)/2

        part1 = phi_gamma / phi_mr
        part2 = np.exp((var_0 + var_t) * (cosh_mr * phi_mr - cosh_gamma * phi_gamma) / 2)
        part3 = spsp.iv(iv_index, var_mean * phi_gamma) / spsp.iv(iv_index, var_mean * phi_mr)

        mgf = part1 * part2 * part3

        if np.iscomplexobj(aa):
            # handle branch cut
            tmp = gamma / (1 - exph_gamma**2)
            mgf *= np.exp(iv_index * (np.log(tmp) - gamma/2)) / np.power(tmp * exph_gamma, iv_index)

        return mgf

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
        def cumgenfunc_cond(aa):
            return np.log(self.cond_avgvar_mgf(aa, var_0, var_t, dt))

        m1 = derivative(cumgenfunc_cond, 0, n=1, dx=1e-5)
        var = derivative(cumgenfunc_cond, 0, n=2, dx=1e-5)
        return m1, var

    def x1star_avgvar_mv_asymp(self, dt, kk=0):
        """
        Asymptotic mean and variance of the truncated terms of X1/dt.
        (v_0 + v_t) need to be multiplied to mean and variance afterwards.
        This is NOT used for pricing, but for verification purpose.

        Args:
            dt: time step
            kk: number of gamma expansion terms

        References:
            - Lemma 3.1 in Glasserman & Kim (2011)

        Returns:
            mean, variance
        """

        vov2dt = self.vov**2 * dt
        # remainder (truncated) terms
        trunc_mean = 2 / (np.pi**2 * kk)
        trunc_var = 2 * vov2dt / (3 * np.pi**4 * kk**3)

        return trunc_mean, trunc_var

    def x2star_avgvar_mv_asymp(self, dt, kk=0):
        """
        Asymptotic mean and variance of the truncated terms of X2/dt (with shape=1 or delta=2).
        This is NOT used for pricing, but for verification purpose.

        Args:
            dt: time step
            kk: number of gamma expansion terms

        References:
            - Lemma 3.1 in Glasserman & Kim (2011)

        Returns:
            mean, variance
        """

        vov2dt = self.vov**2 * dt
        trunc_mean = vov2dt / (2 * np.pi**2 * kk)
        trunc_var = vov2dt**2 / (12 * np.pi**4 * kk**3)

        return trunc_mean, trunc_var

    def x2_avgvar_mgf(self, aa, dt, shape):
        """
        MGF of X2/dt (or Z/dt)

        Args:
            aa: dummy variable in the transformation
            shape: gamma shape parameter. delta/2 for X2. 2 for Z.
            dt: time step

        Returns:
            Conditional MGF at dummy variable aa

        References:
            Lemma 2.4 in Glasserman & Kim (2011)
        """

        vov2dt = self.vov**2 * dt
        mrt = self.mr * dt

        # Note that dt term is additionally multiplied to aa
        # to make it Laplace transform of average variance, not integrated variance
        gamma = np.sqrt(mrt**2 - 2 * vov2dt * aa)
        phi_mr, _ = self.phi_exp(dt)

        ### mgf without considering branchcut
        # exph_gamma = np.exp(-gamma/2)
        # phi_gamma = 4 * gamma / vov2dt * exph_gamma / (1 - exph_gamma**2)
        # mgf = np.power(phi_gamma / phi_mr, shape)

        tmp = (4 * gamma / vov2dt) / (1 - np.exp(-gamma))
        mgf = np.exp(shape*(np.log(tmp/phi_mr) - gamma/2))

        return mgf

    def x2star_avgvar_mv_numeric(self, dt):
        """
        Mean and variance of X2/dt (with shape=1 or delta=2) numerically computed from
        the MGF (Laplace transform) of X2/dt.

        Args:
            dt: time step

        Returns:
            mean, variance
        """
        # conditional Cumulant Generating Fuction
        def cumgenfunc_cond(aa):
            return np.log(self.x2_avgvar_mgf(aa, dt, 1))

        m1 = derivative(cumgenfunc_cond, 0, n=1, dx=1e-4)
        var = derivative(cumgenfunc_cond, 0, n=2, dx=1e-4)
        return m1, var

    def draw_x1(self, var_0, var_t, dt):
        """
        Samples of x1/dt using truncated Gamma expansion in Glasserman & Kim (2011)

        Args:
            var_0: initial variance
            var_t: final variance
            dt: time step

        Returns:
            x1/dt
        """
        # For fixed k, theta, vov, texp, generate some parameters firstly

        gamma_n, lambda_n = self.gamma_lambda(dt, self.kk)
        # the following para will change with VO and VT
        pois = self.rng_spawn[3].poisson(lam=(var_0 + var_t) * lambda_n[:, None])  # (kk, n_path)

        rv_exp_sum = self.rng_spawn[1].standard_gamma(shape=pois)
        x1 = np.sum(rv_exp_sum / gamma_n[:, None], axis=0)

        trunc_mean_x1, trunc_var_x1 = self.x1star_avgvar_mv(dt, kk=self.kk)
        trunc_scale = trunc_var_x1 / trunc_mean_x1
        trunc_shape = trunc_mean_x1 / trunc_scale * (var_0 + var_t)

        self.result['x1_trunc'] = {'shape': trunc_shape.mean(), 'scale': trunc_scale.mean()}

        x1 += trunc_scale * self.rng_spawn[1].standard_gamma(trunc_shape)
        return x1

    def x2_cdf_points_aw(self, shape, dt):
        """
        Simulation of X2 or Z from its CDF based on Abate-Whitt algorithm from formula (4.1) in Glasserman & Kim (2011)
        Used if self.tabulate_x2_z is True.

        Args:
            shape: gamma shape parameter. delta/2 for X2. 2 for Z.
            dt: time step

        Returns: (x points, cdf values)
        """

        ww, M = 0.01, 200

        mean_x2, var_x2 = self.x2star_avgvar_mv(dt, kk=0)
        mean_x2 *= shape
        var_x2 *= shape

        mu_e = mean_x2 + 12 * np.sqrt(var_x2)

        # ln_sig = np.sqrt(np.log(1 + var_x2/mean_x2**2))
        # zz = np.arange(-5, 5, 0.25)
        # x_i = mean_x2 * np.exp(ln_sig*(zz - 0.5*ln_sig))
        x_i = ww * mean_x2 + np.arange(M + 1) / M * (mu_e - ww * mean_x2)  # x1,...,x M+1
        hh = 2 * np.pi / (x_i + mu_e)

        # determine nn
        err_limit = np.pi/2 * 1e-6  # the up limit error of distribution Fx1(x)
        for nn in np.arange(1000, 5001, 100):
            if np.all(np.abs(self.x2_avgvar_mgf(1j*hh*nn, dt, shape))/nn < err_limit):
                break
        k_grid = np.arange(1, nn + 1)[:, None]
        cdf_i = np.sum(np.sin(hh * x_i * k_grid) / k_grid * self.x2_avgvar_mgf(1j * hh * k_grid, dt, shape).real, axis=0)
        cdf_i = (hh * x_i + 2 * cdf_i) / np.pi

        return x_i, cdf_i

    @functools.lru_cache()
    def x2_icdf_interp(self, shape, dt, *args, **kwargs):
        xx, cdf = self.x2_cdf_points_aw(shape, dt)
        xx = np.insert(xx, 0, 0)
        cdf = np.insert(cdf, 0, 0)
        rv = spinterp.interp1d(cdf, xx, kind='linear')
        print(f'Tabulated icdf for gamma shape={shape}')
        return rv

    def draw_eta(self, var_0, var_t, dt):
        """
        generate Bessel RV from p 285 of Glasserman & Kim (2011)

        Args:
            var_0: initial variance
            var_t: final variance
            dt: time step

        Returns:
            eta (integer >= 0) values (n, )
        """
        phi, exp = self.phi_exp(dt)
        zz = np.sqrt(var_0 * var_t) * phi

        iv_index = 0.5 * self.chi_dim() - 1
        p0 = np.power(0.5 * zz, iv_index) / (spsp.iv(iv_index, zz) * spsp.gamma(iv_index + 1))
        temp = np.arange(1, 16)[:, None]  # Bessel distribution has short tail, 30 maybe enough
        p = zz**2 / (4 * temp * (temp + iv_index))
        p = np.vstack((p0, p)).cumprod(axis=0).cumsum(axis=0)
        rv_uni = self.rv_uniform(spawn=2)
        eta = np.sum(p < rv_uni, axis=0).astype(np.uint32)

        return eta

    def draw_x2(self, shape, dt, size):
        """
        Simulation of x2/dt (or Z/dt) using truncated Gamma expansion in Glasserman & Kim (2011)
        X2 is the case with shape = delta / 2 and Z is the case with shape = 2
        
        Args:
            shape: shape parameter of gamma distribution
            dt: time-to-expiry
            size: number of RVs to generate

        Returns:
            x2/dt (or Z/dt) with shape (n_path,)
        """

        gamma_n, _ = self.gamma_lambda(dt, kk=self.kk)

        gamma_rv = self.rng_spawn[1].standard_gamma(shape, size=(self.kk, size))
        x2 = np.sum(gamma_rv / gamma_n[:, None], axis=0)

        # remainder (truncated) terms
        trunc_mean, trunc_var = self.x2star_avgvar_mv(dt, self.kk)
        trunc_scale = trunc_var / trunc_mean
        trunc_shape = trunc_mean / trunc_scale * shape

        if shape == 2:
            self.result['z_trunc'] = {'shape': trunc_shape, 'scale': trunc_scale}
        else:
            self.result['x2_trunc'] = {'shape': trunc_shape, 'scale': trunc_scale}

        x2 += trunc_scale * self.rng_spawn[1].standard_gamma(trunc_shape, size=size)
        return x2

    def cond_states_step(self, var_0, dt):

        var_t = self.var_step_ncx2(var_0=var_0, dt=dt)
        eta = self.draw_eta(var_0, var_t, dt)

        # sample int_var(integrated variance): Gamma expansion / transform inversion
        # int_var = X1+X2+X3 from formula(2.7) in Glasserman & Kim (2011)

        var_avg = self.draw_x1(var_0, var_t, dt)
        if self.tabulate_x2_z:
            interp_obj = self.x2_icdf_interp(self.chi_dim()/2, dt, k1=self.params_hash())
            var_avg += interp_obj(self.rng_spawn[1].uniform(size=self.n_path))

            interp_obj = self.x2_icdf_interp(2, dt, k1=self.params_hash())
            zz = interp_obj(self.rng_spawn[1].uniform(size=eta.sum()))
        else:
            var_avg += self.draw_x2(self.chi_dim()/2, dt, size=self.n_path)
            zz = self.draw_x2(2.0, dt, size=eta.sum())

        total = 0
        for i in np.arange(eta.max()):
            eta_above_i = (eta > i)
            count = eta_above_i.sum()
            if count == 0:
                continue
            var_avg[eta_above_i] += zz[total:total+count]
            total += count

        assert eta.sum() == total

        return var_t, var_avg


class HestonMcAndersen2008(HestonMcABC):
    """
    Heston model with conditional Monte-Carlo simulation

    Conditional MC for Heston model based on QE discretization scheme by Andersen (2008).

    Underlying price follows a geometric Brownian motion, and variance of the price follows a CIR process.

    References:
        - Andersen L (2008) Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11:1–42. https://doi.org/10.21314/JCF.2008.189

    Examples:
        >>> import numpy as np
        >>> import pyfeng.ex as pfex
        >>> strike = np.array([60, 100, 140])
        >>> spot = 100
        >>> sigma, vov, mr, rho, texp = 0.04, 1, 0.5, -0.9, 10
        >>> m = pfex.HestonMcAndersen2008(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.set_num_params(n_path=1e5, dt=1/8, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.330, 13.085, 0.296
        array([44.31943535, 13.09371251,  0.29580431])
    """
    psi_c = 1.5  # parameter used by the Andersen QE scheme
    scheme = 4  # Andersen's QE scheme, but can be overide

    def set_num_params(self, n_path=10000, dt=0.05, rn_seed=None, antithetic=True, scheme=4):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step for Euler/Milstein steps
            rn_seed: random number seed
            antithetic: antithetic
            scheme: 0 for Euler, 1 for Milstein, 2 for NCX2, 3 for Poisson-mixture Gamma, 4 for Andersen (2008)'s QE scheme

        References:
            - Andersen L (2008) Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11:1–42. https://doi.org/10.21314/JCF.2008.189
        """
        super().set_num_params(n_path, dt, rn_seed, antithetic)
        self.scheme = scheme

    def var_step_qe(self, var_0, dt):
        m, psi = self.var_mv(var_0, dt)  # put variance into psi
        psi /= m**2

        zz = self.rv_normal(spawn=0)

        # compute vt(i+1) given psi
        # psi < psi_c
        idx_below = (psi <= self.psi_c)
        ins = 2 / psi[idx_below]
        b2 = (ins - 1) + np.sqrt(ins * (ins - 1))  # b^2. Eq (27)
        a = m[idx_below] / (1 + b2)  # Eq (28)

        var_t = np.zeros(self.n_path)
        var_t[idx_below] = a * (np.sqrt(b2) + zz[idx_below])**2  # Eq (23)

        # psi_c < psi
        one_m_u = spst.norm.cdf(zz[~idx_below])  # 1 - U
        var_t_above = np.zeros_like(one_m_u)

        one_m_p = 2 / (psi[~idx_below] + 1)  # 1 - p. Eq (29)
        beta = one_m_p / m[~idx_below]  # Eq (30)

        # No need to consider (uu <= pp) & ~idx_below because the var_t value will be zero
        idx_above = (one_m_u <= one_m_p)
        var_t_above[idx_above] = (np.log(one_m_p / one_m_u) / beta)[idx_above]  # Eq (25)

        var_t[~idx_below] = var_t_above

        return var_t

    def vol_paths(self, tobs):
        var_0 = self.sigma
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        var_path = np.full((n_dt + 1, self.n_path), var_0)  # variance series: V0, V1,...,VT
        var_t = np.full(self.n_path, var_0)

        if self.scheme < 2:
            milstein = (self.scheme == 1)
            for i in range(n_dt):
                # Euler (or Milstein) scheme
                var_t = self.var_step_euler(var_t, dt[i], milstein=milstein)
                var_path[i + 1, :] = var_t
        elif self.scheme == 2:
            for i in range(n_dt):
                var_t = self.var_step_ncx2(var_t, dt[i])
                var_path[i + 1, :] = var_t
        elif self.scheme == 3:
            for i in range(n_dt):
                var_t, _ = self.var_step_ncx2_eta(var_t, dt[i])
                var_path[i + 1, :] = var_t
        elif self.scheme == 4:
            for i in range(n_dt):
                var_t = self.var_step_qe(var_t, dt[i])
                var_path[i + 1, :] = var_t
        else:
            raise ValueError(f'Invalid scheme: {self.scheme}')

        return var_path

    def cond_states_step(self, var_0, dt):

        if self.scheme < 2:
            milstein = (self.scheme == 1)
            var_t = self.var_step_euler(var_0, dt, milstein=milstein)
        elif self.scheme == 2:
            var_t = self.var_step_ncx2(var_0, dt)
        elif self.scheme == 3:
            var_t, _ = self.var_step_ncx2_eta(var_0, dt)
        elif self.scheme == 4:
            var_t = self.var_step_qe(var_0, dt)
        else:
            ValueError(f"Incorrect scheme: {self.scheme}.")

        # Trapezoidal rule
        var_avg = (var_0 + var_t)/2

        return var_t, var_avg


class HestonMcPoisTimeStep(HestonMcABC):
    """
    Heston simulation scheme Poisson-conditioned time discretization quadrature

    References:
        - Choi and Kwok (2023)

    Examples:
        >>> import numpy as np
        >>> import pyfeng.ex as pfex
        >>> strike = np.array([60, 100, 140])
        >>> spot = 100
        >>> sigma, vov, mr, rho, texp = 0.04, 1, 0.5, -0.9, 10
        >>> m = pfex.HestonMcPoisTimeStep(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.set_num_params(n_path=1e5, dt=1/8, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.330, 13.085, 0.296
        array([44.31943535, 13.09371251,  0.29580431])
    """

    def vol_paths(self, tobs):
        var_0 = self.sigma
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        var_path = np.full((n_dt + 1, self.n_path), var_0)  # variance series: V0, V1,...,VT
        var_t = np.full(self.n_path, var_0)

        for i in range(n_dt):
            var_t, _ = self.var_step_ncx2_eta(var_t, dt[i])
            var_path[i + 1, :] = var_t

        return var_path

    def cond_states_step(self, var_0, dt):

        m_x, _ = self.x1star_avgvar_mv(dt, kk=0)
        m_z, _ = self.x2star_avgvar_mv(dt, kk=0)

        var_t, eta = self.var_step_ncx2_eta(var_0, dt)
        var_avg = (var_0 + var_t)*m_x + (2*eta + 0.5*self.chi_dim())*m_z

        return var_t, var_avg

    def avgvar_var_unexplained(self, texp, dt=None):
        """
        Unexplained variance ratio of average variance
        This is valid only for time discretisation with Poisson conditioning.

        Args:
            texp: time to expiry
            dt: time step

        Returns:
            ratio
        """

        dt = dt or self.dt
        mean, var = self.avgvar_mv(texp)

        m_x, v_x = self.x1star_avgvar_mv(dt, kk=0)
        m_z, v_z = self.x2star_avgvar_mv(dt, kk=0)

        vov2dt = self.vov**2 * dt
        unex = (v_x*2 + v_z*4/vov2dt) * mean * dt / texp
        return unex / var


class HestonMcTseWan2013(HestonMcABC):
    """
    Almost exact MC for Heston model.

    Underlying price is assumed to follow a geometric Brownian motion.
    Volatility (variance) of the price is assumed to follow a CIR process.
    Example:
        >>> import numpy as np
        >>> import pyfeng.ex as pfex
        >>> strike = np.array([60, 100, 140])
        >>> spot = 100
        >>> sigma, vov, mr, rho, texp = 0.04, 1, 0.5, -0.9, 10
        >>> m = pfex.HestonMcTseWan2013(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.set_num_params(n_path=1e4, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.330, 13.085, 0.296
        array([12.08981758,  0.33379748, 42.28798189])  # not close so far
    """
    dist = 'ig'

    def set_num_params(self, n_path=10000, dt=None, rn_seed=None, scheme=3, dist=None):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step
            rn_seed: random number seed
            scheme: simulation scheme for jumping from 0 to texp
            dist: distribution to use for approximation.
                'ig' for inverse Gaussian (default), 'ga' for Gamma, 'ln' for LN
        """
        super().set_num_params(n_path, dt, rn_seed, scheme=scheme)
        if dist is not None:
            self.dist = dist

    def cond_states_step(self, var_0, dt):

        var_t = self.var_step_ncx2(var_0, dt)
        m1, var = self.cond_avgvar_mv(var_0, var_t, dt, eta=None, kk=0)

        if self.dist.lower() == 'ig':
            # mu and lambda defined in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
            # RNG.wald takes the same parameters
            lam = m1**3 / var
            var_avg = self.rng_spawn[1].wald(mean=m1, scale=lam)
        elif self.dist.lower() == 'ga':
            scale = var / m1
            shape = m1 / scale
            var_avg = scale * self.rng_spawn[1].standard_gamma(shape=shape)
        elif self.dist.lower() == 'ln':
            scale = np.sqrt(np.log(1 + var/m1**2))
            var_avg = m1 * np.exp(scale * (self.rv_normal(spawn=1) - scale/2))
        else:
            raise ValueError(f"Incorrect distribution: {self.dist}.")

        return var_t, var_avg


class HestonMcChoiKwok2023(HestonMcABC):

    dist = 'ig'  # distribution for series truncation
    kk = 1  # K for series truncation.

    def set_num_params(self, n_path=10000, dt=None, rn_seed=None, antithetic=True, kk=1):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step
            rn_seed: random number seed
            kk: truncation index
        """
        super().set_num_params(n_path, dt, rn_seed, antithetic)
        self.kk = kk

    def draw_x123(self, var_sum, dt, shape_sum):
        """
        Samples of (X1 + X2 + X3)/dt using truncated Gamma expansion improved in Choi & Kwok (2023)

        Args:
            var_sum: sum of v_t at the observation times. (n_paths,)
            shape_sum: sum of gamma shape parameters
            dt: time step

        Returns:
            (X1 + X2 + X3)/dt  (n_paths,)
        """
        gamma_n, lambda_n = self.gamma_lambda(dt, kk=self.kk)

        if self.kk > 0:
            pois = self.rng_spawn[3].poisson(lam=var_sum * lambda_n[:, None])
            x123 = np.sum(self.rng_spawn[1].standard_gamma(shape=pois + shape_sum) / gamma_n[:, None], axis=0)
        else:
            x123 = np.zeros_like(var_sum)

        trunc_mean, trunc_var = self.x1star_avgvar_mv(dt, self.kk)
        trunc_mean *= var_sum
        trunc_var *= var_sum

        mean_x23, var_x23 = self.x2star_avgvar_mv(dt, self.kk)
        trunc_mean += mean_x23 * shape_sum
        trunc_var += var_x23 * shape_sum

        if self.dist.lower() == 'ig':
            lam = trunc_mean**3 / trunc_var
            x123 += self.rng_spawn[1].wald(mean=trunc_mean, scale=lam)
        elif self.dist.lower() == 'ga':
            trunc_scale = trunc_var / trunc_mean
            trunc_shape = trunc_mean / trunc_scale
            self.result['x123_trunc'] = {'shape': trunc_shape.mean(), 'scale': trunc_scale.mean()}

            x123 += trunc_scale * self.rng_spawn[1].standard_gamma(trunc_shape)
        elif self.dist.lower() == 'ln':
            scale = np.sqrt(np.log(1 + trunc_var / trunc_mean**2))
            x123 += trunc_mean * np.exp(scale * (self.rv_normal(spawn=1) - scale / 2))
        else:
            raise ValueError(f"Incorrect distribution: {self.dist}.")

        return x123

    def cond_states_step(self, var_0, dt):

        var_t = np.full(self.n_path, var_0)
        var_t, eta = self.var_step_ncx2_eta(var_t, dt)
        shape = 0.5 * self.chi_dim() + 2*eta

        # self.draw_x123 returns the average by dt. Need to convert to the average by texp
        var_avg = self.draw_x123(var_0 + var_t, dt, shape)

        return var_t, var_avg
