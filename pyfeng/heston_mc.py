import abc
import numpy as np
import scipy.stats as spst
import scipy.special as spsp
import scipy.integrate as spint
import scipy.optimize as spop
from scipy import interpolate
from scipy.misc import derivative
from . import sv_abc as sv


class HestonMcABC(sv.SvABC, sv.CondMcBsmABC, abc.ABC):
    var_process = True
    model_type = "Heston"
    scheme = None

    def chi_dim(self):
        """
        Noncentral Chi-square (NCX) distribution's degree of freedom

        Returns:
            degree of freedom (scalar)
        """
        chi_dim = 4 * self.theta * self.mr / self.vov**2
        return chi_dim

    def chi_lambda(self, df):
        """
        Noncentral Chi-square (NCX) distribution's noncentrality parameter

        Returns:
            noncentrality parameter (scalar)
        """
        chi_lambda = 4 * self.sigma * self.mr / self.vov**2
        chi_lambda /= np.exp(self.mr * df) - 1
        return chi_lambda

    def phi_exp(self, texp):
        exp = np.exp(-self.mr*texp/2)
        phi = 4*self.mr / self.vov**2 / (1/exp - exp)
        return phi, exp

    def var_mv(self, var_0, dt):
        """
        Mean and variance of the variance V(t+dt) given V(0) = var_0

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            mean, variance
        """

        expo = np.exp(-self.mr * dt)
        m = self.theta + (var_0 - self.theta) * expo
        s2 = var_0 * expo + self.theta * (1 - expo) / 2
        s2 *= self.vov**2 * (1 - expo) / self.mr
        return m, s2

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

    def cond_spot_sigma(self, var_0, texp):
        var_final, var_avg = self.cond_states(var_0, texp)

        spot_cond = ((var_final - var_0) - self.mr * texp * (self.theta - var_avg)) / self.vov \
             - 0.5 * self.rho * var_avg * texp
        np.exp(self.rho * spot_cond, out=spot_cond)
        sigma_cond = np.sqrt((1.0 - self.rho**2) / var_0 * var_avg)  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond


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
        >>> m.set_mc_params(n_path=1e5, dt=1/8, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.330, 13.085, 0.296
        array([44.31943535, 13.09371251,  0.29580431])
    """
    psi_c = 1.5  # parameter used by the Andersen QE scheme
    scheme = 4

    def set_mc_params(self, n_path=10000, dt=0.05, rn_seed=None, antithetic=True, scheme=4):
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

    def var_step_qe(self, var_0, dt):
        m, s2 = self.var_mv(var_0, dt)
        psi = s2 / m**2

        zz = self.rv_normal(spawn=0)

        # compute vt(i+1) given psi
        # psi < psi_c
        idx_below = (psi <= self.psi_c)
        ins = 2 / psi[idx_below]
        b2 = (ins - 1) + np.sqrt(ins * (ins - 1))
        a = m[idx_below] / (1 + b2)

        var_t = np.zeros(self.n_path)
        var_t[idx_below] = a * (np.sqrt(b2) + zz[idx_below])**2

        # psi_c < psi
        one_m_u = spst.norm.cdf(zz[~idx_below])  # 1 - U
        var_t_above = np.zeros_like(one_m_u)

        one_m_p = 2 / (psi[~idx_below] + 1)  # 1 - p
        beta = one_m_p / m[~idx_below]

        # No need to consider (uu <= pp) & ~idx_below because the var_t value will be zero
        idx_above = (one_m_u <= one_m_p)
        var_t_above[idx_above] = (np.log(one_m_p / one_m_u) / beta)[idx_above]

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

    def cond_states_old(self, var_0, texp):

        tobs = self.tobs(texp)
        n_dt = len(tobs)
        var_paths = self.vol_paths(tobs)
        var_final = var_paths[-1, :]
        var_avg = spint.simps(var_paths, dx=1, axis=0) / n_dt

        return var_final, var_avg

    def cond_states(self, var_0, texp):

        tobs = self.tobs(texp)
        n_dt = len(tobs)
        dt = np.diff(tobs, prepend=0)

        # precalculate the Simpson's rule weight
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
                var_t = self.var_step_ncx2(var_t, dt[i])
                var_avg += weight[i + 1] * var_t

        elif self.scheme == 3:
            for i in range(n_dt):
                var_t, _ = self.var_step_ncx2_eta(var_t, dt[i])
                var_avg += weight[i + 1] * var_t

        elif self.scheme == 4:
            for i in range(n_dt):
                var_t = self.var_step_qe(var_t, dt[i])
                var_avg += weight[i + 1] * var_t

        return var_t, var_avg  # * texp


class HestonMcGlassermanKim2011(HestonMcABC):
    """
    Exact simulation using the gamma series based on Glasserman & Kim (2011)

    References:
        - Glasserman P, Kim K-K (2011) Gamma expansion of the Heston stochastic volatility model. Finance Stoch 15:267–296. https://doi.org/10.1007/s00780-009-0115-y
    """

    antithetic = False
    scheme = 3
    KK = 1  # K for series truncation.

    def set_mc_params(self, n_path=10000, dt=None, rn_seed=None, scheme=3, KK=1):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step
            rn_seed: random number seed
            scheme: simulation scheme for variance
            KK: truncation index

        """
        super().set_mc_params(n_path, dt, rn_seed, antithetic=False)
        self.scheme = scheme
        self.KK = KK

    def gamma_lambda(self, dt, KK=None):
        """
        gamma_n and lambda_n below Eq. (2.8) in Glasserman & Kim (2011).
        gamma_n is the original value * dt to make x1, x2, and x3 the average variance.

        Args:
            dt: time step
            KK: number of terms

        Returns:
            gamma_n, lambda_n
        """

        if KK is None:
            KK = self.KK

        mrt2 = (self.mr * dt)**2
        vov2dt = self.vov**2 * dt

        n_2pi_2 = (np.arange(1, KK + 1) * 2 * np.pi)**2
        gamma_n = (mrt2 + n_2pi_2) / (2 * vov2dt)  # / dt
        lambda_n = 4 * n_2pi_2 / vov2dt / (mrt2 + n_2pi_2)

        return gamma_n, lambda_n

    def x1star_avgvar_mv_asymp(self, dt, KK=0):
        """
        Asymptotic mean and variance of the truncated terms of X1/dt in Lemma 3.1 in Glasserman & Kim (2011)
        (v_0 + v_t) need to be multiplied to mean and variance afterwards.

        Args:
            dt: time step
            KK: number of terms

        Returns:
            mean, variance
        """

        vov2dt = self.vov**2 * dt
        # remainder (truncated) terms
        rem_mean = 2 / (np.pi**2 * KK)
        rem_var = 2 / (3 * np.pi**4 * KK**3)

        return rem_mean, rem_var * vov2dt

    def x1star_avgvar_mv(self, dt, KK=0):
        """
        Mean and variance of the truncated terms of (X1/dt) in p 281-282 Glasserman & Kim (2011)
        (v_0 + v_t) need to be multiplied to mean and variance afterwards.

        Args:
            dt: time step
            KK: # of exact terms

        Returns:
            mean, variance
        """

        mrt_h = self.mr * dt / 2
        vov2dt = self.vov**2 * dt
        csch = 1 / np.sinh(mrt_h)
        coth = np.cosh(mrt_h) * csch

        x1_mean = (coth/mrt_h - csch**2) / 2
        x1_var = (coth / mrt_h**3 + csch**2 / mrt_h**2 - 2 * coth*csch**2 / mrt_h) / 8

        if KK > 0:
            n_2pi_2 = (np.arange(1, KK + 1) * 2 * np.pi)**2
            term = 8 * n_2pi_2 / (4*mrt_h**2 + n_2pi_2)**2
            x1_mean -= np.sum(term)
            x1_var -= np.sum(4 * term / (4*mrt_h**2 + n_2pi_2))

        return x1_mean, x1_var * vov2dt

    def x2star_avgvar_mv_asymp(self, dt, KK=0):
        """
        Asymptotic mean and variance of the truncated terms of X2/dt (with delta=1) in Lemma 3.1 in Glasserman & Kim (2011)

        Args:
            dt: time step
            KK: # of exact terms

        Returns:
            mean, variance
        """

        vov2dt = self.vov**2 * dt
        mean = 1 / (4 * np.pi**2 * KK)
        var = 1 / (24 * np.pi**4 * KK**3)

        return mean * vov2dt, var * vov2dt**2

    def x2star_avgvar_mv(self, dt, KK=0):
        """
        Mean and variance of the truncated terms of X2/dt (with delta=1) in p 284 in Glasserman & Kim (2011)

        Args:
            dt: time step
            KK: # of exact terms

        Returns:
            mean, variance
        """

        mrt_h = self.mr * dt / 2
        vov2dt = self.vov**2 * dt

        csch = 1 / np.sinh(mrt_h)
        coth = np.cosh(mrt_h) * csch

        mean = (mrt_h * coth - 1) / (8 * mrt_h**2)
        var = (mrt_h * coth + mrt_h**2 * csch**2 - 2) / (32 * mrt_h**4)

        if KK > 0:
            term = 1 / (4*mrt_h**2 + (np.arange(1, KK + 1) * 2 * np.pi)**2)
            mean -= np.sum(term)
            var -= 2 * np.sum(term**2)

        return mean * vov2dt, var * vov2dt**2

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

        gamma_n, lambda_n = self.gamma_lambda(dt, self.KK)

        # the following para will change with VO and VT
        pois = self.rng_spawn[3].poisson(lam=(var_0 + var_t) * lambda_n[:, None])  # (KK, n_path)

        rv_exp_sum = self.rng_spawn[1].standard_gamma(shape=pois)
        x1 = np.sum(rv_exp_sum / gamma_n[:, None], axis=0)

        rem_mean_x1, rem_var_x1 = self.x1star_avgvar_mv(dt, self.KK)
        rem_scale = rem_var_x1 / rem_mean_x1
        rem_shape = rem_mean_x1 / rem_scale * (var_0 + var_t)

        x1 += rem_scale * self.rng_spawn[1].standard_gamma(rem_shape)
        return x1

    def draw_X2_AW(self, mu_X2_0, sigma_square_X2_0, ncx_df, texp, num_rv):
        """
        Simulation of X2 or Z from its CDF based on Abate-Whitt algorithm from formula (4.1) in Glasserman & Kim (2011)

        Parameters
        ----------
        mu_X2_0:  float
            mean of X2 from formula(4.2)
        sigma_square_X2_0: float
            variance of X2 from formula(4.3)
        ncx_df: float
            a parameter, which equals to 4*theta*k / (vov**2) when generating X2 and equals to 4 when generating Z
        texp: float
            time-to-expiry
        num_rv: int
            number of random variables you want to generate

        Returns
        -------
         an 1-d array with shape (num_rv,), random variables X2 or Z
        """

        mu_X2 = ncx_df * mu_X2_0
        sigma_square_X2 = ncx_df * sigma_square_X2_0

        mu_e = mu_X2 + 14 * np.sqrt(sigma_square_X2)
        w = 0.01
        M = 200
        xi = w * mu_X2 + np.arange(M + 1) / M * (mu_e - w * mu_X2)  # x1,...,x M+1
        L = lambda x: np.sqrt(2 * self.vov**2 * x + self.mr**2)
        fha_2 = lambda x: (L(x) / self.mr * (np.sinh(0.5 * self.mr * texp) / np.sinh(0.5 * L(x) * texp)))**(
                    0.5 * ncx_df)
        fha_2_vec = np.vectorize(fha_2)
        err_limit = np.pi * 1e-5 * 0.5  # the up limit error of distribution Fx1(x)

        h = 2 * np.pi / (xi + mu_e)
        # increase N to make truncation error less than up limit error, N is sensitive to xi and the model parameter
        F_X2_part = np.zeros(len(xi))
        for pos in range(len(xi)):
            Nfunc = lambda N: abs(fha_2(-1j * h[pos] * N)) - err_limit * N
            N = int(spop.brentq(Nfunc, 0, 5000)) + 1
            N_all = np.arange(1, N + 1)
            F_X2_part[pos] = np.sum(np.sin(h[pos] * xi[pos] * N_all) * fha_2_vec(-1j * h[pos] * N_all).real / N_all)

        F_X2 = (h * xi + 2 * F_X2_part) / np.pi

        # Next we can sample from this tabulated distribution using linear interpolation
        rv_uni = self.rng.uniform(size=num_rv)

        xi = np.insert(xi, 0, 0.)
        F_X2 = np.insert(F_X2, 0, 0.)
        F_X2_inv = interpolate.interp1d(F_X2, xi, kind="slinear")
        X2 = F_X2_inv(rv_uni)

        return X2

    def eta_mv(self, var_0, var_t, texp):
        """
        The mean and variance of eta RV.

        Args:
            var_0: initial variance
            var_t: final variance
            texp: time step

        Returns:
            eta (n_path, 1)
        """
        phi, exp = self.phi_exp(texp)
        zz = np.sqrt(var_0 * var_t) * phi

        iv_index = 0.5 * self.chi_dim() - 1
        iv0 = spsp.iv(iv_index, zz)
        iv1 = spsp.iv(iv_index + 1, zz)
        iv2 = spsp.iv(iv_index + 2, zz)

        mean = (zz/2) * (iv1/iv0)
        var = (zz/2)**2 * (iv2/iv0) + mean - mean**2

        return mean, var

    def draw_eta(self, var_0, var_t, texp):
        """
        generate Bessel RV from p 285 of Glasserman & Kim (2011)

        Args:
            var_0: initial variance
            var_t: final variance
            texp: time step

        Returns:
            eta (integer >= 0) values (n, )
        """
        phi, exp = self.phi_exp(texp)
        zz = np.sqrt(var_0 * var_t) * phi

        iv_index = 0.5 * self.chi_dim() - 1
        p0 = np.power(0.5 * zz, iv_index) / (spsp.iv(iv_index, zz) * spsp.gamma(iv_index + 1))
        temp = np.arange(1, 16)[:, None]  # Bessel distribution has short tail, 30 maybe enough
        p = zz**2 / (4 * temp * (temp + iv_index))
        p = np.vstack((p0, p)).cumprod(axis=0).cumsum(axis=0)
        rv_uni = self.rv_uniform(spawn=2)
        eta = np.sum(p < rv_uni, axis=0).astype(np.uint32)

        return eta

    def draw_x2(self, ncx_df, dt, size):
        """
        Simulation of x2/dt (or Z/dt) using truncated Gamma expansion in Glasserman & Kim (2011)
        Z is the special case with ncx_df = 4
        
        Args:
            ncx_df: ncx2 degree of freedom
            dt: time-to-expiry
            size: number of RVs to generate

        Returns:
            x2/dt (or Z/dt) with shape (n_path,)
        """

        gamma_n, _ = self.gamma_lambda(dt)

        gamma_rv = self.rng_spawn[1].standard_gamma(ncx_df / 2, size=(self.KK, size))
        x2 = np.sum(gamma_rv / gamma_n[:, None], axis=0)

        # remainder (truncated) terms
        rem_mean, rem_var = self.x2star_avgvar_mv(dt, self.KK)
        rem_scale = rem_var / rem_mean
        rem_shape = rem_mean / rem_scale * ncx_df

        x2 += rem_scale * self.rng_spawn[1].standard_gamma(rem_shape, size=size)
        return x2

    def cond_avgvar_mv(self, var_0, var_t, dt, eta=None, KK=0):
        """
        Mean and variance of the average variance conditional on initial var, final var, and eta

        Args:
            var_0: initial variance
            var_t: final variance
            eta: Poisson RV
            dt: time step
            KK: number of exact terms

        Returns:
            mean, variance
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

        x1_mean, x1_var = self.x1star_avgvar_mv(dt, KK=KK)
        x1_mean *= (var_0 + var_t)
        x1_var *= (var_0 + var_t)

        z_mean, z_var = self.x2star_avgvar_mv(dt, KK=KK)
        z_mean *= 4
        z_var *= 4

        x23_mean = (eta_mean + self.chi_dim()/4) * z_mean
        x23_var = (eta_mean + self.chi_dim()/4) * z_var
        x23_var += eta_var * z_mean**2

        return x1_mean + x23_mean, x1_var + x23_var

    def cond_states(self, var_0, texp):

        var_t, _ = self.var_step_ncx2_eta(var_0=var_0, dt=texp)

        # sample int_var(integrated variance): Gamma expansion / transform inversion
        # int_var = X1+X2+X3 from formula(2.7) in Glasserman & Kim (2011)

        # Simulation X1: truncated Gamma expansion
        var_avg = self.draw_x1(var_0, var_t, texp)
        var_avg += self.draw_x2(self.chi_dim(), texp, size=self.n_path)
        eta = self.draw_eta(var_0, var_t, texp)

        zz = self.draw_x2(4, texp, size=eta.sum())

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


class HestonMcTseWan2013(HestonMcGlassermanKim2011):
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
        >>> m.set_mc_params(n_path=1e4, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.330, 13.085, 0.296
        array([12.08981758,  0.33379748, 42.28798189])  # not close so far
    """
    dist = 0

    def set_mc_params(self, n_path=10000, dt=None, rn_seed=None, scheme=3, dist=0):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step
            rn_seed: random number seed
            scheme: simulation scheme for jumping from 0 to texp
            dist: distribution to use for approximation. 0 for inverse Gaussian (default), 1 for lognormal.
        """
        super().set_mc_params(n_path, dt, rn_seed, scheme=scheme)
        self.dist = dist

    def mgf(self, aa, var_0, var_t, dt):
        """
        MGF of the average variance given the initial and final variance

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

        gamma = np.sqrt(mrt**2 - 2 * vov2dt * aa)

        var_mean = np.sqrt(var_0 * var_t)
        phi_mr, _ = self.phi_exp(dt)
        cosh_mr = np.cosh(mrt / 2)

        phi_gamma = 2 * gamma / vov2dt / np.sinh(gamma / 2)
        cosh_gamma = np.cosh(gamma / 2)

        #part1 = gamma * np.exp(-0.5 * (gamma * texp - self.mr * texp)) * (1 - decay) / (self.mr * (1 - decay_gamma))
        part1 = phi_gamma / phi_mr

        #part2 = np.exp((var_0 + var_final) / vov2dt
        #    * (self.mr * (1 + decay) / (1 - decay) - gamma * (1 + decay_gamma) / (1 - decay_gamma)))
        part2 = np.exp((var_0 + var_t) * (cosh_mr * phi_mr - cosh_gamma * phi_gamma) / 2)

        part3 = spsp.iv(iv_index, var_mean * phi_gamma) / spsp.iv(iv_index, var_mean * phi_mr)

        ch_f = part1 * part2 * part3
        return ch_f

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
        # conditional MGF function
        def mgf_cond(aa):
            return self.mgf(aa, var_0, var_t, dt)

        # Get the first 2 moments
        m1 = derivative(mgf_cond, 0, n=1, dx=1e-5)
        m2 = derivative(mgf_cond, 0, n=2, dx=1e-5)
        return m1, m2 - m1**2

    def cond_states(self, var_0, texp):

        var_t, eta = self.var_step_ncx2_eta(self.sigma, texp)
        #m1, var = self.cond_avgvar_mv_numeric(var_0, var_t, texp)
        m1, var = self.cond_avgvar_mv(var_0, var_t, texp, eta=None)

        if self.dist == 0:
            # mu and lambda defined in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
            # RNG.wald takes the same parameters
            lam = m1**3 / var
            var_avg = self.rng_spawn[1].wald(mean=m1, scale=lam)
        elif self.dist == 1:
            scale = var / m1
            shape = m1 / scale
            var_avg = scale * self.rng_spawn[1].standard_gamma(shape=shape)
        elif self.dist == 2:
            scale = np.sqrt(np.log(1 + var/m1**2))
            var_avg = self.rv_normal(spawn=1)
            var_avg = m1 * np.exp(scale * (var_avg - scale/2))
        else:
            raise ValueError(f"Incorrect distribution: {self.dist}.")

        return var_t, var_avg


class HestonMcChoiKwok2023(HestonMcGlassermanKim2011):

    dist = 0

    def set_mc_params(self, n_path=10000, dt=None, rn_seed=None, scheme=3, KK=0, dist=0):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step
            rn_seed: random number seed
            scheme: simulation scheme for jumping from 0 to texp
            dist: distribution to use for approximation. 0 for inverse Gaussian (default), 1 for Gamma, 2 for LN
        """
        super().set_mc_params(n_path, dt, rn_seed, scheme=scheme, KK=KK)
        self.dist = dist

    def draw_x123(self, var_sum, dt, eta_sum):
        """
        Samples of (X1 + X2 + X3)/dt using truncated Gamma expansion improved in Choi & Kwok (2023)

        Args:
            var_sum: sum of v_t at the observation times. (n_paths,)
            eta_sum: sum of Bessel RVs
            dt: time step

        Returns:
            (X1 + X2 + X3)/dt  (n_paths,)
        """
        gamma_n, lambda_n = self.gamma_lambda(dt)

        pois = self.rng_spawn[3].poisson(lam=var_sum * lambda_n[:, None])
        x123 = np.sum(self.rng_spawn[1].standard_gamma(shape=pois + eta_sum * 2) / gamma_n[:, None], axis=0)

        # The approximated mean and variance of the truncated terms
        #rem_mean_x1 = 2 * dt / (np.pi**2 * self.KK) * var_sum
        #rem_var_x1 = 2 * self.vov**2 * dt**3 / (3 * np.pi**4 * self.KK**3) * var_sum
        #rem_mean_x23 = (self.vov * dt)**2 / (4 * np.pi**2 * self.KK) * ncx_df
        #rem_var_x23 = (self.vov * dt)**4 / (24 * np.pi**4 * self.KK**3) * ncx_df

        m1, var = self.x1star_avgvar_mv(dt, self.KK)
        m1 *= var_sum
        var *= var_sum

        m1_x23, var_x23 = self.x2star_avgvar_mv(dt, self.KK)
        m1 += m1_x23 * 4 * eta_sum
        var += var_x23 * 4 * eta_sum

        if self.dist == 0:
            lam = m1**3 / var
            x123 += self.rng_spawn[1].wald(mean=m1, scale=lam)
        elif self.dist == 1:
            rem_scale = var / m1
            rem_shape = m1 / rem_scale
            x123 += rem_scale * self.rng_spawn[1].standard_gamma(rem_shape)
        elif self.dist == 2:
            scale = np.sqrt(np.log(1 + var / m1**2))
            x123 += m1 * np.exp(scale * (self.rv_normal(spawn=1) - scale / 2))
        else:
            raise ValueError(f"Incorrect distribution: {self.dist}.")

        return x123

    def cond_states(self, var_0, texp):

        tobs = self.tobs(texp)
        n_dt = len(tobs)
        dt = np.diff(tobs, prepend=0)

        # precalculate the weights: 1, 2, 1, ..., 2, 1
        weight = np.ones(n_dt + 1)
        weight[1:-1] = 2

        var_t = np.full(self.n_path, var_0)
        var_sum = weight[0] * var_t
        eta_sum = np.zeros_like(var_t)

        for i in range(n_dt):
            var_t, eta = self.var_step_ncx2_eta(var_t, dt[i])
            var_sum += weight[i+1] * var_t
            eta_sum += eta

        eta_sum += self.chi_dim() / 4 * n_dt
        # self.draw_x123 returns the average by dt. Need to convert to the average by texp
        var_avg = self.draw_x123(var_sum, dt[0], eta_sum) / n_dt

        return var_t, var_avg

