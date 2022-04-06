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

    def var_mean_var(self, var_0, dt):
        """
        Mean and variance of the variance V(t+dt) given V(0) = var_0

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            (mean, variance)
        """

        expo = np.exp(-self.mr * dt)
        m = self.theta + (var_0 - self.theta) * expo
        s2 = var_0 * expo + self.theta * (1 - expo) / 2
        s2 *= self.vov**2 * (1 - expo) / self.mr
        return m, s2

    def var_step_ncx2(self, var_0, dt):
        """
        Draw final variance from NCX distribution

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            final variance (at t=T)
        """
        chi_df = self.chi_dim()
        phi, exp = self.phi_exp(dt)
        chi_nonc = var_0 * exp * phi
        var_t = (exp / phi) * self.rng.noncentral_chisquare(df=chi_df, nonc=chi_nonc, size=self.n_path)
        return var_t

    def var_step_ncx2_eta(self, var_0, dt):
        """
        Draw final variance from NCX2 distribution

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            final variance and eta (at t=T)
        """
        chi_df = self.chi_dim()
        phi, exp = self.phi_exp(dt)
        chi_nonc = var_0 * exp * phi
        nn = self.rng.poisson(chi_nonc / 2, size=self.n_path)
        var_t = (exp / phi) * 2 * self.rng.standard_gamma(shape=chi_df / 2 + nn, size=self.n_path)
        return var_t, nn

    def cond_var_mean(self, var_0, var_t, eta, dt):
        """
        Average variance conditional on initial var, final var, and eta

        Args:
            var_0: initial variance
            var_t: final variance
            eta: Poisson RV
            dt: time step

        Returns:
            (integarted variance / dt)
        """

        # x = np.arange(1, 10) * 0.02
        # y1 = 1 / x / np.tanh(x) - 1 / np.sinh(x)**2
        # y2 = 2 / 3 - (4 / 45) * x**2 + (4 / 315) * x**4 - (8 / 4725) * x**6
        # y2 - y1
        # y1 = (x / np.tanh(x) - 1) / x**2
        # y2 = 1 / 3 - (1 / 45) * x**2 + (2 / 945) * x**4
        # y2 - y1

        mrt_half = 0.5 * self.mr * dt
        csch = 1 / np.sinh(mrt_half)
        coth = np.cosh(mrt_half) * csch

        X1_mean = (coth/mrt_half - csch**2) * (var_0 + var_t) / 2
        Z_mean = 0.5 * self.vov**2 * dt * (mrt_half * coth - 1) / mrt_half**2
        Z_mean *= (eta + self.chi_dim()/4)
        return X1_mean + Z_mean  # * dt


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

        spot_cond = ((var_final - var_0) - self.mr * texp * (self.theta - var_mean)) / self.vov \
             - 0.5 * self.rho * var_mean * texp
        np.exp(self.rho * spot_cond, out=spot_cond)
        sigma_cond = np.sqrt((1.0 - self.rho**2) / var_0 * var_mean)  # normalize by initial variance

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
    scheme = 2

    def set_mc_params(self, n_path=10000, dt=0.05, rn_seed=None, antithetic=True, scheme=2):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step for Euler/Milstein steps
            rn_seed: random number seed
            antithetic: antithetic
            scheme: 0 for Euler, 1 for Milstein, 2 for Andersen (2008)'s QE scheme (default)

        References:
            - Andersen L (2008) Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11:1–42. https://doi.org/10.21314/JCF.2008.189
        """
        super().set_mc_params(n_path, dt, rn_seed, antithetic)
        self.scheme = scheme

    def var_step_euler(self, var_0, dt, milstein=False):
        zz = self.rv_normal()

        # Euler (or Milstein) scheme
        var_t = var_0 + self.mr * (self.theta - var_0) * dt + np.sqrt(var_0) * self.vov * zz
        # Extra-term for Milstein scheme
        if milstein:
            var_t += 0.25 * self.vov**2 * (zz**2 - dt)

        var_t[var_t < 0] = 0  # variance should be larger than zero
        return var_t

    def var_step_qe(self, var_0, dt):
        m, s2 = self.var_mean_var(var_0, dt)
        psi = s2 / m**2

        zz = self.rv_normal()

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
                var_t = self.var_step_qe(var_t, dt[i])
                var_path[i + 1, :] = var_t

        elif self.scheme == 3:
            for i in range(n_dt):
                var_t = self.var_step_ncx2(var_t, dt[i])
                var_path[i + 1, :] = var_t

        elif self.scheme == 4:
            for i in range(n_dt):
                var_t, _ = self.var_step_ncx2_eta(var_t, dt[i])
                var_path[i + 1, :] = var_t

        else:
            raise ValueError(f'Invalid scheme: {self.scheme}')

        return var_path

    def cond_states_old(self, var_0, texp):

        tobs = self.tobs(texp)
        n_dt = len(tobs)
        var_paths = self.vol_paths(tobs)
        var_final = var_paths[-1, :]
        var_mean = spint.simps(var_paths, dx=1, axis=0) / n_dt

        return var_final, var_mean

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

        if self.scheme < 2:
            milstein = (self.scheme == 1)
            for i in range(n_dt):
                # Euler (or Milstein) scheme
                var_t = self.var_step_euler(var_t, dt[i], milstein=milstein)
                var_mean += weight[i + 1] * var_t

        elif self.scheme == 2:
            for i in range(n_dt):
                var_t = self.var_step_qe(var_t, dt[i])
                var_mean += weight[i + 1] * var_t

        elif self.scheme == 3:
            for i in range(n_dt):
                var_t = self.var_step_ncx2(var_t, dt[i])
                var_mean += weight[i + 1] * var_t

        elif self.scheme == 4:
            for i in range(n_dt):
                var_t, _ = self.var_step_ncx2_eta(var_t, dt[i])
                var_mean += weight[i + 1] * var_t

        return var_t, var_mean  # * texp


class HestonMcAe(HestonMcABC):
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
        >>> m = pfex.HestonMcAe(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.set_mc_params(n_path=1e4, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.330, 13.085, 0.296
        array([12.08981758,  0.33379748, 42.28798189])  # not close so far
    """
    dist = 0

    def set_mc_params(self, n_path=10000, rn_seed=None, antithetic=True, dist=0):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            rn_seed: random number seed
            antithetic: antithetic
            dist: distribution to use for approximation. 0 for inverse Gaussian (default), 1 for lognormal.
        """
        self.n_path = int(n_path)
        self.rn_seed = rn_seed
        self.antithetic = antithetic
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)
        self.dist = dist

    def mgf(self, aa, var_0, var_final, dt):
        """
            MGF of the integrated variance given the initial and final variance

        Args:
            aa: dummy variable in the transformation
            var_0: initial variance
            var_final: final variance
            dt: time step

        Returns:
            Conditional MGF at dummy variable aa
        """

        vov2 = self.vov**2
        iv_index = 0.5 * self.chi_dim() - 1

        gamma = np.sqrt(self.mr**2 - 2 * vov2 * aa)
        #decay = np.exp(-self.mr * texp)
        #decay_gamma = np.exp(-gamma * texp)

        var_mean = np.sqrt(var_0 * var_final)
        phi_mr = 2 * self.mr / vov2 / np.sinh(self.mr * dt / 2)
        cosh_mr = np.cosh(self.mr * dt / 2)

        phi_gamma = 2 * gamma / vov2 / np.sinh(gamma * dt / 2)
        cosh_gamma = np.cosh(gamma * dt / 2)

        #part1 = gamma * np.exp(-0.5 * (gamma * texp - self.mr * texp)) * (1 - decay) / (self.mr * (1 - decay_gamma))
        part1 = phi_gamma / phi_mr

        #part2 = np.exp((var_0 + var_final) / vov2
        #    * (self.mr * (1 + decay) / (1 - decay) - gamma * (1 + decay_gamma) / (1 - decay_gamma)))
        part2 = np.exp((var_0 + var_final)*(cosh_mr*phi_mr - cosh_gamma*phi_gamma)/2)

        part3 = spsp.iv(iv_index, var_mean * phi_gamma) / spsp.iv(iv_index, var_mean * phi_mr)

        ch_f = part1 * part2 * part3
        return ch_f

    def cond_states(self, var_0, texp):

        var_final = self.var_step_ncx2(self.sigma, texp)

        # conditional MGF function
        def mgf_cond(aa):
            return self.mgf(aa, var_0, var_final, texp)

        # Get the first 2 moments
        m1 = derivative(mgf_cond, 0, n=1, dx=1e-5)
        m2 = derivative(mgf_cond, 0, n=2, dx=1e-5)

        if self.dist == 0:
            # mu and lambda defined in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
            # RNG.wald takes the same parameters
            mu = m1
            lam = m1**3 / (m2 - m1**2)
            var_mean = self.rng.wald(mean=mu, scale=lam) / texp
        elif self.dist == 1:
            scale_ln = np.sqrt(np.log(m2) - 2 * np.log(m1))
            miu_ln = np.log(m1) - 0.5 * scale_ln**2
            var_mean = self.rng.lognormal(mean=miu_ln, sigma=scale_ln) / texp
        else:
            raise ValueError(f"Incorrect distribution.")

        return var_final, var_mean


class HestonMcGlassermanKim2011(HestonMcABC):
    """
    Exact simulation using the gamma series based on Glasserman and Kim (2011)

    References:
        - Glasserman P, Kim K-K (2011) Gamma expansion of the Heston stochastic volatility model. Finance Stoch 15:267–296. https://doi.org/10.1007/s00780-009-0115-y
    """

    antithetic = False
    KK = 1  # K for series truncation.

    def set_mc_params(self, n_path=10000, dt=None, rn_seed=None, KK=1):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            rn_seed: random number seed
            KK: truncation index

        """
        self.n_path = int(n_path)
        self.dt = dt
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)
        self.KK = KK

    def gamma_lambda(self, dt):
        """
        gamma_n and lambda_n in Glasserman and Kim (2011)

        Args:
            dt: time step

        Returns:
            gamma, lambda
        """

        mrt2 = (self.mr * dt)**2
        vovn = self.vov**2 * dt

        n_2pi_2 = (np.arange(1, self.KK + 1) * 2 * np.pi)**2
        gamma_n = (mrt2 + n_2pi_2) / (2 * vovn * dt)
        lambda_n = 4 * n_2pi_2 / vovn / (mrt2 + n_2pi_2)

        return gamma_n, lambda_n

    @staticmethod
    def x1_mean_var(mrt, KK=0):

        # remainder (truncated) terms
        rem_mean = 2 / (np.pi**2 * KK)
        rem_var = 2 / (3 * np.pi**4 * KK**3)

        return rem_mean, rem_var

    @staticmethod
    def z_mean_var(mrt, KK=0):

        rem_mean = 1 / (np.pi**2 * KK)
        rem_var = 1 / (6 * np.pi**4 * KK**3)

        return rem_mean, rem_var

    def draw_x1(self, var_sum, dt):
        """
        Simulation of X1 using truncated Gamma expansion in Glasserman and Kim (2011)

        Parameters
        ----------
        var_t : an 1-d array with shape (n_paths,)
            final variance
        dt: float
            time-to-expiry

        Returns
        -------
         an 1-d array with shape (n_paths,), random variables X1
        """
        # For fixed k, theta, vov, texp, generate some parameters firstly

        mrt = self.mr * dt
        vovn = self.vov**2 * dt
        gamma_n, lambda_n = self.gamma_lambda(dt)

        # the following para will change with VO and VT
        Nn = self.rng.poisson(lam=var_sum * lambda_n[:, None])  # (KK, n_path)

        rv_exp_sum = self.rng.standard_gamma(shape=Nn)
        X1 = np.sum(rv_exp_sum / gamma_n[:, None], axis=0)

        rem_mean_X1, rem_var_X1 = self.x1_mean_var(mrt, self.KK)
        rem_mean_X1 *= dt
        rem_var_X1 *= vovn * dt**2

        rem_scale = rem_var_X1 / rem_mean_X1
        rem_shape = rem_mean_X1 / rem_scale * var_sum

        X1 += rem_scale * self.rng.standard_gamma(rem_shape)
        return X1

    def draw_X2_AW(self, mu_X2_0, sigma_square_X2_0, ncx_df, texp, num_rv):
        """
        Simulation of X2 or Z from its CDF based on Abate-Whitt algorithm from formula (4.1) in Glasserman and Kim (2011)

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

    def draw_eta(self, zz):
        """
        generate Bessel random variables from inverse of CDF, formula(2.4) in George and Dimitris (2010)

        Args:
            zz:  Bessel RV parameter (n, )

        Returns:
            eta (integer >= 0) values (n, )
        """

        iv_index = 0.5 * self.chi_dim() - 1
        p0 = np.power(0.5 * zz, iv_index) / (spsp.iv(iv_index, zz) * spsp.gamma(iv_index + 1))
        temp = np.arange(1, 16)[:, None]  # Bessel distribution has short tail, 30 maybe enough
        p = zz**2 / (4 * temp * (temp + iv_index))
        p = np.vstack((p0, p)).cumprod(axis=0).cumsum(axis=0)
        rv_uni = self.rv_uniform()
        eta = np.sum(p < rv_uni, axis=0).astype(np.uint32)

        return eta

    def draw_x2(self, ncx_df, dt, size):
        """
        Simulation of x2 (or Z) using truncated Gamma expansion in Glasserman and Kim (2011)
        Z is the special case with ncx_df = 4
        
        Args:
            ncx_df: ncx2 degree of freedom
            dt: time-to-expiry
            size: number of RVs to generate

        Returns:
            Random samples of x2 (or Z) with shape (n_path,)
        """

        mrt = self.mr * dt
        vovn = self.vov**2 * dt
        gamma_n, _ = self.gamma_lambda(dt)

        gamma_rv = self.rng.standard_gamma(ncx_df / 2, size=(self.KK, size))
        x2 = np.sum(gamma_rv / gamma_n[:, None], axis=0)

        # remainder (truncated) terms
        rem_mean, rem_var = self.z_mean_var(mrt, self.KK)
        rem_mean *= ncx_df/4 * vovn * dt
        rem_var *= ncx_df/4 * (vovn * dt)**2
        rem_scale = rem_var / rem_mean
        rem_shape = rem_mean / rem_scale

        x2 += rem_scale * self.rng.standard_gamma(rem_shape, size=size)
        return x2

    def cond_states(self, var_0, texp):

        phi, exp = self.phi_exp(texp)
        var_t = self.var_step_ncx2(var_0=var_0, dt=texp)
        zz = np.sqrt(var_0 * var_t) * phi

        # sample int_var(integrated variance): Gamma expansion / transform inversion
        # int_var = X1+X2+X3 from formula(2.7) in Glasserman and Kim (2011)

        # Simulation X1: truncated Gamma expansion
        X123 = self.draw_x1(var_0 + var_t, texp)
        X123 += self.draw_x2(self.chi_dim(), texp, size=self.n_path)
        eta = self.draw_eta(zz)

        ZZ = self.draw_x2(4, texp, size=eta.sum())

        total = 0
        for i in np.arange(eta.max()):
            eta_above_i = (eta > i)
            count = eta_above_i.sum()
            if count == 0:
                continue
            X123[eta_above_i] += ZZ[total:total+count]
            total += count

        assert eta.sum() == total
        var_mean = X123 / texp

        return var_t, var_mean


class HestonMcChoiKwok2023(HestonMcGlassermanKim2011):

    @staticmethod
    def x1_mean_var(mrt, KK=0):
        mrt_h = mrt / 2
        csch = 1 / np.sinh(mrt_h)
        coth = np.cosh(mrt_h) * csch

        x1_mean = (coth/mrt_h - csch**2) / 2
        x1_var = (coth / mrt_h**3 + csch**2 / mrt_h**2 - 2 * coth*csch**2 / mrt_h) / 8

        if KK > 0:
            n_2pi_2 = (np.arange(1, KK + 1) * 2 * np.pi)**2
            term = 8 * n_2pi_2 / (mrt**2 + n_2pi_2)**2
            x1_mean -= np.sum(term)
            x1_var -= np.sum(4 * term / (mrt**2 + n_2pi_2))

        return x1_mean, x1_var

    @staticmethod
    def z_mean_var(mrt, KK=0):
        mrt_h = mrt / 2
        csch = 1 / np.sinh(mrt_h)
        coth = np.cosh(mrt_h) * csch

        z_mean = (mrt_h * coth - 1) / (2 * mrt_h**2)
        z_var = (mrt_h * coth + mrt_h**2 * csch**2 - 2) / (8*mrt_h**4)

        if KK > 0:
            term = 1 / (mrt**2 + (np.arange(1, KK + 1) * 2 * np.pi)**2)
            z_mean -= 4 * np.sum(term)
            z_var -= 8 * np.sum(term**2)

        return z_mean, z_var

    def draw_x123(self, var_sum, eta_sum, dt):
        """
        Simulation of X1 + X2 + X3 using truncated Gamma expansion in Glasserman and Kim (2011)

        Args:
            var_sum: initial + final variance. (n_paths,)
            eta_sum: Bessel RV
            dt: time step

        Returns:
            X1 + X2 + X3  (n_paths,)
        """
        mrt = self.mr * dt
        vovn = self.vov**2 * dt
        gamma_n, lambda_n = self.gamma_lambda(dt)

        Nn = self.rng.poisson(lam=var_sum * lambda_n[:, None])
        x123 = np.sum(self.rng.standard_gamma(shape=Nn + eta_sum*2) / gamma_n[:, None], axis=0)

        # The approximated mean and variance of the truncated terms
        #rem_mean_x1 = 2 * dt / (np.pi**2 * self.KK) * var_sum
        #rem_var_x1 = 2 * self.vov**2 * dt**3 / (3 * np.pi**4 * self.KK**3) * var_sum
        #rem_mean_x23 = (self.vov * dt)**2 / (4 * np.pi**2 * self.KK) * ncx_df
        #rem_var_x23 = (self.vov * dt)**4 / (24 * np.pi**4 * self.KK**3) * ncx_df

        rem_mean_x1, rem_var_x1 = self.x1_mean_var(mrt, self.KK)
        rem_mean_x1 *= var_sum * dt
        rem_var_x1 *= var_sum * vovn * dt**2

        rem_mean_x23, rem_var_x23 = self.z_mean_var(mrt, self.KK)
        rem_mean_x23 *= eta_sum * vovn * dt
        rem_var_x23 *= eta_sum * vovn**2 * dt**2

        rem_scale = (rem_var_x1 + rem_var_x23) / (rem_mean_x1 + rem_mean_x23)
        rem_shape = (rem_mean_x1 + rem_mean_x23) / rem_scale
        x123 += rem_scale * self.rng.standard_gamma(rem_shape)

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
        var_mean = self.draw_x123(var_sum, eta_sum, dt[0]) / texp

        return var_t, var_mean

