import numpy as np
import scipy.stats as spst
import scipy.special as spsp
import scipy.integrate as spint
import scipy.optimize as spop
import math
from scipy import interpolate
from scipy.misc import derivative
from . import sv_abc as sv


class HestonCondMcQE(sv.SvABC, sv.CondMcBsmABC):
    """
    Heston model with conditional Monte-Carlo simulation

    Conditional MC for Heston model based on QE discretization scheme by Andersen (2008).

    Underlying price follows a geometric Brownian motion, and variance of the price follows a CIR process.

    References:
        - Andersen L (2008) Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11:1–42. https://doi.org/10.21314/JCF.2008.189

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.array([60, 100, 140])
        >>> spot = 100
        >>> sigma, vov, mr, rho, texp = 0.04, 1, 0.5, -0.9, 10
        >>> m = pf.HestonCondMcQE(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.set_mc_params(n_path=1e5, dt=1/8, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.330, 13.085, 0.296
        array([44.31943535, 13.09371251,  0.29580431])
    """
    var_process = True
    psi_c = 1.5  # parameter used by the Andersen QE scheme

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

    def vol_paths(self, tobs):
        var0 = self.sigma
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        var_path = np.full((n_dt + 1, self.n_path), var0)  # variance series: V0, V1,...,VT
        var_t = np.full(self.n_path, var0)

        if self.scheme < 2:
            dB_t = self._bm_incr(tobs, cum=False)  # B_t (0 <= s <= 1)
            for i in range(n_dt):
            # Euler (or Milstein) scheme
                var_t += self.mr * (self.theta - var_t) * dt[i] + np.sqrt(var_t) * self.vov * dB_t[i, :]

                # Extra-term for Milstein scheme
                if self.scheme == 1:
                    var_t += 0.25 * self.vov ** 2 * (dB_t[i, :] ** 2 - dt[i])

                var_t[var_t < 0] = 0  # variance should be larger than zero
                var_path[i + 1, :] = var_t

        elif self.scheme == 2:
            # obtain the standard normals
            zz = self._bm_incr(tobs=np.arange(1, n_dt+0.1), cum=False)

            for i in range(n_dt):
                # compute m, s_square, psi given vt(i)
                expo = np.exp(-self.mr * dt[i])
                m = self.theta + (var_t - self.theta) * expo
                s2 = var_t * expo + self.theta * (1 - expo)/2
                s2 *= self.vov ** 2 * (1 - expo) / self.mr
                psi = s2 / m**2

                # compute vt(i+1) given psi
                # psi < psi_c
                idx_below = (psi <= self.psi_c)
                ins = 2 / psi[idx_below]
                b2 = (ins - 1) + np.sqrt(ins * (ins - 1))
                a = m[idx_below] / (1 + b2)
                var_t[idx_below] = a * (np.sqrt(b2) + zz[i, idx_below]) ** 2

                # psi_c < psi
                uu = spst.norm.cdf(zz[i, :])
                pp = (psi - 1) / (psi + 1)
                beta = (1 - pp) / m

                idx_above = (uu <= pp) & ~idx_below
                var_t[idx_above] = 0.0
                idx_above = (uu > pp) & ~idx_below
                var_t[idx_above] = (np.log((1 - pp)/(1 - uu))/beta)[idx_above]

                var_path[i + 1, :] = var_t

        return var_path

    def cond_spot_sigma(self, texp):

        var0 = self.sigma  # inivial variance
        rhoc = np.sqrt(1.0 - self.rho ** 2)
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        var_paths = self.vol_paths(tobs)
        var_final = var_paths[-1, :]
        int_var_std = spint.simps(var_paths, dx=1, axis=0) / n_dt

        int_var_dw = ((var_final - var0) - self.mr * texp * (self.theta - int_var_std)) / self.vov
        spot_cond = np.exp(self.rho * (int_var_dw - 0.5 * self.rho * int_var_std * texp))
        sigma_cond = rhoc * np.sqrt(int_var_std / var0)  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond


class HestonMcAe(sv.SvABC, sv.CondMcBsmABC):
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

    def vol_paths(self, tobs):
        return np.ones(size=(len(tobs), self.n_path))

    def chi_dim(self):
        """
        Noncentral Chi-square (NCX) distribution's degree of freedom

        Returns:
            degree of freedom (scalar)
        """
        chi_dim = 4 * self.theta * self.mr / self.vov ** 2
        return chi_dim

    def chi_lambda(self, texp):
        """
        Noncentral Chi-square (NCX) distribution's noncentrality parameter

        Returns:
            noncentrality parameter (scalar)
        """
        chi_lambda = 4 * self.sigma * self.mr / self.vov ** 2
        chi_lambda /= np.exp(self.mr * texp) + 1
        return chi_lambda

    def var_final(self, texp):
        """
        Draw final variance from NCX distribution

        Args:
            texp: time to expiry

        Returns:
            final variance (at t=T)
        """
        chi_dim = self.chi_dim()
        chi_lambda = self.chi_lambda(texp)

        cof = self.vov ** 2 * (1 - np.exp(-self.mr * texp)) / (4 * self.mr)
        var_t = cof * self.rng.noncentral_chisquare(df=chi_dim, nonc=chi_lambda, size=self.n_path)
        return var_t

    def mgf(self, aa, texp, var_final):
        """
            Characteristic function

        Args:
            aa: dummy variable in the transformation
            texp: time to expiry
            var_final: volatility at time T

        Returns:
            ch_f: characteristic function of the distribution of integral sigma_t
        """

        var_0 = self.sigma
        vov2 = self.vov ** 2
        iv_index = 0.5 * self.chi_dim() - 1

        gamma = np.sqrt(self.mr ** 2 - 2 * vov2 * aa)
        #decay = np.exp(-self.mr * texp)
        #decay_gamma = np.exp(-gamma * texp)

        var_mean = np.sqrt(var_0 * var_final)
        phi_mr = 2 * self.mr / vov2 / np.sinh(self.mr * texp / 2)
        cosh_mr = np.cosh(self.mr * texp / 2)

        phi_gamma = 2 * gamma / vov2 / np.sinh(gamma * texp / 2)
        cosh_gamma = np.cosh(gamma * texp / 2)

        #part1 = gamma * np.exp(-0.5 * (gamma * texp - self.mr * texp)) * (1 - decay) / (self.mr * (1 - decay_gamma))
        part1 = phi_gamma / phi_mr

        #part2 = np.exp((var_0 + var_final) / vov2
        #    * (self.mr * (1 + decay) / (1 - decay) - gamma * (1 + decay_gamma) / (1 - decay_gamma)))
        part2 = np.exp((var_0 + var_final)*(cosh_mr*phi_mr - cosh_gamma*phi_gamma)/2)

        part3 = spsp.iv(iv_index, var_mean * phi_gamma) / spsp.iv(iv_index, var_mean * phi_mr)

        ch_f = part1 * part2 * part3
        return ch_f


    def cond_spot_sigma(self, texp):

        var0 = self.sigma  # inivial variance
        rhoc = np.sqrt(1.0 - self.rho ** 2)

        var_final = self.var_final(texp)

        # conditional MGF function
        def mgf_cond(aa):
            return self.mgf(aa, texp, var_final)

        # Get the first 2 moments
        m1 = derivative(mgf_cond, 0, n=1, dx=1e-5)
        m2 = derivative(mgf_cond, 0, n=2, dx=1e-5)

        if self.dist == 0:
            # mu and lambda defined in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
            # RNG.wald takes the same parameters
            mu = m1
            lam = m1 ** 3 / (m2 - m1 ** 2)
            int_var_std = self.rng.wald(mean=mu, scale=lam) / texp
        elif self.dist == 1:
            scale_ln = np.sqrt(np.log(m2) - 2 * np.log(m1))
            miu_ln = np.log(m1) - 0.5 * scale_ln ** 2
            int_var_std = self.rng.lognormal(mean=miu_ln, sigma=scale_ln) / texp
        else:
            raise ValueError(f"Incorrect distribution.")

        ### Common Part
        int_var_dw = ((var_final - var0) - self.mr * texp * (self.theta - int_var_std)) / self.vov
        spot_cond = np.exp(self.rho * (int_var_dw - 0.5 * self.rho * int_var_std * texp))
        sigma_cond = rhoc * np.sqrt(int_var_std / var0)  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond


class HestonMcExactGK(HestonMcAe):

    KK = 1

    def set_mc_params(self, n_path=10000, rn_seed=None, antithetic=True, KK=1):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            rn_seed: random number seed
            antithetic: antithetic
            KK:

        References:
            -
        """
        self.n_path = int(n_path)
        self.rn_seed = rn_seed
        self.antithetic = antithetic
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)
        self.KK = KK

    def cond_spot_sigma(self, texp):

        var0 = self.sigma  # inivial variance
        rhoc = np.sqrt(1.0 - self.rho ** 2)

        ncx_df = self.chi_dim()
        var_final = self.var_final(texp)

        # sample int_var(integrated variance): Gamma expansion / transform inversion
        # int_var = X1+X2+X3 from formula(2.7) in Glasserman and Kim (2011)

        # Simulation X1: truncated Gamma expansion
        X1 = self.generate_X1_gamma_expansion(var_final, texp)

        # Simulation X2: transform inversion
        coth = 1 / np.tanh(self.mr * texp * 0.5)
        csch = 1 / np.sinh(self.mr * texp * 0.5)
        mu_X2_0 = self.vov ** 2 * (-2 + self.mr * texp * coth) / (4 * self.mr ** 2)
        sigma_square_X2_0 = self.vov ** 4 * (-8 + 2 * self.mr * texp * coth + (self.mr * texp * csch) ** 2) / (
                    8 * self.mr ** 4)
        X2 = self.generate_X2_and_Z_AW(mu_X2_0, sigma_square_X2_0, ncx_df, texp, self.n_path)
        # X2 = self.generate_X2_and_Z_gamma_expansion(ncx_df, texp, self.n_path)

        # Simulation X3: X3=sum(Z, eta), Z is a special case of X2 with ncx_df=4
        Z = self.generate_X2_and_Z_AW(mu_X2_0, sigma_square_X2_0, 4, texp, self.n_path * 10)
        # Z = self.generate_X2_and_Z_gamma_expansion(4, texp, self.n_path*10)

        v = 0.5 * ncx_df - 1
        z = 2 * self.mr * self.sigma * np.sqrt(var_final) * csch / self.vov ** 2
        eta = self.generate_eta(v, z)

        X3 = np.zeros(len(eta))
        for ii in range(len(eta)):
            X3[ii] = np.sum(Z[np.random.randint(0, len(Z), int(eta[ii]))])

        int_var_std = (X1 + X2 + X3) / texp

        ### Common Part
        int_var_dw = ((var_final - var0) - self.mr * texp * (self.theta - int_var_std)) / self.vov
        spot_cond = np.exp(self.rho * (int_var_dw - 0.5 * self.rho * int_var_std * texp))
        sigma_cond = rhoc * np.sqrt(int_var_std / var0)  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond

    def generate_X1_gamma_expansion(self, VT, texp):
        """
        Simulation of X1 using truncated Gamma expansion in Glasserman and Kim (2011)

        Parameters
        ----------
        VT : an 1-d array with shape (n_paths,)
            final variance
        texp: float
            time-to-expiry

        Returns
        -------
         an 1-d array with shape (n_paths,), random variables X1
        """
        var_0 = self.sigma
        # For fixed k, theta, vov, texp, generate some parameters firstly
        range_K = np.arange(1, self.KK + 1)
        temp = 4 * np.pi ** 2 * range_K ** 2
        gamma_n = (self.mr ** 2 * texp ** 2 + temp) / (2 * self.vov ** 2 * texp ** 2)
        lambda_n = 4 * temp / (self.vov ** 2 * texp * (self.mr ** 2 * texp ** 2 + temp))

        E_X1_K_0 = 2 * texp / (np.pi ** 2 * range_K)
        Var_X1_K_0 = 2 * self.vov ** 2 * texp ** 3 / (3 * np.pi ** 4 * range_K ** 3)

        # the following para will change with VO and VT
        Nn_mean = ((var_0 + VT)[:, None] * lambda_n[None, :])  # every row K numbers (one path)
        Nn = np.random.poisson(lam=Nn_mean).flatten()
        rv_exp_sum = np.zeros(len(Nn))
        for ii in range(len(Nn)):
            rv_exp_sum[ii] = np.sum(np.random.exponential(scale=1, size=Nn[ii]))
        rv_exp_sum = rv_exp_sum.reshape(len(VT), len(lambda_n))
        X1_main = np.sum((rv_exp_sum / gamma_n), axis=1)

        gamma_mean = (var_0 + VT) * E_X1_K_0[-1]
        gamma_var = (var_0 + VT) * Var_X1_K_0[-1]
        beta = gamma_mean / gamma_var
        alpha = gamma_mean * beta
        X1_truncation = np.random.gamma(alpha, 1 / beta)
        # X1_truncation = np.random.normal(loc=gamma_mean, scale=np.sqrt(gamma_var))
        X1 = X1_main + X1_truncation

        return X1


    def generate_X2_and_Z_AW(self, mu_X2_0, sigma_square_X2_0, ncx_df, texp, num_rv):
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
        L = lambda x: np.sqrt(2 * self.vov ** 2 * x + self.mr ** 2)
        fha_2 = lambda x: (L(x) / self.mr * (np.sinh(0.5 * self.mr * texp) / np.sinh(0.5 * L(x) * texp))) ** (
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
        rv_uni = np.random.uniform(size=num_rv)
        xi = np.insert(xi, 0, 0.)
        F_X2 = np.insert(F_X2, 0, 0.)
        F_X2_inv = interpolate.interp1d(F_X2, xi, kind="slinear")
        X2 = F_X2_inv(rv_uni)

        return X2

    def generate_eta(self, v, z):
        """
        generate Bessel random variables from inverse of CDF, formula(2.4) in George and Dimitris (2010)

        Parameters
        ----------
        v:  float
            parameter in Bessel distribution
        z: an 1-d array with shape (n_paths,)
            parameter in Bessel distribution

        Returns
        -------
         an 1-d array with shape (n_paths,), Bessel random variables eta
        """

        p0 = np.power(0.5 * z, v) / (spsp.iv(v, z) * math.gamma(v + 1))
        temp = np.arange(1, 31)[:, None]  # Bessel distribution has sort tail, 30 maybe enough
        p = z ** 2 / (4 * temp * (temp + v))
        p = np.vstack((p0, p)).cumprod(axis=0).cumsum(axis=0)
        rv_uni = np.random.uniform(size=len(z))
        eta = np.sum(p < rv_uni, axis=0)

        return eta

    def generate_X2_and_Z_gamma_expansion(self, ncx_df, texp, num_rv):
        """
        Simulation of X2 or Z using truncated Gamma expansion in Glasserman and Kim (2011)

        Parameters
        ----------
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

        range_K = np.arange(1, self.KK + 1)
        temp = 4 * np.pi ** 2 * range_K ** 2
        gamma_n = (self.mr ** 2 * texp ** 2 + temp) / (2 * self.vov ** 2 * texp ** 2)

        rv_gamma = np.random.gamma(0.5 * ncx_df, 1, size=(num_rv, self.KK))
        X2_main = np.sum(rv_gamma / gamma_n, axis=1)

        gamma_mean = ncx_df * (self.vov * texp) ** 2 / (4 * np.pi ** 2 * self.KK)
        gamma_var = ncx_df * (self.vov * texp) ** 4 / (24 * np.pi ** 4 * self.KK ** 3)
        beta = gamma_mean / gamma_var
        alpha = gamma_mean * beta
        X2_truncation = np.random.gamma(alpha, 1 / beta, size=num_rv)
        X2 = X2_main + X2_truncation

        return X2