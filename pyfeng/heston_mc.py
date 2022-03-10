import numpy as np
import scipy.stats as spst
import scipy.special as spsp
import scipy.integrate as spint
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
        var_t = cof * np.random.noncentral_chisquare(chi_dim, chi_lambda, self.n_path)
        return var_t

    def ch_f(self, aa, texp, var_final):
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

        gamma = np.sqrt(self.mr ** 2 - 2 * vov2 * aa * 1j)
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

        def ch_f(aa):
            return self.ch_f(aa, texp, var_final)

        moment_1st = derivative(ch_f, 0, n=1, dx=1e-5).imag
        moment_2st = -derivative(ch_f, 0, n=2, dx=1e-5).real

        if self.dist == 0:
            scale_ig = moment_1st ** 3 / (moment_2st - moment_1st ** 2)
            miu_ig = moment_1st / scale_ig
            int_var_std = spst.invgauss.rvs(miu_ig, scale=scale_ig) / texp
        elif self.dist == 1:
            scale_ln = np.sqrt(np.log(moment_2st) - 2 * np.log(moment_1st))
            miu_ln = np.log(moment_1st) - 0.5 * scale_ln ** 2
            int_var_std = np.random.lognormal(miu_ln, scale_ln) / texp
        else:
            raise ValueError(f"Incorrect distribution.")

        ### Common Part
        int_var_dw = ((var_final - var0) - self.mr * texp * (self.theta - int_var_std)) / self.vov
        spot_cond = np.exp(self.rho * (int_var_dw - 0.5 * self.rho * int_var_std * texp))
        sigma_cond = rhoc * np.sqrt(int_var_std / var0)  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond