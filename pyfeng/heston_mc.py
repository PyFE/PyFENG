import abc
import warnings
import numpy as np
import scipy.stats as spst
import scipy.interpolate as spinterp
from scipy import special as spsp
import functools
from .sv_abc import CondMcBsmABC
from .heston import HestonABC
from .mgf2mom import Mgf2Mom

#### Use of RN generation spawn:
# 0: simulation of variance (gamma/ncx2/normal)
# 1: eta or poisson (mu) for variance
# 2: gamma series and truncation (gamma/IG) for integrated variance (avgvar)
# 3: poisson in gamma series (Glasserman-Kim, Choi-Kwok)
# 4: not used
# 5: asset return


class HestonMcABC(HestonABC, CondMcBsmABC):
    scheme = None
    correct_fwd = False
    correct_martingale = False

    def draw_from_mv(self, mean, var_scaled, dist):
        """
        Draw RNs from distributions with mean and var_scaled matched
        Args:
            mean: mean
            var_scaled: scaled variance (= variance / mean^2)
            dist: distribution. 'ig' for IG, 'ga' for Gamma, 'ln' for log-normal

        Returns:
            RNs with size of mean/var_scaled
        """
        if dist.lower() == 'ig':
            # mu and lambda defined in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
            # RNG.wald takes the same parameters
            avgvar = self.rng_spawn[2].wald(mean=mean, scale=mean / var_scaled)
        elif dist.lower() == 'ga':
            avgvar = var_scaled * mean * self.rng_spawn[2].standard_gamma(shape=1 / var_scaled)
        elif dist.lower() == 'ln':
            scale = np.sqrt(np.log1p(var_scaled))
            avgvar = mean * np.exp(scale * (self.rv_normal(spawn=2) - scale / 2))
        else:
            raise ValueError(f"Incorrect distribution: {dist}.")

        return avgvar

    @abc.abstractmethod
    def cond_states_step(self, dt, var_0):
        """
        Final variance after dt and average variance over (0, dt) given var_0.
        `var_0` should be an array of (self.n_path, )

        Args:
            dt: time step
            var_0: initial variance

        Returns:
            (variance after dt, average variance during dt)
        """
        raise NotImplementedError

    def cond_spot_sigma(self, texp, var_0):
        tobs = self.tobs(texp)
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        var_t = np.full(self.n_path, var_0)
        avgvar = np.zeros(self.n_path)
        m_corr = np.zeros(self.n_path)  # martingale correction

        for i in range(n_dt):
            var_t, avgvar_inc, extra = self.cond_states_step(dt[i], var_t)
            avgvar += avgvar_inc * dt[i]

            if self.correct_martingale:
                pois_avgvar_v = extra.get('pois_avgvar_v', None)
                qe_m_corr = extra.get('qe_m_corr', None)

                if pois_avgvar_v is not None:  # martingale correction in POIS-TD
                    m_corr += 0.5*(self.rho*(1/self.vov*self.mr - 0.5*self.rho)*dt[i])**2 * pois_avgvar_v
                elif qe_m_corr is not None:  # martingale correction in QE
                    m_corr += qe_m_corr

        avgvar /= texp

        avgvar_m_anal, avgvar_v_anal, _ = self.avgvar_mv(texp)  # analytic mean and variance of avgvar
        self.result = {**self.result,
                       'avgvar mean': avgvar_m_anal,
                       'avgvar mean error': avgvar.mean()/avgvar_m_anal - 1,
                       'avgvar var': avgvar_v_anal,
                       'avgvar var error': np.square(avgvar - avgvar_m_anal).mean()/avgvar_v_anal - 1
                       }

        spot_cond = ((var_t - var_0) + self.mr * texp * (avgvar - self.theta)) / self.vov - 0.5 * self.rho * texp * avgvar
        spot_cond *= self.rho
        spot_cond += m_corr
        np.exp(spot_cond, out=spot_cond)

        sigma_cond = np.sqrt((1.0 - self.rho**2) / var_0 * avgvar)  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond

    def strike_var_swap_analytic(self, texp, dt=None):
        if dt is None:
            dt = self.dt
        rv = super().strike_var_swap_analytic(texp, dt)
        return rv

    def cond_log_return_var(self, dt, var_0, var_t, avgvar):

        mean_ln = self.rho / self.vov * ((var_t - var_0) + self.mr * dt * (avgvar - self.theta)) \
                  + (self.intr - self.divr - 0.5 * avgvar) * dt
        sigma_ln2 = (1.0 - self.rho**2) * dt * avgvar
        return mean_ln**2 + sigma_ln2

    def draw_log_return(self, dt, var_0, var_t, avgvar):
        """
        Samples log return, log(S_t/S_0)

        Args:
            dt: time step
            var_0: initial variance
            var_t: final variance
            avgvar: average variance

        Returns:
            log return
        """
        ln_m = self.rho/self.vov * ((var_t - var_0) + self.mr * dt * (avgvar - self.theta)) \
            + (self.intr - self.divr - 0.5 * avgvar) * dt
        ln_sig = np.sqrt((1.0 - self.rho**2) * dt * avgvar)
        zn = self.rv_normal(spawn=5)
        return ln_m + ln_sig * zn

    def return_var_realized(self, texp, cond=False):
        """
        Annualized realized return variance

        Args:
            texp: time to expiry
            cond: use conditional expectation without simulating price

        Returns:

        """
        tobs = self.tobs(texp)
        n_dt = len(tobs)
        dt = np.diff(tobs, prepend=0)

        var_r = np.zeros(self.n_path)
        var_0 = np.full(self.n_path, self.sigma)

        tmp = self.rho/self.vov*self.mr - 0.5

        for i in range(n_dt):
            var_t, avgvar_inc, extra = self.cond_states_step(dt[i], var_0)

            if self.correct_martingale:
                pois_avgvar_v = extra.get('pois_avgvar_v', None)
                if pois_avgvar_v is not None:  # missing variance:
                    var_r += (tmp*dt[i])**2 * pois_avgvar_v
                qe_m_corr = extra.get('qe_m_corr', 0.0)
            else:
                qe_m_corr = 0.0

            if cond:
                var_r += self.cond_log_return_var(dt[i], var_0, var_t, avgvar_inc)
            else:
                var_r += (self.draw_log_return(dt[i], var_0, var_t, avgvar_inc) + qe_m_corr) ** 2

            var_0 = var_t

        return var_r / texp  # annualized

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
        Mean and var_scaled of the truncated terms of (X1^*/dt) in Glasserman & Kim (2011).
        (v_0 + v_t) need to be multiplied to mean afterwards; var_scaled = var/mean^2 is per-unit.

        Args:
            dt: time step
            kk: number of gamma expansion terms

        References:
            - p 281-282 in Glasserman & Kim (2011)
            - Proposition 3.1 in Tse & Wan (2013)

        Returns:
            mean, var_scaled (= variance / mean^2)
        """
        mrt_h = self.mr * dt / 2
        vov2dt = self.vov**2 * dt
        a2 = mrt_h**2

        # mean is independent of vov2dt; var = vov2dt * vf → var_scaled = vov2dt * vf / mean**2
        # Taylor for small mrt_h avoids cancellation in (coth/a - csch²) and (coth/a³ + csch²/a² - 2*coth*csch²/a)
        # mean = 1/3 - 2a²/45 + 2a⁴/315 - ...
        # vf   = 1/45 - 2a²/315 + 2a⁴/1575 - ...
        if np.all(np.abs(mrt_h) < 0.1):
            mean = 1/3 + a2*(-2/45 + a2*2/315)
            vf   = 1/45 + a2*(-2/315 + a2*2/1575)
        else:
            csch = 1 / np.sinh(mrt_h)
            coth = np.cosh(mrt_h) * csch
            mean = (coth/mrt_h - csch**2) / 2
            vf = (coth / mrt_h**3 + csch**2 / mrt_h**2 - 2 * coth*csch**2 / mrt_h) / 8

        if kk > 0:
            gamma_n, lambda_n = self.gamma_lambda(dt, kk=kk)
            mean -= np.sum(lambda_n/gamma_n)             # lambda_n/gamma_n independent of vov2dt
            vf -= 2*np.sum(lambda_n/gamma_n**2) / vov2dt  # lambda_n/gamma_n**2 ∝ vov2dt, so ratio is pure

        return mean, vov2dt * vf / mean**2

    def x2star_avgvar_mv(self, dt, kk=0):
        """
        Mean and var_scaled of the truncated terms of X2/dt (with shape=1 or delta=2) in Glasserman & Kim (2011)

            X2 = sum_{n=1}^kk standard_gamma(1) / gamma_n

        Args:
            dt: time step
            kk: number of gamma expansion terms

        References:
            - p 284 in Glasserman & Kim (2011)
            - Proposition 3.1 in Tse & Wan (2013)

        Returns:
            mean, var_scaled (= variance / mean^2)
        """
        mrt_h = self.mr * dt / 2
        vov2dt = self.vov**2 * dt
        a2 = mrt_h**2

        # mean = vov2dt * mf, var = vov2dt**2 * vf → var_scaled = vf/mf**2 (vov2dt cancels)
        # Taylor for small mrt_h avoids cancellation in (a*coth - 1) and (a*coth + a²*csch² - 2)
        # mf = 1/12 - a²/180 + a⁴/1890 - ...
        # vf = 1/360 - a²/1890 + ...
        if np.all(np.abs(mrt_h) < 0.1):
            mf = 1/12 + a2*(-1/180 + a2/1890)
            vf = 1/360 - a2/1890
        else:
            csch = 1 / np.sinh(mrt_h)
            coth = np.cosh(mrt_h) * csch
            mf = (mrt_h * coth - 1) / (4 * a2)
            vf = (mrt_h * coth + a2 * csch**2 - 2) / (16 * a2**2)

        if kk > 0:
            gamma_n, _ = self.gamma_lambda(dt, kk)
            mf -= np.sum(1/gamma_n) / vov2dt       # 1/gamma_n ∝ vov2dt, so ratio is pure
            vf -= np.sum(1/gamma_n**2) / vov2dt**2  # 1/gamma_n**2 ∝ vov2dt**2, so ratio is pure

        return vov2dt * mf, vf / mf**2

    def eta_mv(self, dt, var_0, var_t):
        """
        The mean and var_scaled of eta RV.

        Args:
            dt: time step
            var_0: initial variance
            var_t: final variance

        Returns:
            mean, var_scaled (= variance / mean^2)

        References:
            Proposition 3.1 in Tse & Wan (2013)
        """
        phi, exp = self.cir.phi_exp(dt)
        zz = np.sqrt(var_0 * var_t) * phi

        iv_index = 0.5 * self.cir.chi_dim() - 1
        # ive(): exponentially scaled iv function
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html
        iv0 = spsp.ive(iv_index, zz)
        iv1 = spsp.ive(iv_index + 1, zz)
        iv2 = spsp.ive(iv_index + 2, zz)

        r1 = iv1 / iv0
        mean = (zz/2) * r1
        # var = (zz/2)**2*(iv2/iv0) + mean*(1-mean); divided by mean**2:
        # = (iv2/iv0)/r1**2 + 1/mean - 1
        var_scaled = (iv2/iv0) / r1**2 + 1/mean - 1

        return mean, var_scaled

    def cond_avgvar_mv(self, dt, var_0, var_t, pois=None, kk=0):
        """
        Mean and var_scaled of the average variance conditional on initial var, final var, and eta

        Args:
            dt: time step
            var_0: initial variance
            var_t: final variance
            pois: Poisson RV. If None, assume to be an eta RV.
            kk: number of gamma expansion terms

        Returns:
            mean, var_scaled (= variance / mean^2)

        References:
            Proposition 3.1 in Tse & Wan (2013)
        """

        if pois is None:
            m_eta, vs_eta = self.eta_mv(dt, var_0, var_t)
        else:
            m_eta, vs_eta = pois, 0.0

        m_x, vs_x = self.x1star_avgvar_mv(dt, kk=kk)
        fac_x = var_0 + var_t

        m_z, vs_z = self.x2star_avgvar_mv(dt, kk=kk)
        fac_z = 2*m_eta + self.cir.chi_dim()/2

        m_total = m_x * fac_x + m_z * fac_z
        # variances scale linearly with the factor; reconstruct from var_scaled * m^2 * factor
        v_total = vs_x * m_x**2 * fac_x + vs_z * m_z**2 * fac_z
        if pois is None:
            v_total += vs_eta * m_eta**2 * (2*m_z)**2

        return m_total, v_total / m_total**2


class HestonMcGlassermanKim2011(HestonMcABC):
    """
    Exact simulation using the gamma series based on Glasserman & Kim (2011)

    References:
        - Glasserman P, Kim K-K (2011) Gamma expansion of the Heston stochastic volatility model. Finance Stoch 15:267–296. https://doi.org/10.1007/s00780-009-0115-y

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.array([60, 70, 100, 140])
        >>> sigma, vov, mr, rho, texp, spot = 0.04, 1, 0.5, -0.9, 10, 100
        >>> m = pf.HestonMcGlassermanKim2011(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.set_num_params(n_path=1e5, kk=4, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.32997507, 35.8497697, 13.08467014, 0.29577444
        array([44.35153812, 35.86029054, 13.17026256,  0.29550527])

    """

    dist = 'ga'  # distribution for the truncated series
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

    def cond_avgvar_mgf(self, aa, dt, var_0, var_t):
        """
        MGF of the average variance conditional on the initial and final variance.

        Args:
            aa: dummy variable in the transformation
            dt: time step
            var_0: initial variance
            var_t: final variance

        Returns:
            Conditional MGF at dummy variable aa
        """

        vov2dt = self.vov**2 * dt
        mrt = self.mr * dt
        iv_index = 0.5 * self.cir.chi_dim() - 1

        # Note that dt term is additionally multiplied to aa
        # to make it Laplace transform of average variance, not integrated variance
        gamma = np.sqrt(mrt**2 - 2 * vov2dt * aa)

        var_mean = np.sqrt(var_0 * var_t)
        phi_mr, _ = self.cir.phi_exp(dt)
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

    def cond_avgvar_mv_numeric(self, dt, var_0, var_t):
        """
        Mean and var_scaled of the average variance conditional on initial var, final var.
        Computed from the first two cumulants of the conditional MGF.

        Args:
            dt: time step
            var_0: initial variance
            var_t: final variance

        Returns:
            mean, var_scaled (= variance / mean^2)

        See Also:
            cond_avgvar_mv
        """
        cum = Mgf2Mom(lambda aa: self.cond_avgvar_mgf(np.asarray(aa, dtype=complex), dt, var_0, var_t)).cumulants(2)
        return cum[0], cum[1] / cum[0]**2

    def x1star_avgvar_mv_asymp(self, dt, kk=0):
        """
        Asymptotic mean and var_scaled of the truncated terms of X1/dt.
        (v_0 + v_t) need to be multiplied to mean afterwards.
        This is NOT used for pricing, but for verification purpose.

        Args:
            dt: time step
            kk: number of gamma expansion terms

        References:
            - Lemma 3.1 in Glasserman & Kim (2011)

        Returns:
            mean, var_scaled (= variance / mean^2)
        """

        vov2dt = self.vov**2 * dt
        trunc_mean = 2 / (np.pi**2 * kk)
        # var = vov2dt * 2/(3*pi^4*kk^3); mean = 2/(pi^2*kk) → var_scaled = vov2dt/(6*kk)
        return trunc_mean, vov2dt / (6 * kk)

    def x2star_avgvar_mv_asymp(self, dt, kk=0):
        """
        Asymptotic mean and var_scaled of the truncated terms of X2/dt (with shape=1 or delta=2).
        This is NOT used for pricing, but for verification purpose.

        Args:
            dt: time step
            kk: number of gamma expansion terms

        References:
            - Lemma 3.1 in Glasserman & Kim (2011)

        Returns:
            mean, var_scaled (= variance / mean^2)
        """

        vov2dt = self.vov**2 * dt
        trunc_mean = vov2dt / (2 * np.pi**2 * kk)
        # var = vov2dt**2/(12*pi^4*kk^3); mean = vov2dt/(2*pi^2*kk) → var_scaled = 1/(3*kk)
        return trunc_mean, 1 / (3 * kk)

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
        phi_mr, _ = self.cir.phi_exp(dt)

        ### mgf without considering branchcut
        # exph_gamma = np.exp(-gamma/2)
        # phi_gamma = 4 * gamma / vov2dt * exph_gamma / (1 - exph_gamma**2)
        # mgf = np.power(phi_gamma / phi_mr, shape)

        tmp = (4 * gamma / vov2dt) / (1 - np.exp(-gamma))
        mgf = np.exp(shape*(np.log(tmp/phi_mr) - gamma/2))

        return mgf

    def x2star_avgvar_mv_numeric(self, dt):
        """
        Mean and var_scaled of X2/dt (with shape=1 or delta=2) numerically computed from
        the MGF (Laplace transform) of X2/dt.

        Args:
            dt: time step

        Returns:
            mean, var_scaled (= variance / mean^2)
        """
        cum = Mgf2Mom(lambda aa: self.x2_avgvar_mgf(np.asarray(aa, dtype=complex), dt, 1)).cumulants(2)
        return cum[0], cum[1] / cum[0]**2

    def draw_x1(self, dt, var_0, var_t):
        """
        Samples of x1/dt using truncated Gamma expansion in Glasserman & Kim (2011)

        Args:
            var_0: initial variance
            var_t: final variance
            dt: time step

        Returns:
            x1/dt
        """
        gamma_n, lambda_n = self.gamma_lambda(dt, self.kk)
        pois = self.rng_spawn[3].poisson(lam=(var_0 + var_t) * lambda_n[:, None])  # (kk, n_path)

        rv_exp_sum = self.rng_spawn[2].standard_gamma(shape=pois)
        x1 = np.sum(rv_exp_sum / gamma_n[:, None], axis=0)

        trunc_m, trunc_vs = self.x1star_avgvar_mv(dt, kk=self.kk)
        # we could call self.draw_from_mv, but we don't to make the code simple.
        x1 += trunc_vs * trunc_m * self.rng_spawn[2].standard_gamma((var_0 + var_t) / trunc_vs)
        return x1

    def x2_cdf_points_aw(self, dt, shape):
        """
        Simulation of X2 or Z from its CDF based on Abate-Whitt algorithm from formula (4.1) in Glasserman & Kim (2011)
        Used if self.tabulate_x2_z is True.

        Args:
            shape: gamma shape parameter. delta/2 for X2. 2 for Z.
            dt: time step

        Returns: (x points, cdf values)
        """

        ww, M = 0.01, 200

        mean_x2, vs_x2 = self.x2star_avgvar_mv(dt, kk=0)
        mean_x2 *= shape

        mu_e = mean_x2 * (1 + 12 * np.sqrt(vs_x2 / shape))

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
    def x2_icdf_interp(self, dt, shape, *args, **kwargs):
        xx, cdf = self.x2_cdf_points_aw(dt, shape)
        xx = np.insert(xx, 0, 0)
        cdf = np.insert(cdf, 0, 0)
        rv = spinterp.interp1d(cdf, xx, kind='linear')
        warnings.warn(f"Tabulated icdf for gamma shape={shape}.")
        return rv

    def draw_eta(self, dt, var_0, var_t):
        """
        generate Bessel RV from p 285 of Glasserman & Kim (2011)

        Args:
            var_0: initial variance
            var_t: final variance
            dt: time step

        Returns:
            eta (integer >= 0) values (n, )
        """
        phi, exp = self.cir.phi_exp(dt)
        zz = np.sqrt(var_0 * var_t) * phi

        iv_index = 0.5 * self.cir.chi_dim() - 1
        # Equivalent to
        # p0 = np.power(0.5 * zz, iv_index) / (spsp.iv(iv_index, zz) * spsp.gamma(iv_index + 1))
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html
        p0 = np.exp(iv_index*np.log(zz/2) - zz - spsp.gammaln(iv_index + 1))/spsp.ive(iv_index, zz)

        temp = np.arange(1, 16)[:, None]  # Bessel distribution has short tail, 30 maybe enough
        p = zz**2 / (4 * temp * (temp + iv_index))
        p = np.vstack((p0, p)).cumprod(axis=0).cumsum(axis=0)
        rv_uni = self.rv_uniform(spawn=1)
        eta = np.sum(p < rv_uni, axis=0).astype(np.uint32)

        return eta

    def draw_x2(self, dt, shape, size):
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

        gamma_rv = self.rng_spawn[2].standard_gamma(shape, size=(self.kk, size))
        x2 = np.sum(gamma_rv / gamma_n[:, None], axis=0)

        # remainder (truncated) terms
        trunc_m, trunc_vs = self.x2star_avgvar_mv(dt, self.kk)
        # we could call self.draw_from_mv, but we don't to make the code simple.
        # trunc_vs * trunc_m (scale) and shape / trunc_vs are scalar values
        x2 += trunc_vs * trunc_m * self.rng_spawn[2].standard_gamma(shape / trunc_vs, size=size)

        return x2

    def cond_states_step(self, dt, var_0):

        var_t = self.cir.draw_ncx2(dt, var_0, self.rng_spawn[0])
        eta = self.draw_eta(dt, var_0, var_t)

        # sample int_var(integrated variance): Gamma expansion / transform inversion
        # int_var = X1+X2+X3 from formula(2.7) in Glasserman & Kim (2011)

        avgvar = self.draw_x1(dt, var_0, var_t)
        if self.tabulate_x2_z:
            interp_obj = self.x2_icdf_interp(dt, self.cir.chi_dim() / 2, k1=self.params_hash())
            avgvar += interp_obj(self.rng_spawn[2].uniform(size=self.n_path))

            interp_obj = self.x2_icdf_interp(dt, 2, k1=self.params_hash())
            zz = interp_obj(self.rng_spawn[2].uniform(size=eta.sum()))
        else:
            avgvar += self.draw_x2(dt, self.cir.chi_dim() / 2, size=self.n_path)
            zz = self.draw_x2(dt, 2.0, size=eta.sum())

        total = 0
        for i in np.arange(eta.max()):
            eta_above_i = (eta > i)
            count = eta_above_i.sum()
            if count == 0:
                continue
            avgvar[eta_above_i] += zz[total:total+count]
            total += count

        assert eta.sum() == total

        return var_t, avgvar, {}


class HestonMcAndersen2008(HestonMcABC):
    """
    Heston model with conditional Monte-Carlo simulation

    Conditional MC for Heston model based on QE discretization scheme by Andersen (2008).

    Underlying price follows a geometric Brownian motion, and variance of the price follows a CIR process.

    References:
        - Andersen L (2008) Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11:1–42. https://doi.org/10.21314/JCF.2008.189

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.array([60, 70, 100, 140])
        >>> sigma, vov, mr, rho, texp, spot = 0.04, 1, 0.5, -0.9, 10, 100
        >>> m = pf.HestonMcAndersen2008(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.set_num_params(n_path=1e5, dt=1/8, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.32997507, 35.8497697, 13.08467014, 0.29577444
        array([44.28356337, 35.80059515, 13.05391402,  0.29848727])
    """
    psi_c = 1.5  # parameter used by the Andersen QE scheme
    scheme = 4  # Andersen's QE scheme. Alternative: 0/1 for Euler/Milstein, 2 for NCX2, 3 for Pois-Gamma
    correct_martingale = True

    def var_step_qe(self, dt, var_0):
        """
        QE step by Andersen (2008)

        Args:
            dt: time step
            var_0: initial variance

        Returns:
           (variance after dt, martingale-correction log-increment per path)
        """
        m, psi = self.var_mv(dt, var_0)
        psi /= m**2

        zz = self.rv_normal(spawn=0)

        # psi <= psi_c branch: moment-matched normal approximation, Eq (23)
        idx_below = (psi <= self.psi_c)
        ins = 2 / psi[idx_below]
        b2 = (ins - 1) + np.sqrt(ins * (ins - 1))  # b^2, Eq (27)
        a = m[idx_below] / (1 + b2)  # Eq (28)

        var_t = np.zeros(self.n_path)
        var_t[idx_below] = a * (np.sqrt(b2) + zz[idx_below])**2

        # psi > psi_c branch: exponential approximation, Eq (25)
        one_m_u = spst.norm.cdf(zz[~idx_below])  # 1 - U
        var_t_above = np.zeros_like(one_m_u)

        one_m_p = 2 / (psi[~idx_below] + 1)  # 1 - p, Eq (29)
        beta = one_m_p / m[~idx_below]  # Eq (30)

        idx_above = (one_m_u <= one_m_p)
        var_t_above[idx_above] = (np.log(one_m_p / one_m_u) / beta)[idx_above]  # Eq (25)
        var_t[~idx_below] = var_t_above

        # Martingale correction: Andersen (2008) p 24, Proposition 7
        k1_half_k3 = dt/4 * self.rho * (2*self.mr/self.vov - self.rho) - self.rho/self.vov  # k1 + k3/2
        aa = k1_half_k3 + 2*self.rho/self.vov  # A in Proposition 7

        m_corr = -k1_half_k3 * var_0 + dt * self.rho * self.mr * self.theta / self.vov
        m_corr[idx_below] += -aa * b2 * a / (1 - 2*aa*a) + 0.5*np.log1p(-2*aa*a)
        m_corr[~idx_below] += -np.log1p(-one_m_p + (beta*one_m_p)/(beta - aa))

        return var_t, m_corr

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
                var_t = self.cir.draw_euler(dt[i], var_t, self.rng_spawn[0], milstein=milstein)
                var_path[i + 1, :] = var_t
        elif self.scheme == 2:
            for i in range(n_dt):
                var_t = self.cir.draw_ncx2(dt[i], var_t, self.rng_spawn[0])
                var_path[i + 1, :] = var_t
        elif self.scheme == 3:
            for i in range(n_dt):
                var_t, _ = self.cir.draw_pois_gamma(dt[i], var_t, self.rng_spawn[0], self.rng_spawn[1])
                var_path[i + 1, :] = var_t
        elif self.scheme == 4:
            for i in range(n_dt):
                var_t, _ = self.var_step_qe(dt[i], var_t)
                var_path[i + 1, :] = var_t
        else:
            raise ValueError(f'Invalid scheme: {self.scheme}')

        return var_path

    def cond_states_step(self, dt, var_0):

        extra = {}
        if self.scheme < 2:
            milstein = (self.scheme == 1)
            var_t = self.cir.draw_euler(dt, var_0, self.rng_spawn[0], milstein=milstein)
        elif self.scheme == 2:
            var_t = self.cir.draw_ncx2(dt, var_0, self.rng_spawn[0])
        elif self.scheme == 3:
            var_t, _ = self.cir.draw_pois_gamma(dt, var_0, self.rng_spawn[0], self.rng_spawn[1])
        elif self.scheme == 4:
            var_t, m_corr = self.var_step_qe(dt, var_0)
            extra = {'qe_m_corr': m_corr}
        else:
            ValueError(f"Incorrect scheme: {self.scheme}.")

        # Trapezoidal rule
        avgvar = (var_0 + var_t)/2

        return var_t, avgvar, extra


class HestonMcChoiKwok2023PoisTd(HestonMcABC):
    """
    Heston simulation scheme Poisson-conditioned time discretization quadrature

    References:
        - Choi J, Kwok YK (2023) Simulation schemes for the Heston model with Poisson conditioning. Working paper.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.array([60, 70, 100, 140])
        >>> spot = 100
        >>> sigma, vov, mr, rho, texp = 0.04, 1, 0.5, -0.9, 10
        >>> m = pf.HestonMcChoiKwok2023PoisTd(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.set_num_params(n_path=1e5, dt=1/8, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.32997507, 35.8497697, 13.08467014, 0.29577444
        array([44.36484309, 35.87571609, 13.08606262,  0.29620234])
    """

    correct_martingale = True

    def vol_paths(self, tobs):
        var_0 = self.sigma
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        var_path = np.full((n_dt + 1, self.n_path), var_0)  # variance series: V0, V1,...,VT
        var_t = np.full(self.n_path, var_0)

        for i in range(n_dt):
            var_t, _ = self.cir.draw_pois_gamma(dt[i], var_t, self.rng_spawn[0], self.rng_spawn[1])
            var_path[i + 1, :] = var_t

        return var_path

    def cond_states_step(self, dt, var_0):

        var_t, pois = self.cir.draw_pois_gamma(dt, var_0, self.rng_spawn[0], self.rng_spawn[1])
        avgvar_m, avgvar_vs = self.cond_avgvar_mv(dt, var_0, var_t, pois=pois, kk=0)
        extra = {'pois_avgvar_v': avgvar_vs * avgvar_m**2}

        return var_t, avgvar_m, extra

    # def avgvar_var_unexplained(self, texp, dt=None):
    #     """
    #     Unexplained variance ratio of average variance
    #     This is valid only for time discretisation with Poisson conditioning.
    #
    #     Args:
    #         texp: time to expiry
    #         dt: time step
    #
    #     Returns:
    #         ratio
    #     """
    #
    #     if dt is None:
    #         dt = self.dt
    #
    #     mean, var, _ = self.avgvar_mv(texp)
    #
    #     m_x, vs_x = self.x1star_avgvar_mv(dt, kk=0)
    #     m_z, vs_z = self.x2star_avgvar_mv(dt, kk=0)
    #
    #     vov2dt = self.vov**2 * dt
    #
    #     unex = (vs_x * m_x**2 * 2 + vs_z * m_z**2 * 4/vov2dt) * mean * dt / texp
    #     return unex / var


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
    dist = 'ig'  # can override with 'ga' 'ln' 'n'

    def cond_states_step(self, dt, var_0):
        var_t = self.cir.draw_ncx2(dt, var_0, self.rng_spawn[0])
        avgvar_m, avgvar_vs = self.cond_avgvar_mv(dt, var_0, var_t, pois=None, kk=0)
        avgvar = self.draw_from_mv(avgvar_m, avgvar_vs, dist=self.dist)

        return var_t, avgvar, {}


class HestonMcChoiKwok2023PoisGe(HestonMcABC):
    """
    Poisson-conditioned exact simulation scheme from Choi & Kwok (2023).
    This scheme enhances the gamma expansion of Glasserman & Kim (2011).

    References:
        - Choi J, Kwok YK (2023) Simulation schemes for the Heston model with Poisson conditioning. Working paper.
        - Glasserman P, Kim K-K (2011) Gamma expansion of the Heston stochastic volatility model. Finance Stoch 15:267–296. https://doi.org/10.1007/s00780-009-0115-y

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.array([60, 70, 100, 140])
        >>> sigma, vov, mr, rho, texp, spot = 0.04, 1, 0.5, -0.9, 10, 100
        >>> m = pf.HestonMcChoiKwok2023PoisGe(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.set_num_params(n_path=1e5, kk=4, rn_seed=123456)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.32997507, 35.8497697, 13.08467014, 0.29577444
        array([44.35753578, 35.8767866 , 13.12277647,  0.29461611])

    """

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

    def draw_x123(self, dt, var_sum, shape_sum):
        """
        Samples of (X1 + X2 + X3)/dt using truncated Gamma expansion improved in Choi & Kwok (2023)

        Args:
            dt: time step
            var_sum: sum of v_t at the observation times. (n_paths,)
            shape_sum: sum of gamma shape parameters

        Returns:
            (X1 + X2 + X3)/dt  (n_paths,)
        """
        gamma_n, lambda_n = self.gamma_lambda(dt, kk=self.kk)

        if self.kk > 0:
            pois = self.rng_spawn[3].poisson(lam=var_sum * lambda_n[:, None])
            x123 = np.sum(self.rng_spawn[2].standard_gamma(shape=pois + shape_sum) / gamma_n[:, None], axis=0)
        else:
            x123 = np.zeros_like(var_sum)

        trunc_m, trunc_vs = self.x1star_avgvar_mv(dt, self.kk)
        x23_m, x23_vs = self.x2star_avgvar_mv(dt, self.kk)
        m_trunc = trunc_m * var_sum + x23_m * shape_sum
        v_trunc = trunc_vs * trunc_m**2 * var_sum + x23_vs * x23_m**2 * shape_sum
        x123 += self.draw_from_mv(m_trunc, v_trunc / m_trunc**2, self.dist)

        return x123

    def cond_states_step(self, dt, var_0):

        var_t, pois = self.cir.draw_pois_gamma(dt, var_0, self.rng_spawn[0], self.rng_spawn[1])
        shape = 0.5*self.cir.chi_dim() + 2*pois

        # self.draw_x123 returns the average by dt. Need to convert to the int variance by multiplying texp
        avgvar = self.draw_x123(dt, var_0 + var_t, shape)

        return var_t, avgvar, {}
