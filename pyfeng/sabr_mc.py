import math
import abc
import numpy as np
import scipy.optimize as spop
import scipy.stats as spst
import scipy.special as spsp
from functools import partial
import scipy.integrate as scint
from . import sabr
from . import sv_abc as sv

#### Use of RN generation spawn:
# 0: simulation of volatility (normal)
# 2: integrated/average variance (lognormal)
# 5: asset return

class SabrMcABC(sabr.SabrABC, sv.CondMcBsmABC, abc.ABC):

    def vol_step(self, dt, log=False):
        """
        SABR sigma after dt according to the volatility dynamics (GBM).
        Because of the multiplicative property of sigma, we assume sigma_0 = 1

        Args:
            dt: time step
            log: if True, return log(sigma). False by default.
        Returns:
            sigma_dt
        """
        vovn = self.vov * np.sqrt(dt)
        zz = self.rv_normal(spawn=0)

        if log:
            return vovn * (zz - vovn/2)
        else:
            return np.exp(vovn * (zz - vovn/2))

    @abc.abstractmethod
    def cond_states_step(self, dt, sigma_0):
        """
        Final variance after dt and average variance over (0, dt) given sigma_0.
        `sigma_0` should be an array of (self.n_path, )

        Args:
            dt: time step
            sigma_0: initial volatility

        Returns:
            (sigma after dt, average variance during dt)
        """
        return NotImplementedError

    def draw_log_return(self, dt, sigma_0, sigma_t, avgvar):
        """
        Samples log return, log(S_t/S_0). Currently implemented only for beta=1

        Args:
            dt: time step
            sigma_0: initial variance
            sigma_t: final variance
            avgvar: average variance

        Returns:
            log return (self.n_path, )
        """

        assert np.isclose(self.beta, 1.0)
        ln_m = (self.intr - self.divr)*dt + self.rho/self.vov*(sigma_t - sigma_0) - 0.5*avgvar*dt
        ln_sig = np.sqrt((1.0 - self.rho**2) * dt * avgvar)
        zn = self.rv_normal(spawn=5)
        return ln_m + ln_sig * zn

    def cond_spot_sigma(self, texp, fwd, mu=0):
        """
        Spot and sigma ratio.

        Args:
            texp: time to expiry
            fwd: forward. Only used for calculating alpha
            mu: BM shift (currently not used)

        Returns:
            (spot ratio, sigma ratio)
        """
        tobs = self.tobs(texp)
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        #### sigma is normalized to 1
        sigma_t = np.full(self.n_path, 1.0)
        avgvar = np.zeros(self.n_path)

        for i in range(n_dt):
            sigma_t, avgvar_inc = self.cond_states_step(dt[i], sigma_t)
            avgvar += avgvar_inc * dt[i]

        avgvar /= texp

        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)
        vol_cond = rhoc * np.sqrt(avgvar)
        rho_alpha = self.rho * alpha

        if np.isclose(self.beta, 0):
            spot_cond = 1 + rho_alpha / self.vov * (sigma_t - 1)
        else:
            spot_cond = 1.0 / self.vov * (sigma_t - 1) - 0.5 * rho_alpha * avgvar * texp
            np.exp(rho_alpha * spot_cond, out=spot_cond)

        return spot_cond, vol_cond

    def price(self, strike, spot, texp, cp=1):
        fwd = self.forward(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)
        kk = strike / fwd

        fwd_ratio, vol_ratio = self.cond_spot_sigma(texp, fwd)

        if self.correct_fwd:
            fwd_ratio /= np.mean(fwd_ratio)

        if self.beta > 0:
            ind = (fwd_ratio > 1e-16)
        else:
            ind = (fwd_ratio > -999)

        fwd_ratio = np.expand_dims(fwd_ratio[ind], -1)
        vol_ratio = np.expand_dims(vol_ratio[ind], -1)

        base_model = self.base_model(alpha * vol_ratio)
        base_model.is_fwd = True
        price_vec = base_model.price(kk, fwd_ratio, texp, cp=cp)
        price = fwd * np.sum(price_vec, axis=0) / self.n_path

        return price


class SabrMcTimeDisc(SabrMcABC):
    """
    Conditional MC for SABR model (beta=0,1 or rho=0) with conditional Monte-Carlo simulation

    """

    scheme = 0

    def cond_states_step_trapez(self, dt, sigma_0):
        sigma_t = sigma_0 * self.vol_step(dt)
        avgvar = (sigma_0**2 + sigma_t**2) / 2.

        return sigma_t, avgvar

    def cond_states_step_chen_2012(self, dt, sigma_0):

        vovn = self.vov * np.sqrt(dt)
        zhat = self.rv_normal(spawn=0) - vovn/2.
        sigma_t = sigma_0 * np.exp(vovn * zhat)

        m1, mnc2, mnc3, mnc4 = self.cond_avgvar_mnc4(vovn, zhat)
        scale = np.sqrt(np.log(mnc2/m1**2))
        avgvar = sigma_0**2 * m1 * np.exp(scale*(self.rv_normal(spawn=2) - scale/2.))

        return sigma_t, avgvar

    def cond_states_step(self, dt, sigma_0):
        sigma_t = self.vol_step(dt)

        if self.scheme == 0:
            sigma_t, avgvar = self.cond_states_step_trapez(dt, sigma_0)
        elif self.scheme == 1:
            sigma_t, avgvar = self.cond_states_step_chen_2012(dt, sigma_0)
        else:
            ValueError(f"Incorrect scheme: {self.scheme}.")

        return sigma_t, avgvar


    def vol_paths(self, tobs, mu=0):
        """
        exp(vovn B_s - 0.5*vovn^2 * s)  where s = 0, ..., 1, vovn = vov * sqrt(T)

        Args:
            tobs: observation time (array)
            mu: rn-derivative

        Returns: volatility path (time, path) including the value at t=0
        """

        texp = tobs[-1]
        tobs01 = tobs / texp  # normalized time: 0<s<1
        vovn = self.vov * np.sqrt(texp)

        log_sig_s = self._bm_incr(tobs01, cum=True)  # B_s (0 <= s <= 1)
        log_rn_deriv = 0.0 if mu == 0 else -mu * (log_sig_s[-1, :] + 0.5 * mu)

        log_sig_s = vovn * (log_sig_s + (mu - 0.5 * vovn) * tobs01[:, None])
        log_sig_s = np.insert(log_sig_s, 0, np.zeros(log_sig_s.shape[1]), axis=0)
        return np.exp(log_sig_s), log_rn_deriv

    def cond_spot_sigma_volpath(self, texp, sigma_0, mu=0):
        """
        Kept for backward compatibility and rn_deriv

        Args:
            texp:
            sigma_0:
            mu:

        Returns:

        """
        rhoc = np.sqrt(1.0 - self.rho**2)
        rho_sigma = self.rho * sigma_0

        tobs = self.tobs(texp)
        n_dt = len(tobs)
        sigma_paths, log_rn_deriv = self.vol_paths(tobs, mu=mu)
        sigma_final = sigma_paths[-1, :]
        int_var = scint.simps(sigma_paths**2, dx=1, axis=0) / n_dt
        vol_cond = rhoc * np.sqrt(int_var)

        if np.isclose(self.beta, 0):
            spot_cond = rho_sigma / self.vov * (sigma_final - 1)
        else:
            spot_cond = 1.0 / self.vov * (sigma_final - 1) - 0.5 * rho_sigma * int_var * texp
            np.exp(rho_sigma * spot_cond, out=spot_cond)

        return spot_cond, vol_cond, log_rn_deriv

    def mass_zero(self, spot, texp, log=False, mu=0):

        assert 0 < self.beta < 1
        assert np.isclose(self.rho, 0.0)

        ### We calculate under normalization by fwd.
        fwd = self.forward(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)

        ### mu is currently not used.
        if mu is None:
            eta = self.vov * np.power(spot, betac) / (self.sigma * betac)
            vovn = self.vov * np.sqrt(texp)
            mu = 0.5 * (vovn + np.log1p(eta**2) / vovn)
            # print(f'mu = {mu}')

        fwd_ratio, vol_ratio = self.cond_spot_sigma(texp, fwd)
        assert np.isclose(fwd_ratio, 1.0).all()
        log_rn_deriv = 0.0  ## currently not used

        base_model = self.base_model(alpha * vol_ratio)
        base_model.is_fwd = True

        if log:
            log_mass_grid = base_model.mass_zero(1.0, texp, log=True) + log_rn_deriv
            log_mass_max = np.amax(log_mass_grid)
            log_mass_grid -= log_mass_max
            log_mass = log_mass_max + np.log(np.mean(np.exp(log_mass_grid)))
            return log_mass
        else:
            mass_grid = base_model.mass_zero(1.0, texp, log=False) * np.exp(log_rn_deriv)
            mass = np.mean(mass_grid)
            return mass

    def return_var_realized(self, texp, cond):
        return None


class SabrMcCai2017Exact(SabrMcABC):
    """
    Cai et al. (2017)'s exact simulation of the SABR model

    References:
        - Cai N, Song Y, Chen N (2017) Exact Simulation of the SABR Model. Oper Res 65:931–951. https://doi.org/10.1287/opre.2017.1617
    """
    m_inv = 20
    m_euler = 20
    n_euler = 35
    comb_coef = None
    nn = None

    def set_num_params(self, n_path=10000, dt=None, rn_seed=None, antithetic=True, m_inv=20, m_euler=20, n_euler=35):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step
            rn_seed: random number seed
            antithetic: antithetic
            m_inv: parameter M in Laplace inversion, Eq. (16)
            m_euler: parameter m in Euler transformation E(m,n)
            n_euler: parameter n in Euler transformation E(m,n)
        """
        self.m_inv = m_inv
        self.m_euler = m_euler
        self.n_euler = n_euler
        self.comb_coef = spsp.comb(self.m_euler, np.arange(0, self.m_euler+0.1)) * np.power(0.5, self.m_euler)
        assert abs(self.comb_coef.sum()-1) < 1e-8
        self.nn = np.arange(0, self.m_euler + self.n_euler + 0.1)
        super().set_num_params(n_path, dt, rn_seed, antithetic)


    def cond_laplace(self, theta, vovn, sigma_t):
        """
        Eq. (15) of the paper
        Return the laplace transform function

        Args:
            theta: dummy variable
            vovn: vov * sqrt(texp)
            sigma_t: normalized sigma final

        Returns:
            (Laplace transform function)
        """

        x = np.log(sigma_t)
        lam = theta * vovn**2
        z = 0.5*sigma_t + (0.5 + lam)/sigma_t
        phi = np.log(z + np.sqrt(z**2 - 1))

        return np.exp((x**2 - phi**2) / (2*vovn**2)) / theta

    def inv_laplace(self, u, vovn, sigma_t):
        """
        Eq. (16) in the article
        Return the original function from transform function

        Args:
            u: dummy variable
            vovn: vov * sqrt(texp)
            sigma_t: final volatility

        Returns:
            original function value at u
        """

        ## index from 0 to m + n
        ss_j = self.cond_laplace((self.m_inv - 2j * np.pi * self.nn[:, None]) / (2*u), vovn, sigma_t).real
        term1 = 0.5 * ss_j[0, :]
        ss_j[1::2, :] *= -1
        np.cumsum(ss_j, axis=0, out=ss_j)
        term2 = np.sum(self.comb_coef[:, None] * ss_j[self.n_euler:, :], axis=0)

        origin_L = np.exp(self.m_inv/2) / u * (-term1 + term2)

        if np.isscalar(u):
            origin_L = origin_L[0]

        return origin_L

    def draw_cond_avgvar(self, dt, sigma_t):
        """
        Draw normalized average variance given sigma_t (and sigma_0 = 1)

        int_0^1 exp{2 vovn Z_s - vovn^2 s} ds | exp(Z_1 - vovn/2) = sigma_t

        Args:
            dt: time step
            sigma_t: final sigma given sigma_0=1

        Returns:
            (n_path, ) array
        """
        vovn = self.vov * np.sqrt(dt)
        uu = self.rv_uniform(spawn=2)
        avgvar = np.zeros_like(sigma_t)

        for i in range(len(sigma_t)):
            obj_func = lambda x: self.inv_laplace(x, vovn, sigma_t[i]) - uu[i]
            sol = spop.brentq(obj_func, 1e-7, 1e5)
            avgvar[i] = 1 / sol

        """
        ## Vectorized newton method, but doesn't work well
        
        zz = self.rv_normal(spawn=2)
        uu = spst.norm.cdf(zz)
        ln_m, m2 = self.cond_avgvar_mv(vovn, np.log(sigma_t) / vovn)
        ln_sig = np.sqrt(np.log(m2 / ln_m ** 2))
        print(sigma_t, '\n', ln_sig, '\n', uu)

        def obj_func(z):
            x = ln_m * np.exp(ln_sig * (z - ln_sig / 2))
            return self.inv_laplace(x, vovn, sigma_t) - uu

        avgvar = 1 / spop.newton(obj_func, zz)
        """
        return avgvar

    def cond_states_step(self, dt, sigma_0):

        sigma_t = self.vol_step(dt)  ## ratio
        avgvar = sigma_0**2 * self.draw_cond_avgvar(dt, sigma_t)
        sigma_t *= sigma_0

        return sigma_t, avgvar

    # The algorithem below is about pricing when 0<=beta<1
    def simu_ST(self, beta, VT, spot):
        '''
        calculate C(u), C(u) will be used in the Ft's cdf
        equation (6) in Cai(2017)
        Parameters
        ----------
        VT:float,  intergated sigma
        beta: float, beta of the sabr model
        spot: spot prices
        Returns
        ----------
        cdf of a central chi2 distribution with x=A0, degree of freedom = 1/(1 - beta)
        '''

        u_lst = self.rng.uniform(size=self.n_path)
        forward_ls = np.zeros(self.n_path)

        for i in range(self.n_path):
            u = u_lst[i]
            VTi = VT[i]
            P0 = 1 - self.central_chi2_cdf(beta, VTi, spot)
            if u <= P0:
                forward_ls[i] = 0
            else:
                _chi2_cdf = partial(self.sabr_chi2_cdf, beta, VTi, spot)
                obj_func = lambda u_hat: 1 - _chi2_cdf(u_hat) - u
                sol = spop.root(obj_func, spot)
                forward_ls[i] = sol.x
        return forward_ls

    @staticmethod
    def central_chi2_cdf(beta, VT, spot):
        '''
        calculate C(u), C(u) will be used in the Ft's cdf
        equation (6) in Cai(2017)
        Parameters
        ----------
        VT:float,  intergated sigma
        beta: float, beta of the sabr model
        spot: spot prices
        Returns
        ----------
        cdf of a central chi2 distribution with x=A0, degree of freedom = 1/(1 - beta)
        '''

        A0 = 1 / VT * (spot**(1 - beta) / (1 - beta))**2
        return spst.chi2.cdf(A0, 1 / (1 - beta))

    @staticmethod
    def C0_func(VT, beta, u):
        '''
        calculate C(u), C(u) will be used in the Ft's cdf
        equation (6) in Cai(2017)
        Parameters
        ----------
        VT:float,  intergated sigma
        beta: float, beta of the sabr model
        u: float, C0 function's input
        Returns
        ----------
        C0 function
        '''
        numerator = u**(2 * (1 - beta))
        return 1 / VT * numerator / (1 - beta)**2

    @classmethod
    def sabr_chi2_cdf(cls, beta, VT, spot, u):
        '''
        Equation (18) in Cai(2017)'s paper
        calculate chi2_cdf only for sabr model
        (based on chi2_cdf_approximation, but modify to cater the need of sabr model)
        Parameters
        ----------
        beta: float,  beta in the sabr model
        VT: float,  intergrated sigma
        spot: float,  spot prices
        u: float, C0 function's input
        Returns
        ----------
        cdf of the chi-square distribution specified by a sabr model's parameter and u
        '''
        A0 = 1 / VT * (np.power(spot, 1.0 - beta) / (1 - beta))**2  # Equation (6) in Cai's paper
        C0 = cls.C0_func(VT, beta, u)
        return cls.chi2_cdf_appr(A0, 1 / (1 - beta), C0)

    @staticmethod
    def chi2_cdf_appr(x, sigma, l):
        '''
        when x < 500 and l < 500:
            equation (19) in Cai(2017)
            The recursive alogorithm propose by Ding(1992) to calculate chi-2 cdf
        when x > 500 or l > 500:
            analytic approximation of Penev and Raykov(2000)
        Parameters
        ----------
        x: x value in the cdf
        sigma:  sigma parameter for the chi2 distribution
        l: lambda parameter for the chi2 distribution
        Returns
        ----------
        cdf of chi2 distribution of given x, sigma and lambda
        '''
        cdf = 0
        k = 0
        if x <= 500 and l <= 500:
            while True:
                if k >= 1 and (sigma + 2 * k) > x and t * x / (sigma + 2 * k - x) <= 1e-7:
                    # note that this condition come from the Cai(2017) and Ding(1992) and use short-circuit tricks in cs
                    break
                elif k == 0:
                    t = 1 / math.gamma(sigma / 2 + 1) * (x / 2)**(sigma / 2) * np.exp(-x / 2)
                    y = np.exp(-l / 2)
                    u = y
                else:
                    t = t * x / (sigma + 2 * k)
                    u = u * l / (2 * k)
                    y = y + u
                k += 1
                element = y * t
                cdf += element
        else:  # x >-500 or l>500
            K_func = lambda s: ((1 - s) * np.log(1 - s) + s - s**2 / 2) / s**2

            def yita_func(mu2, s):
                K_s = K_func(1 - s)
                numerator = 1 + 2 * mu2 * s - 2 * K_s - s - 2 * mu2 * s**2
                denominator = 1 + 2 * mu2 * s - 2 * K_s
                return numerator / denominator

            def theta_func(mu2, yita, s):
                return -1.5 * (1 + 4 * mu2 * s) / (1 + 2 * mu2 * s)**2 + 5 / 3 * (1 + 3 * mu2 * s)**2 / (
                            1 + 2 * mu2 * s)**3 + \
                       2 * (1 + 3 * mu2 * s) / (s - 1) / (1 + 2 * mu2 * s)**2 + 3 * yita / (s - 1)**2 / (
                                   1 + 2 * mu2 * s) - \
                       (1 + 2 * K_func(yita)) * yita**2 / 2 / (s - 1)**2 / (1 + 2 * mu2 * s)

            mu2 = l / sigma
            mu = np.sqrt(mu2)
            s = (np.sqrt(1 + 4 * x * mu**2 / sigma) - 1) / (2 * mu2)
            yita = yita_func(mu2, s)
            theta_s = partial(theta_func, mu2, yita)
            z = np.sign(s - 1) * (sigma * (s - 1)**2 * (1 / 2 / s + mu2 - K_func(1 - s) / s) - np.log(
                1 / s - 2 * K_func(1 - s) / (s * (1 + 2 * mu2 * s)))
                                  + 2 * theta_s(s) / sigma)**0.5
            cdf = spst.norm.cdf(z)
        return cdf

    def return_var_realized(self, texp, cond):
        return None
