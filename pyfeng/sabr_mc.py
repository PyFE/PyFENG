import math
import numpy as np
import scipy.optimize as spop
import scipy.stats as spst
import scipy.special as spsp
from functools import partial
import scipy.integrate as scint
from . import sabr
from . import sv_abc as sv
from . import cev


class SabrMcCond(sabr.SabrABC, sv.CondMcBsmABC):
    """
    Conditional MC for SABR model (beta=0,1 or rho=0) with conditional Monte-Carlo simulation

    """

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

    def cond_spot_sigma(self, texp, mu=0):
        rhoc = np.sqrt(1.0 - self.rho ** 2)
        rho_sigma = self.rho * self.sigma

        tobs = self.tobs(texp)
        n_dt = len(tobs)
        sigma_paths, log_rn_deriv = self.vol_paths(tobs, mu=mu)
        sigma_final = sigma_paths[-1, :]
        int_var = scint.simps(sigma_paths ** 2, dx=1, axis=0) / n_dt
        vol_cond = rhoc * np.sqrt(int_var)
        if np.isclose(self.beta, 0):
            spot_cond = rho_sigma / self.vov * (sigma_final - 1)
        else:
            spot_cond = 1.0 / self.vov * (sigma_final - 1) - 0.5 * rho_sigma * int_var * texp
            np.exp(rho_sigma * spot_cond, out=spot_cond)

        return spot_cond, vol_cond, log_rn_deriv

    def price(self, strike, spot, texp, cp=1):
        fwd = self.forward(spot, texp)
        fwd_cond, vol_cond, log_rn_deriv = self.cond_spot_sigma(texp)
        if np.isclose(self.beta, 0):
            base_model = self._m_base(self.sigma * vol_cond, is_fwd=True)
            price_grid = base_model.price(strike[:, None], fwd + fwd_cond, texp, cp=cp)
            price = np.mean(price_grid * np.exp(log_rn_deriv), axis=1)
        else:
            alpha = self.sigma / np.power(spot, 1.0 - self.beta)
            kk = strike / fwd

            base_model = self._m_base(alpha * vol_cond, is_fwd=True)
            price_grid = base_model.price(kk[:, None], fwd_cond, texp, cp=cp)
            price = fwd * np.mean(price_grid * np.exp(log_rn_deriv), axis=1)

        return price

    def mass_zero(self, spot, texp, log=False, mu=0):
        assert 0 < self.beta < 1
        assert self.rho == 0

        eta = (
            self.vov
            * np.power(spot, 1.0 - self.beta)
            / (self.sigma * (1.0 - self.beta))
        )
        vovn = self.vov * np.sqrt(texp)

        if mu is None:
            mu = 0.5 * (vovn + np.log(1 + eta ** 2) / vovn)
            # print(f'mu = {mu}')

        fwd_cond, vol_cond, log_rn_deriv = self.cond_spot_sigma(texp, mu=mu)
        base_model = cev.Cev(sigma=self.sigma * vol_cond, beta=self.beta)
        if log:
            log_mass_grid = base_model.mass_zero(spot, texp, log=True) + log_rn_deriv
            log_mass_max = np.amax(log_mass_grid)
            log_mass_grid -= log_mass_max
            log_mass = log_mass_max + np.log(np.mean(np.exp(log_mass_grid)))
            return log_mass
        else:
            mass_grid = base_model.mass_zero(spot, texp, log=False) * np.exp(
                log_rn_deriv
            )
            mass = np.mean(mass_grid)
            return mass


class SabrMcExactCai2017(sabr.SabrABC, sv.CondMcBsmABC):
    """
    Cai et al (2017)'s exact simulation of the SABR model

    References:
        - Cai N, Song Y, Chen N (2017) Exact Simulation of the SABR Model. Oper Res 65:931–951. https://doi.org/10.1287/opre.2017.1617
    """
    m_inv = 20
    m_euler = 20
    n_euler = 35
    comb_coef = None
    nn = None

    def set_num_params(self, n_path=10000, m_inv=20, m_euler=20, n_euler=35, rn_seed=None, antithetic=True):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            m_inv: parameter M in Laplace inversion, Eq. (16)
            m_euler: parameter m in Euler transformation E(m,n)
            n_euler: parameter n in Euler transformation E(m,n)
            rn_seed: random number seed
            antithetic: antithetic
        """
        self.n_path = int(n_path)
        self.m_inv = m_inv
        self.m_euler = m_euler
        self.n_euler = n_euler
        self.dt = None
        self.rn_seed = rn_seed
        self.antithetic = antithetic
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)
        self.comb_coef = spsp.comb(self.m_euler, np.arange(0, self.m_euler+0.1)) * np.power(0.5, self.m_euler)
        assert abs(self.comb_coef.sum()-1) < 1e-8
        self.nn = np.arange(0, self.m_euler + self.n_euler + 0.1)

    def sigma_final(self, vovn):
        """
        Final Sigma

        Parameters
        ----------
        texp: time to expiry
        Returns
        -------
        vol at maturity
        """

        if self.antithetic:
            zz = self.rng.standard_normal(size=self.n_path // 2)
            zz = np.hstack([zz, -zz])
        else:
            zz = self.rng.standarad_normal(size=self.n_path)

        sigma_T = np.exp(vovn * (zz - vovn/2))
        return sigma_T

    def cond_laplace(self, theta, vovn, sigma_T):
        """
        Eq. (15) of the paper
        Return the laplace transform function

        Args:
            theta: dummy variable
            vovn: vov * sqrt(texp)
            sigma_T: normalized sigma final

        Returns:
            (Laplace transform function)
        """

        x = np.log(sigma_T)
        lam = theta * vovn**2
        z = 0.5*sigma_T + (0.5 + lam)/sigma_T
        phi = np.log(z + np.sqrt(z ** 2 - 1))

        return np.exp((x**2 - phi**2) / (2*vovn**2)) / theta

    def inv_laplace(self, u, vovn, sigma_T):
        """
        Eq. (16) in the article
        Return the original function from transform function

        Args:
            u: original variable
            vovn: vov * sqrt(texp)
            sigma_T: final volatility

        Returns:
            original function value at u
        """

        ## index from 0 to m + n
        ss_j = self.cond_laplace((self.m_inv - 2j * np.pi * self.nn) / (2*u), vovn, sigma_T).real
        term1 = 0.5 * ss_j[0]
        ss_j[1::2] *= -1
        np.cumsum(ss_j, out=ss_j)
        term2 = np.sum(self.comb_coef * ss_j[self.n_euler:])

        origin_L = np.exp(self.m_inv/2) / u * (-term1 + term2)

        return origin_L

    def cond_int_var(self, vovn, sigma_final):
        """
        Normalized integraged variance samples.

        Args:
            vovn: vov * sqrt(texp)
            sigma_final: final volatility

        Returns:
            (n_path, 1) array
        """

        if self.antithetic:
            u_rn = self.rng.uniform(size=self.n_path // 2)
            u_rn = np.hstack([u_rn, 1 - u_rn])
        else:
            u_rn = self.rng.uniform(size=self.n_path)

        int_var = np.zeros(self.n_path)

        for i in range(self.n_path):
            obj_func = lambda x: self.inv_laplace(x, vovn, sigma_final[i]) - u_rn[i]

            sol = spop.brentq(obj_func, 0.000001, 100)
            int_var[i] = 1 / sol

        return int_var

    def cond_spot_sigma(self, texp):

        rhoc = np.sqrt(1.0 - self.rho ** 2)
        rho_sigma = self.rho * self.sigma
        vovn = self.vov * np.sqrt(texp)

        sigma_final = self.sigma_final(vovn)
        int_var = self.cond_int_var(vovn, sigma_final)
        #print(1/np.max(int_var), 1/np.min(int_var))

        vol_cond = rhoc * np.sqrt(int_var)
        if np.isclose(self.beta, 0):
            fwd_cond = rho_sigma / self.vov * (sigma_final - 1)
        else:
            fwd_cond = np.exp(
                rho_sigma * (1.0/self.vov * (sigma_final - 1) - 0.5 * rho_sigma * int_var * texp)
            )
        return fwd_cond, vol_cond

    def price(self, strike, spot, texp, cp=1):
        # The formula is exactly same as that of SabrCondMc except rn_deriv. Need to merge
        fwd = self.forward(spot, texp)
        fwd_cond, vol_cond = self.cond_spot_sigma(texp)
        if np.isclose(self.beta, 0):
            base_model = self._m_base(self.sigma * vol_cond, is_fwd=True)
            price_grid = base_model.price(strike[:, None], fwd + fwd_cond, texp, cp=cp)
            price = np.mean(price_grid, axis=1)
        else:
            alpha = self.sigma / np.power(spot, 1.0 - self.beta)
            kk = strike / fwd

            base_model = self._m_base(alpha * vol_cond, is_fwd=True)
            price_grid = base_model.price(kk[:, None], fwd_cond, texp, cp=cp)
            price = fwd * np.mean(price_grid, axis=1)

        return price


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

        A0 = 1 / VT * (spot ** (1 - beta) / (1 - beta)) ** 2
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
        numerator = u ** (2 * (1 - beta))
        return 1 / VT * numerator / (1 - beta) ** 2

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
        A0 = 1 / VT * (spot ** (1 - beta) / (1 - beta)) ** 2  # Equation (6) in Cai's paper
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
                    t = 1 / math.gamma(sigma / 2 + 1) * (x / 2) ** (sigma / 2) * np.exp(-x / 2)
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
            K_func = lambda s: ((1 - s) * np.log(1 - s) + s - s ** 2 / 2) / s ** 2

            def yita_func(mu2, s):
                K_s = K_func(1 - s)
                numerator = 1 + 2 * mu2 * s - 2 * K_s - s - 2 * mu2 * s ** 2
                denominator = 1 + 2 * mu2 * s - 2 * K_s
                return numerator / denominator

            def theta_func(mu2, yita, s):
                return -1.5 * (1 + 4 * mu2 * s) / (1 + 2 * mu2 * s) ** 2 + 5 / 3 * (1 + 3 * mu2 * s) ** 2 / (
                            1 + 2 * mu2 * s) ** 3 + \
                       2 * (1 + 3 * mu2 * s) / (s - 1) / (1 + 2 * mu2 * s) ** 2 + 3 * yita / (s - 1) ** 2 / (
                                   1 + 2 * mu2 * s) - \
                       (1 + 2 * K_func(yita)) * yita ** 2 / 2 / (s - 1) ** 2 / (1 + 2 * mu2 * s)

            mu2 = l / sigma
            mu = np.sqrt(mu2)
            s = (np.sqrt(1 + 4 * x * mu ** 2 / sigma) - 1) / (2 * mu2)
            yita = yita_func(mu2, s)
            theta_s = partial(theta_func, mu2, yita)
            z = np.sign(s - 1) * (sigma * (s - 1) ** 2 * (1 / 2 / s + mu2 - K_func(1 - s) / s) - np.log(
                1 / s - 2 * K_func(1 - s) / (s * (1 + 2 * mu2 * s)))
                                  + 2 * theta_s(s) / sigma) ** 0.5
            cdf = spst.norm.cdf(z)
        return cdf
