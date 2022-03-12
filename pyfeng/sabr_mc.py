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


class SabrCondMc(sabr.SabrABC, sv.CondMcBsmABC):
    """
    Conditional MC for SABR model (beta=0,1 or rho=0) with conditional Monte-Carlo simulation

    """

    def vol_paths(self, tobs, mu=0):
        """
        exp(vov_std B_s - 0.5*vov_std^2 * s)  where s = 0, ..., 1, vov_std = vov*sqrt(T)

        Args:
            tobs: observation time (array)
            mu: rn-derivative

        Returns: volatility path (time, path) including the value at t=0
        """

        texp = tobs[-1]
        tobs01 = tobs / texp  # normalized time: 0<s<1
        vov_std = self.vov * np.sqrt(texp)

        log_sig_s = self._bm_incr(tobs01, cum=True)  # B_s (0 <= s <= 1)
        log_rn_deriv = 0.0 if mu == 0 else -mu * (log_sig_s[-1, :] + 0.5 * mu)

        log_sig_s = vov_std * (log_sig_s + (mu - 0.5 * vov_std) * tobs01[:, None])
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
            fwd_cond = rho_sigma / self.vov * (sigma_final - 1)
        else:
            fwd_cond = np.exp(
                rho_sigma
                * (
                    1.0 / self.vov * (sigma_final - 1)
                    - 0.5 * rho_sigma * int_var * texp
                )
            )

        return fwd_cond, vol_cond, log_rn_deriv

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
        vov_std = self.vov * np.sqrt(texp)

        if mu is None:
            mu = 0.5 * (vov_std + np.log(1 + eta ** 2) / vov_std)
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


class SabrExactMcModel(sabr.SabrABC, sv.CondMcBsmABC):
    """
    SABR model with exact simulation
    References:
    Cai, N., Song, Y., & Chen, N. (2017). Exact Simulation of the SABR Model.
    Operations Research, 65(4), 931–951. https://doi.org/10.1287/opre.2017.1617
    """

    def simu_sigma_T(self, sigma_0, texp):
        """
        Volatility at maturity.
        Parameters
        ----------
        texp: time to expiry
        Returns
        -------
        vol at maturity
        """
        zz = np.random.normal(size=self.n_path)
        sigma_T = sigma_0 * np.exp(-1 / 2 * self.vov ** 2 * texp + self.vov * zz * np.sqrt(texp))
        return sigma_T

    def laplace_tranfrom(self, theta, texp, sigma_T):
        """
        formula (15) in the article
        Return the laplace transform function
        Parameters
        ----------
        theta: represent the input variable
        texp: time-to-expiry
        Returns
        -------
        (Laplace transform function)
        """

        l = theta * self.vov ** 2 / self.sigma ** 2
        x = np.log(sigma_T / self.sigma)
        z = l * np.exp(-x) + np.cosh(x)
        phi_value = np.log(z + np.sqrt(z ** 2 - 1))

        numerator = phi_value ** 2 - x ** 2
        denominator = 2 * (self.vov ** 2) * texp
        return 1 / theta * np.exp(-numerator / denominator)

    def origin_func(self, texp, sigma_T, u, m=20, n=35):
        '''
        formula (16) in the article
        Return the original function from transform function
        Parameters
        ----------
        texp: time to expiration
        sigma_T: simulated sigma at time of expiration
        u: the u of original function
        Returns
        -------
        original functions with parameter u
        '''
        Euler_term = np.zeros(self.n_path)
        sum_term = np.zeros(m + 1)
        comb_vec = np.frompyfunc(spsp.comb, 2, 1)
        comb_term = comb_vec(m, np.arange(0, m + 1)).astype(int)
        for i in range(0, m + 1):
            sum_term[i] = np.sum(
                (-1) ** np.arange(0, n + i + 1) * self.laplace_tranfrom((m - 2j * np.pi * np.arange(0, n + i + 1)) / u,
                                                                        texp, sigma_T).real)
        Euler_term = np.sum(comb_term * sum_term * 2 ** (-m))

        origin_L = 1 / (2 * u) * self.laplace_tranfrom(m / (2 * u), texp, sigma_T).real \
                   * np.exp(m / 2) + Euler_term / u * np.exp(m / 2) - np.exp(-m)

        return origin_L

    def conditional_state(self, texp):
        """
        Return Exact integrated variance from samples of disturibution
        The returns values are for sigma_0 = self.sigma and t = T
        Parameters
        ----------
        texp: float
            time-to-expiry
        Returns
        -------
        (sigma at the time of expiration, Exact integrated variance)
        """
        _u = np.random.uniform(size=self.n_path)
        int_var = np.zeros(self.n_path)
        sigma_final = self.simu_sigma_T(self.sigma, texp)
        for i in range(self.n_path):
            Lh_func = partial(self.origin_func, texp, sigma_final[i])
            obj_func = lambda x: Lh_func(x) - _u[i]
            sol = spop.root(obj_func, 1 / texp ** 3)
            int_var[i] = 1 / sol.x
        return sigma_final, int_var

    def exact_fwd_vol(self, spot, texp):
        """
        Returns forward prices and volatility conditional on sigma at time of expiration  and integrated variance)
        Parameters
        ----------
        spot: spot prices
        texp: time-to-expiry
        Returns
        -------
        (forward price, volatility)
        """
        assert abs(self.beta - 1.0) < 1e-8

        rhoc = np.sqrt(1.0 - self.rho ** 2)
        sigma_final, int_var = self.conditional_state(texp)
        fwd_exact = spot * np.exp(self.rho / self.vov * (sigma_final - self.sigma) - 0.5 * self.rho ** 2 * int_var)
        vol_exact = rhoc * np.sqrt(int_var / texp)
        return fwd_exact, vol_exact

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
        u_lst = np.random.uniform(size=self.n_path)
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

    @staticmethod
    def sabr_chi2_cdf(beta, VT, spot, u):
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
        C0 = SabrExactMcModel.C0_func(VT, beta, u)
        return SabrExactMcModel.chi2_cdf_appr(A0, 1 / (1 - beta), C0)

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

    def price(self, strike, spot, texp, cp_sign=1):
        """
        over ride the price method in parent class to incorporate 0< beta < 1
        if beta = 1 or 0, use bsm model
        if 0 < beta < 1, use  mc simulation
        Args:
            strike: 1d array,  an array of strike prices
            spot: float, spot prices
            texp: time to expiry
            cp_sign:
        Returns:
            1d array of option prices
            when x < 500 and l < 500:
                equation (19) in Cai(2017)
                The recursive alogorithm propose by Ding(1992) to calculate chi-2 cdf
            when x > 500 or l > 500:
                analytic approximation of Penev and Raykov(2000)
        Example(beta = 1):
            fwd = 100
            strike = np.arange(50,151,10)
            texp = 1
            params = {"sigma": 0.2, "vov": 0.3, "rho": 0, "beta":1}
            sabr_mc_model = pyfe.ExactMcSabr(**params)
            sabr_mc_model.set_mc_params(n_path=1000)
            price = sabr_mc_model.price(strike, fwd, texp)
        """

        if isinstance(strike, float):
            strike = np.array([strike])
        if self.beta == 1:
            fwd_exact, vol_exact = self.exact_fwd_vol(spot, texp)
            base_model = bsm.BsmModel(vol_exact[:, None], intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)
            price_grid = base_model.price(strike, fwd_exact[:, None], texp, cp_sign)
            price = np.mean(price_grid, axis=0)
            return price[0] if price.size == 1 else price
        else:  # 0 < beta < 1
            assert self.rho == 0, "rho has to be 0 fro 0<beta<1"
            price_lst = []
            for kk in strike:
                _, VT = self.conditional_state(texp)
                price = np.maximum(self.simu_ST(self.beta, VT, spot) - kk, 0) * np.exp(-self.intr * texp)
                price = np.mean(price)
                price_lst.append(price)
            return np.array(price_lst)