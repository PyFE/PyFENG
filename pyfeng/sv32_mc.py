# -*- coding: utf-8 -*-
"""
Created on Mon, May 3, 2021
Last modified on Fri, May 7, 2021
Conditional MC for 3/2 model based on QE discretization scheme by Andersen(2008)
@author: Xueyang & Xiaoyin
"""
import numpy as np
import pyfeng as pf
import scipy.stats as spst
import scipy.integrate as spint
import scipy.optimize as spop
from scipy.misc import derivative
from mpmath import besseli
from .bsm import Bsm
from .norm import Norm
from . import sv_abc, sv32_mc2

class Sv32McCondQE:
    """
    Conditional MC for 3/2 model based on QE discretization scheme by Andersen(2008)

    Underlying price is assumed to follow a geometric Brownian motion.
    Volatility (variance) of the price is assumed to follow 3/2 model by Heston (1997) and Lewis (2000).

    Example:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = [100.0, 140.0, 70.0]
        >>> forward = 100
        >>> delta = [1, 1/2, 1/4, 1/8, 1/16, 1/32]
        >>> vov, kappa, rho, texp, theta, sigma = [1, 0.5, -0.9, 10, 0.04, np.sqrt(0.04)]
        >>> sv32_cmc_qe = pf.Sv32McCondQE(vov=vov, kappa=kappa, rho=rho, theta=theta)
        >>> price_cmc = np.zeros([len(delta), len(strike)])
        >>> for d in range(len(delta)):
        >>>     price_cmc[d, :] = sv32_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], path=1e5, seed=123456)
        >>> price_cmc
        array([[22.95314785, 10.44093783, 38.98364955],
               [23.2425543 , 10.67162543, 39.26731165],
               [23.20965635, 10.64143576, 39.21865023],
               [22.93527518, 10.4758762 , 38.87971674],
               [22.9298084 , 10.47613694, 38.88556212],
               [23.12806844, 10.56484306, 39.16893668]])
    """

    def __init__(self, vov=1, kappa=0.5, rho=-0.9, theta=0.04):
        """
        Initiate a 3/2 model

        Args:
            vov: volatility of variance, strictly positive
            kappa: speed of variance's mean-reversion, strictly positive
            rho: correlation between BMs of price and vol
            theta: long-term mean (equilibirum level) of the variance, strictly positive
        """
        self.vov = vov
        self.kappa = kappa
        self.rho = rho
        self.theta = theta

        self.psi_points = None  # for TG scheme only
        self.rx_results = None
        self.dis = 1e-3

    def price(
        self,
        strike,
        spot,
        texp,
        sigma,
        delta,
        intr=0,
        divr=0,
        psi_c=1.5,
        path=10000,
        scheme="QE",
        seed=None,
    ):
        """
        Conditional MC routine for 3/2 model
        Generate paths for vol only using QE discretization scheme.
        Compute integrated variance and get BSM prices vector for all strikes.

        Args:
            strike: strike price, in vector form
            spot: spot (or forward)
            texp: time to expiry
            sigma: initial volatility
            delta: length of each time step
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            psi_c: critical value for psi, lying in [1, 2]
            path: number of vol paths generated
            scheme: discretization scheme for vt, {'QE', 'TG', 'Euler', 'Milstein', 'KJ'}
            seed: random seed for rv generation

        Return:
            BSM price vector for all strikes
        """
        self.sigma = sigma
        self.bsm_model = pf.Bsm(self.sigma, intr=intr, divr=divr)
        self.delta = delta
        self.path = int(path)
        self.step = int(texp / self.delta)

        # xt = 1 / vt
        xt = 1 / self.sigma ** 2 * np.ones([self.path, self.step + 1])
        np.random.seed(seed)

        # equivalent kappa and theta for xt to follow a Heston model
        kappa_new = self.kappa * self.theta
        theta_new = (self.kappa + self.vov ** 2) / (self.kappa * self.theta)
        vov_new = -self.vov
        if scheme == "QE":
            u = np.random.uniform(size=(self.path, self.step))

            expo = np.exp(-kappa_new * self.delta)
            for i in range(self.step):
                # compute m, s_square, psi given xt(i)
                m = theta_new + (xt[:, i] - theta_new) * expo
                s2 = xt[:, i] * (vov_new ** 2) * expo * (
                    1 - expo
                ) / kappa_new + theta_new * (vov_new ** 2) * ((1 - expo) ** 2) / (
                    2 * kappa_new
                )
                psi = s2 / m ** 2

                # compute xt(i+1) given psi
                below = np.where(psi <= psi_c)[0]
                ins = 2 * psi[below] ** -1
                b2 = ins - 1 + np.sqrt(ins * (ins - 1))
                b = np.sqrt(b2)
                a = m[below] / (1 + b2)
                z = spst.norm.ppf(u[below, i])
                xt[below, i + 1] = a * (b + z) ** 2

                above = np.where(psi > psi_c)[0]
                p = (psi[above] - 1) / (psi[above] + 1)
                beta = (1 - p) / m[above]
                for k in range(len(above)):
                    if u[above[k], i] > p[k]:
                        xt[above[k], i + 1] = beta[k] ** -1 * np.log(
                            (1 - p[k]) / (1 - u[above[k], i])
                        )
                    else:
                        xt[above[k], i + 1] = 0

        elif scheme == "TG":
            if np.all(self.rx_results) == None:
                self.psi_points, self.rx_results = self.prepare_rx()

            expo = np.exp(-self.kappa * self.delta)
            for i in range(self.step):
                # compute m, s_square, psi given vt(i)
                m = theta_new + (xt[:, i] - theta_new) * expo
                s2 = xt[:, i] * (vov_new ** 2) * expo * (
                    1 - expo
                ) / kappa_new + theta_new * (vov_new ** 2) * ((1 - expo) ** 2) / (
                    2 * kappa_new
                )
                psi = s2 / m ** 2

                rx = np.array([self.find_rx(j) for j in psi])

                z = np.random.normal(size=(self.path, self.step))
                mu_v = np.zeros_like(z)
                sigma_v = np.zeros_like(z)
                mu_v[:, i] = rx * m / (spst.norm.pdf(rx) + rx * spst.norm.cdf(rx))
                sigma_v[:, i] = (
                    np.sqrt(s2)
                    * psi ** (-0.5)
                    / (spst.norm.pdf(rx) + rx * spst.norm.cdf(rx))
                )

                xt[:, i + 1] = np.fmax(mu_v[:, i] + sigma_v[:, i] * z[:, i], 0)

        elif scheme == "Euler":
            z = np.random.normal(size=(self.path, self.step))
            for i in range(self.step):
                xt[:, i + 1] = (
                    xt[:, i]
                    + kappa_new * (theta_new - np.max(xt[:, i], 0)) * self.delta
                    + vov_new * np.sqrt(np.max(xt[:, i], 0) * self.delta) * z[:, i]
                )

        elif scheme == "Milstein":
            z = np.random.normal(size=(self.path, self.step))
            for i in range(self.step):
                xt[:, i + 1] = (
                    xt[:, i]
                    + kappa_new * (theta_new - np.max(xt[:, i], 0)) * self.delta
                    + vov_new * np.sqrt(np.max(xt[:, i], 0) * self.delta) * z[:, i]
                    + vov_new ** 2 * 0.25 * (z[:, i] ** 2 - 1) * self.delta
                )

        elif scheme == "KJ":
            z = np.random.normal(size=(self.path, self.step))
            for i in range(self.step):
                xt[:, i + 1] = (
                    xt[:, i]
                    + kappa_new * theta_new * self.delta
                    + vov_new * np.sqrt(np.max(xt[:, i], 0) * self.delta) * z[:, i]
                    + vov_new ** 2 * 0.25 * (z[:, i] ** 2 - 1) * self.delta
                ) / (1 + kappa_new * self.delta)

        # compute integral of vt, equivalent spot and vol
        vt = 1 / xt
        below_0 = np.where(vt < 0)
        vt[below_0] = 0
        vt_int = spint.simps(vt, dx=self.delta)

        spot_cmc = spot * np.exp(
            self.rho
            / self.vov
            * (
                np.log(vt[:, -1] / vt[:, 0])
                - self.kappa
                * (self.theta * texp - vt_int * (1 + self.vov ** 2 * 0.5 / self.kappa))
            )
            - self.rho ** 2 * vt_int / 2
        )
        vol_cmc = np.sqrt((1 - self.rho ** 2) * vt_int / texp)

        # compute bsm price vector for the given strike vector
        price_cmc = np.zeros_like(strike)
        for j in range(len(strike)):
            price_cmc[j] = np.mean(
                self.bsm_model.price_formula(
                    strike[j], spot_cmc, vol_cmc, texp, intr=intr, divr=divr
                )
            )

        return price_cmc

    def prepare_rx(self):
        """
        Pre-calculate r(x) and store the result
        for TG scheme only
        """
        fx = (
            lambda rx: rx * spst.norm.pdf(rx)
            + spst.norm.cdf(rx)
            * (1 + rx ** 2)
            / ((spst.norm.pdf(rx) + rx * spst.norm.cdf(rx)) ** 2)
            - 1
        )
        rx_results = np.linspace(-2, 100, 10 ** 5)
        psi_points = fx(rx_results)

        return psi_points, rx_results

    def find_rx(self, psi):
        """
        Return r(psi) according to the pre_calculated results
        """

        if self.rx_results[self.psi_points >= psi].size == 0:
            print("Caution: input psi too large")
            return self.rx_results[-1]
        elif self.rx_results[self.psi_points <= psi].size == 0:
            print("Caution: input psi too small")
            return self.rx_results[0]
        else:
            return (
                self.rx_results[self.psi_points >= psi][0]
                + self.rx_results[self.psi_points <= psi][-1]
            ) / 2


          
class Sv32McAe2(sv32_mc2.Sv32McABC):
    def __init__(
        self,
        texp,
        sigma=1,
        rho=-0.5,
        theta=1.5,
        mr=2,
        vov=0.2,
        intr=0,
        divr=0,
        path_num=1000,
     ):
        super().__init__(sigma, vov, rho, mr, theta, intr=0.0, divr=0.0)
        self.path_num = path_num
        self.texp = texp
        """
        Initiate a 3/2 model
        Args:
            texp: time to expiry
            sigma: initial volatility
            rho: correlation between BMs of price and vol
            theta: long-term mean (equilibirum level) of the variance, strictly positive
            mr: speed of mean reversion
            vov: volatility of variance, strictly positive
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            path_num: number of vol paths generated
        Example:
        import numpy as np
        import pyfeng.sv32_mc as svae
        forward = 100
        strike = np.array([95,100,105])
        theta = 0.218
        rho = -0.99
        sigma = np.sqrt(0.06)
        texp = 0.5
        kappas = np.array([22.84, 18.32, 19.76, 20.48])
        vovs = np.array([8.56, 8.56, 3.2, 3.2])
        for i in range(len(kappas)):
            svae.Sv32McAe2(intr=r, texp=T, sigma=sigma, rho=rho, theta=theta, mr=kappas[i], vov=vovs[i], path_num = 1000)
            output1 = aemc.get_price(strike=Ks,spot=S0, texp=T, cp=1)
            table_for_AEMC.iloc[:,i] = output1[0]
        table_for_AEMC
        """ 
       
    def generate_XT(self):
        """
        Generate X_T, which is integral of Xt from t=0 to t=T
        Returns:
            X_T: np.ndarray
                the shape is (1,path_num)
        """
        X_0 = 1 / self.sigma ** 2
        delta = 4 * (self.mr + self.vov ** 2) / self.vov ** 2
        exp_term = np.exp(self.mr * self.theta * self.texp)
        c_T = self.vov ** 2 * (exp_term - 1) / (4 * self.mr * self.theta)    
        alpha = X_0 / c_T
        
        return (
            np.random.noncentral_chisquare(delta, alpha, size=self.path_num)
            / exp_term
            * c_T
        )
    
  
    def char_function(self, X_T):
        """
        This is the characteristic function for the integration of 1/Vt from t=0 to t=T
        Args:
            X_T: np.ndarray
                the shape is (1,path_num)
        Returns:
            It return the characteristic function with only one independent variable a.
        """
        n = 4 * (self.mr + self.vov ** 2) / self.vov ** 2
        j = -2 * self.mr * self.theta / self.vov ** 2
        vega = n / 2 - 1
        delta = self.vov ** 2 * self.texp / 4
        
        X_0 = 1 / self.sigma ** 2
        arg_in_Iv = j * np.sqrt(X_T * X_0) / np.sinh(j * delta)
        #besseli_ufun = np.frompyfunc(sv32_mc2.Sv32McABC.ivc, 2, 1)
        #We found that it's quicker and more stable to use mp.besseli. The value is small and it will get nan.
        #So we choose mp.besseli finally.
        besseli_ufun = np.frompyfunc(besseli, 2, 1)
        
        def char_func(a):
            order_1 = np.sqrt(vega ** 2 + 8 * a * (-1j) / self.vov ** 2)
            return besseli_ufun(order_1, arg_in_Iv) / besseli_ufun(vega, arg_in_Iv)

        return char_func
    
    def generate_VT(self, X_T): 
        """
        Generate V_T, which is integral of 1/Xt t=0 to t=T
        Args:
            X_T: np.ndarray
                the shape is (1,path_num)
        Returns:
            It returns log normal distribution with mean and variance formed by M1, M2
        """
        chfs = self.char_function(X_T)   
        M1 = derivative(chfs, x0=0, dx=0.00001, n=1)
        M2 = -derivative(chfs, x0=0, dx=0.00001, n=2)
        M1 = np.array(np.abs(M1).tolist(), dtype=float)
        M2 = np.array(np.abs(M2).tolist(), dtype=float)
        M2[np.isnan(M2)] = np.mean(M2)

        
        return spst.lognorm.rvs(M1, np.sqrt(np.log(M2 / M1 ** 2)))
    
    def cond_states(self, var_0, dt):
        '''
        Sample variance at maturity and conditional integrated variance
        Args:
            texp: float, time to maturity
        Returns:
            tuple, variance at maturity and conditional integrated variance
        '''
        var_final = self.generate_XT()
        var_mean=self.generate_VT(var_final)
        return var_final,var_mean
      
    def get_price(self, strike=1, spot=1, texp=1, cp=1):
        """
        Get option price
        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry
            cp: 1/-1 for call/put option
        Returns:
           price: float
        """
        X_0 = 1 / self.sigma ** 2
        price = self.price(strike, spot, texp, cp)
        return price
    
    
    
    
    
    
    
    
    
   