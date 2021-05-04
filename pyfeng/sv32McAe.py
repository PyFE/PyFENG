# This is the py-file for 3/2 almost exact Monte Carlo Simulation
# Author: Kaiwen CHEN and Jinze HE | 陈恺文 何金泽
import numpy as np
from scipy import stats as st
from scipy.misc import derivative
from mpmath import besseli
class Sv32McAe:
    def __init__(self, S0, Ks, T, r=0.05, sigma_0=1, beta=1, rho=-0.5, theta=1.5, kappa=2, vov=0.2, path_num = 100, cp=1):
        assert(beta==1 or beta==0), 'Beta must be 0 or 1.'
        self.S0 = S0
        if not isinstance(Ks, np.ndarray):
            self.Ks = np.array([Ks])
        else:
            self.Ks = Ks
        self.T = T
        self.r = r
        self.sigma_0 = sigma_0
        self.beta = beta
        self.rho = rho
        self.theta = theta
        self.kappa = kappa
        self.vov = vov
        self.path_num = path_num
        self.cp = cp

    def optionPrice_version1(self):
        self.simulate_sigma_2_T()
        self.simulate_U_T_version1()
        self.calForwardAndVolatility_version1()
        return self.calOutput()

    def simulate_sigma_2_T(self):
        self.X_T = self.simulate_Heston_outlayer()
        self.V_T = 1/self.X_T

    def simulate_U_T_version1(self):
        chfs = self.charFuncs_version1()
        M1, M2 = self.calTwoMoments_version1(chfs)
        mu, sigma = self.calLogNormalParas_version1(M1, M2)
        self.U_T = st.lognorm.rvs(mu, sigma)

    def calForwardAndVolatility_version1(self):
        internal_term = self.rho / self.vov * (
                    np.log(self.V_T / self.sigma_0 ** 2) - self.kappa * self.T * self.theta + (
                    (self.kappa + self.vov ** 2 / 2) * self.U_T
            ))
        if self.beta==0:
            self.E_F_T = self.S0+internal_term
            self.sigma_N = np.sqrt((1-self.rho**2)*self.U_T/self.T)
        else:
            outside_term = self.rho**2*self.U_T/2
            self.E_F_T = self.S0 * np.exp(internal_term-outside_term)
            self.sigma_BS = np.sqrt((1 - self.rho ** 2) * self.U_T / self.T)

    def calOutput(self):
        if self.beta == 0:
            prices = self.Bachelier(self.E_F_T, self.Ks, self.r, self.T, self.sigma_N, self.cp)
        else:
            prices = self.BSM(self.E_F_T, self.Ks, self.r, self.T, self.sigma_BS, self.cp)
        return prices.mean(), prices.std()

    @staticmethod
    def calLogNormalParas_version1(M1, M2):
        M1 = np.array(np.abs(M1).tolist(), dtype=float)
        M2 = np.array(np.abs(M2).tolist(), dtype=float)
        M2[np.isnan(M2)] = np.mean(M2)
        return M1, np.sqrt(np.log(M2/M1**2))

    @staticmethod
    def calTwoMoments_version1(chfs):
        M1 = derivative(chfs, x0=0, dx=0.00001, n=1)
        M2 = -derivative(chfs, x0=0, dx=0.00001, n=2)
        return M1, M2

    def charFuncs_version1(self):
        n = 4 * self.kappa * self.theta * (self.kappa + self.vov ** 2) / (self.vov ** 2 * self.kappa * self.theta)
        j = -2 * self.kappa * self.theta / self.vov ** 2
        vega = n / 2 - 1
        delta = self.vov ** 2 * self.T / 4
        X_0 = 1 / self.sigma_0 ** 2
        denominator = np.sinh(j*delta)
        arg_in_Iv = j*np.sqrt(self.X_T*X_0)/denominator
        besseli_ufun = np.frompyfunc(besseli, 2, 1)
        def char_func(a):
            order_1 = np.sqrt(vega ** 2 + 8 * a *(-1j)/ self.vov ** 2)
            return (besseli_ufun(order_1, arg_in_Iv)/besseli_ufun(vega, arg_in_Iv))
        return char_func

    def optionPrice_version2(self):
        self.simulate_sigma_2_T()
        M1 = self.simulate_M1()
        self.calForwardAndVolatility_version2(M1)
        return self.calOutput()

    def calForwardAndVolatility_version2(self, M1):
        internal_term = self.rho / self.vov * (
                np.log(self.V_T / self.sigma_0 ** 2) - self.kappa * self.T * self.theta + (
                (self.kappa + self.vov ** 2 / 2) * M1
        ))
        if self.beta == 0:
            self.E_F_T = self.S0 + internal_term
            self.sigma_N = np.sqrt((1 - self.rho ** 2) * M1 / self.T)
        else:
            outside_term = self.rho ** 2 * M1 / 2
            self.E_F_T = self.S0 * np.exp(internal_term - outside_term)
            self.sigma_BS = np.sqrt((1 - self.rho ** 2) * M1 / self.T)

    def simulate_M1(self):
        chfs = self.charFuncs_version1()
        M1 = self.calOneMoment(chfs)
        M1 = np.array(np.abs(M1).tolist(),dtype=float)
        return M1

    @staticmethod
    def calOneMoment(chfs):
        M1 = derivative(chfs, x0=0, dx=0.00001, n=1)
        return M1

    def optionPrice_version3(self):
        self.simulate_sigma_2_T()
        self.simulate_U_T_version3()
        self.calForwardAndVolatility_version1()
        return self.calOutput()

    def simulate_U_T_version3(self):
        chfs = self.charFuncs_version3()
        M1, M2 = self.calTwoMoments_version3(chfs)
        mu, sigma = self.calLogNormalParas_version1(M1, M2)
        self.U_T = st.lognorm.rvs(mu, sigma)

    @staticmethod
    def calTwoMoments_version3(chfs):
        M1 = -derivative(chfs, x0=0, dx=0.00001, n=1)
        M2 = derivative(chfs, x0=0, dx=0.00001, n=2)
        return M1, M2

    @staticmethod
    def calLogNormalParas_version3(M1, M2):
        M1 = np.array(M1, dtype=float)
        M2 = np.sqrt(np.log(np.array(M2/M1**2, dtype=float)))
        M2[np.isnan(M2)] = 0
        return M1, M2

    def charFuncs_version3(self):
        n = 4 * self.kappa * self.theta * (self.kappa + self.vov ** 2) / (self.vov ** 2 * self.kappa * self.theta)
        j = -2 * self.kappa * self.theta / self.vov ** 2
        vega = n / 2 - 1
        delta = self.vov ** 2 * self.T / 4
        X_0 = 1 / self.sigma_0 ** 2
        denominator = np.sinh(j*delta)
        arg_in_Iv = j*np.sqrt(self.X_T*X_0)/denominator
        besseli_ufun = np.frompyfunc(besseli, 2, 1)
        def char_func(a):
            order_1 = np.sqrt(vega ** 2 + 8 * a/ self.vov ** 2)
            return (besseli_ufun(order_1, arg_in_Iv)/besseli_ufun(vega, arg_in_Iv))
        return char_func

    def simulate_Heston_outlayer(self):
        X_0 = 1/self.sigma_0**2
        delta = 4 * (self.kappa + self.vov ** 2) / self.vov ** 2
        c_T = self.c_function(self.T)
        alpha = X_0 / c_T
        exp_term = np.exp(self.kappa*self.theta*self.T)
        return np.random.noncentral_chisquare(delta, alpha, size = self.path_num)/exp_term*c_T

    def c_function(self,t):
        return self.vov**2*(np.exp(self.kappa*self.theta*t)-1)/(4*self.kappa*self.theta)

    @staticmethod
    def BSM(S0,Ks,r,T,sigma,cp):
        sigma_T = np.fmax(sigma * np.sqrt(T),1e-16)
        d1 = (np.log(S0/Ks)+r*T)/sigma_T+sigma_T/2
        d2 = d1-sigma_T
        return cp*(S0*st.norm.cdf(d1*cp)-Ks*np.exp(r*T)*st.norm.cdf(d2*cp))

    @staticmethod
    def Bachelier(S0, Ks, r, T, sigma, cp):
        sigma_T = np.fmax(sigma * np.sqrt(T),1e-16)
        discount_factor = np.exp(-r*T)
        forward_price = S0/discount_factor
        d = (forward_price-Ks)/sigma_T
        return discount_factor*(cp*(forward_price-Ks)*st.norm.cdf(d)+st.norm.pdf(d*cp)*sigma_T)