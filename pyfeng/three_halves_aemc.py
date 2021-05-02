# This is the py-file for 3/2 almost exact Monte Carlo Simulation
# Author: Kaiwen CHEN and Jinze HE | 陈恺文 何金泽
import numpy as np
import scipy.special as sp
from scipy import stats as st
from scipy.misc import derivative
class Three_Halves_AEMC_Model:
    def __init__(self, S0, Ks, T, r=0, sigma_0=0.2, beta=1, rho=0.6, theta=0.04, kappa=1, vov=0.6, path_num = 10000):
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
        self.sigma_2_T = np.zeros(path_num)


    def optionPrice(self):
        self.simulate_sigma_2_T()
        self.simulate_U_T()
        self.calForwardAndVolatility()
        return self.calOutput()

    def calOutput(self):
        if self.beta == 0:
            prices = self.BSM(self.S0, self.Ks, self.r, self.T, self.sigma_N, self.cp)
        else:
            prices = self.Bachelier(self.S0, self.Ks, self.r, self.T, self.sigma_BS, self.cp)
        return prices.mean(), prices.std()

    def calForwardAndVolatility(self):
        if self.beta==0:
            self.E_F_T = self.S0+self.rho/self.vov*(
                    np.log(self.sigma_2_T/self.sigma_0**2)+(
                    self.kappa+self.vov**2/2)*self.U_T-self.kappa*self.theta*self.T)
            self.sigma_N = np.sqrt((1-self.rho**2)*self.U_T/self.T)
        else:
            self.E_F_T = self.S0*np.exp(self.rho/self.vov*(
                    np.log(self.sigma_2_T/self.sigma_0**2)+(
                    self.kappa+self.vov**2/2+self.rho**2-1)*self.U_T-self.kappa*self.theta*self.T))
            self.sigma_BS = np.sqrt((1 - self.rho ** 2) * self.U_T / self.T)


    def simulate_U_T(self):
        chfs = self.charFuncs()
        M1, M2 = self.calTwoMoments(chfs)
        mu, sigma = self.calLogNormalParas(M1, M2)
        self.U_T = st.lognorm.rvs(mu, sigma)


    @staticmethod
    def calLogNormalParas(M1, M2):
        return M1, np.sqrt(np.log(M2/M1**2))

    @staticmethod
    def calTwoMoments(chfs):
        M1 = derivative(chfs, x0=0, dx=0.0000001, n=1)
        M2 = derivative(chfs, x0=0, dx=0.0000001, n=2)
        return M1, M2

    def charFuncs(self):
        n = 4 * self.kappa * self.theta * (self.kappa + self.vov ** 2) / (self.vov ** 2 * self.kappa * self.theta)
        j = -2 * self.kappa * self.theta / self.vov ** 2
        vega = n / 2 - 1
        delta = self.vov ** 2 * self.T / 4
        X_0 = 1 / self.sigma_0 ** 2
        denominator = np.sinh(j*delta)
        arg_in_Iv = j*np.sqrt(self.X_T*X_0)/denominator
        def char_func(a):
            order_1 = np.sqrt(vega ** 2 + 8 * a / self.vov ** 2)
            return sp.iv(order_1, arg_in_Iv)/sp.iv(vega, arg_in_Iv)
        return char_func


    def simulate_sigma_2_T(self):
        self.X_T = self.simulate_Heston_outlayer()
        self.V_T = 1/self.X_T

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
        return discount_factor(cp*(forward_price-Ks)*st.norm.cdf(d)+st.norm.pdf(d*cp)*sigma_T)
