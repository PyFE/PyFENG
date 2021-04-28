# This is the py-file for 3/2 almost exact Monte Carlo Simulation
# Author: Kaiwen CHEN and Jinze HE | 陈恺文 何金泽
class Three_Halves_AEMC_Model:
    def __init__(self, S0, K, T, r = 0, rho = 0.6, theta = 0.2, kappa = 1, epsilon = 0.1, sigma_0 = 0.2):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.rho = rho
        self.theta = theta
        self.kappa = kappa
        self.epsilon = epsilon
        self.sigma_0 = sigma_0

    def draw_sigma_T(self):
        pass

    def charater_func(self):
        pass

    def find_1st_deri(self):
        pass

    def find_2nd_deri(self):
        pass

    def cal_cdf_V_T(self):
        pass

    def draw_V_T(self):
        pass

    @staticmethod
    def BSM(S0,K,r,T,sigma):
        pass

    @staticmethod
    def Bacheliar(S0,K,r,T,sigma):
        pass

    def option_price(self):
        pass