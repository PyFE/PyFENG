# This is the py-file for 3/2 almost exact Monte Carlo Simulation
# Author: Kaiwen CHEN and Jinze HE | 陈恺文 何金泽
import numpy as np
from scipy import stats as st
from scipy.misc import derivative
from mpmath import besseli
import scipy.optimize as sopt
from .bsm import Bsm
from .norm import Norm
class Sv32McAe:
    def __init__(self, S0, Ks, T, r=0.05, sigma_0=1, beta=1, rho=-0.5, theta=1.5, kappa=2, vov=0.2, path_num = 100, cp=1):
        assert(beta==1 or beta==0), 'Beta must be 0 or 1.'
        self.S0 = S0
        if not isinstance(Ks, np.ndarray):
            self.Ks = np.array([Ks])
        else:
            self.Ks = Ks
        self.Ks = self.Ks.reshape(1,-1)
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
        '''
        version-1: This one follows Professor Choi's idea exactly.
        First, we will simulate sigma_T with Heston model.
        Second, we will use every simulated sigma_T with characteristic function to simulate U_T from conditional distribution,
        where U_T is the integral of sigma_t^2 from t=0 to t=T.
        Third, we use 100 (by default, we can change this parameter when we construct the Sv32McAe object) pairs of sigma_T and U_T
        to calculate the corresponding E_F_T, expected forward price, and sigma_N or sigma_BS
        Finally, we will use these pairs of (E_F_T, sigma_N or sigma_BS) to calculate corresponding option prices.

        Returns:
        A tuple with two elements.
        The first element is a np.ndarray, which includes all the means of prices under every strike price.
        That is to say, the shape of self.Ks is the same as the shape of the first element,
        and our calculation supports for vectorization on strike price, not for time to maturity.
        Likewise, the second element is a np.ndarray, too, which contains all the stds of prices under every strike price.
        '''
        self.simulate_sigma_2_T()
        self.simulate_U_T_version1()
        self.calForwardAndVolatility_version1()
        return self.calOutput()

    def simulate_sigma_2_T(self):
        '''
        Generate path_num of sigma_T^2 with Heston Model,
        where self.X_T follow Heston Process, in which the parameters have been changed for 3/2 Model.
        And V_T are the inverses of X_T.
        Returns:
        None
        '''
        self.X_T = self.simulate_Heston_outlayer()
        self.V_T = 1/self.X_T

    def simulate_U_T_version1(self):
        '''
        With characteristic function and path_num of sigma_T^2, we can get conditional distribution of U_T,
        where U_T is the integral of sigma_t^2 from t=0 to t=T.
        With conditional distribution, we will do one RNG on U_T with each sigma_T^2
        Returns:
            None
        '''
        chfs = self.charFuncs()
        M1, M2 = self.calTwoMoments(chfs)
        mu, sigma = self.calLogNormalParas_version1(M1, M2)
        self.U_T = st.lognorm.rvs(mu, sigma) # Generate random numbers from lognormal distribution with mu and sigma

    def calForwardAndVolatility_version1(self):
        '''
        Calculate E_F_T, expected forward price, and sigma_N or sigma_BS.
        Returns:
            None
        '''
        internal_term = self.rho / self.vov * (
                    np.log(self.V_T / self.sigma_0 ** 2) - self.kappa * self.T * self.theta + (
                    (self.kappa + self.vov ** 2 / 2) * self.U_T
            ))
        if self.beta==0:
            self.E_F_T = self.S0+internal_term
            self.sigma_N = np.sqrt((1-self.rho**2)*self.U_T/self.T)
            self.sigma_N = self.sigma_N.reshape(-1, 1)
        else:
            outside_term = self.rho**2*self.U_T/2
            self.E_F_T = self.S0 * np.exp(internal_term-outside_term)
            self.sigma_BS = np.sqrt((1 - self.rho ** 2) * self.U_T / self.T)
            self.sigma_BS = self.sigma_BS.reshape(-1, 1)
        self.E_F_T = self.E_F_T.reshape(-1,1)

    def get_prices(self):
        '''
        After running the simulation, we can get prices with this function, without run the simulation again.
        Returns:
            np.ndarray: Mean and Std of price output from AEMC
        '''
        try:
            return self.option_prices, self.option_prices_std
        except:
            print('Please run object function optionPrice_version1()/optionPrice_version2()/optionPrice_version3() first.')

    def calOutput(self):
        '''
        Calculate the mean and std of price output from AEMC
        Returns:
            np.ndarray: Mean and Std of price output from AEMC
        '''
        if self.beta == 0:
            prices = self.Bachelier(self.E_F_T, self.Ks, self.r, self.T, self.sigma_N, self.cp)
        else:
            prices = self.BSM(self.E_F_T, self.Ks, self.r, self.T, self.sigma_BS, self.cp)
        self.option_prices, self.option_prices_std = prices.mean(axis=0), prices.std(axis=0)
        return self.option_prices, self.option_prices_std

    @staticmethod
    def calLogNormalParas_version1(M1, M2):
        '''
        Calculate mu and sigma of lognormal distribution for first moment and second moment.
        Args:
            M1: np.ndarray, first moment
            M2: np.ndarray, second moment

        Returns:
            M1: np.ndarray, mu of lognormal distribution
            np.sqrt(np.log(M2/M1**2)): np.ndarray, sigma of lognormal distribution
        '''
        M1 = np.array(np.abs(M1).tolist(), dtype=float)
        M2 = np.array(np.abs(M2).tolist(), dtype=float)
        M2[np.isnan(M2)] = np.mean(M2)
        return M1, np.sqrt(np.log(M2/M1**2))

    @staticmethod
    def calTwoMoments(chfs):
        '''
        Calculate two moments from the characteristic function.
        Args:
            chfs: function with only one arg_in

        Returns:
            M1: np.ndarray, first moment
            M2: np.ndarray, second moment
        '''
        M1 = derivative(chfs, x0=0, dx=0.00001, n=1)
        M2 = -derivative(chfs, x0=0, dx=0.00001, n=2)
        return M1, M2

    def charFuncs(self):
        '''
        This is the characteristic function for the integration of sigma_T^2 from t=0 to t=T
        Returns:
            It return the characteristic function with only one independent variable a.
        '''
        n = 4 * (self.kappa + self.vov ** 2) / self.vov ** 2
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
        '''
        version-2: This one follows Professor Choi's idea exactly.
        First, we will simulate sigma_T with Heston model.
        Second, we direct use the first moment calculated from characteristic function as the proxy for U_T,
        as the RNG from conditional distribution have oscillating output,
        which is not good for the standard deviation of option pricing in AEMC approach.
        This is the innovation on insider layer simulation.
        Third, we use 100 (by default, we can change this parameter when we construct the Sv32McAe object) pairs of sigma_T and U_T
        to calculate the corresponding E_F_T, expected forward price, and sigma_N or sigma_BS
        Finally, we will use these pairs of (E_F_T, sigma_N or sigma_BS) to calculate corresponding option prices.

        Returns:
        A tuple with two elements.
        The first element is a np.ndarray, which includes all the means of prices under every strike price.
        That is to say, the shape of self.Ks is the same as the shape of the first element,
        and our calculation supports for vectorization on strike price, not for time to maturity.
        Likewise, the second element is a np.ndarray, too, which contains all the stds of prices under every strike price.
        '''
        self.simulate_sigma_2_T()
        M1 = self.simulate_M1()
        self.calForwardAndVolatility_version2(M1)
        return self.calOutput()

    def calForwardAndVolatility_version2(self, M1):
        '''
        Calculate E_F_T, expected forward price, and sigma_N or sigma_BS.
        The slight difference with version 1 is that we substitute U_T with the mean of distribtion of U_T,
        which is M1 in this function.
        Returns:
            None
        '''
        internal_term = self.rho / self.vov * (
                np.log(self.V_T / self.sigma_0 ** 2) - self.kappa * self.T * self.theta + (
                (self.kappa + self.vov ** 2 / 2) * M1
        ))
        if self.beta == 0:
            self.E_F_T = self.S0 + internal_term
            self.sigma_N = np.sqrt((1 - self.rho ** 2) * M1 / self.T)
            self.sigma_N = self.sigma_N.reshape(-1,1)
        else:
            outside_term = self.rho ** 2 * M1 / 2
            self.E_F_T = self.S0 * np.exp(internal_term - outside_term)
            self.sigma_BS = np.sqrt((1 - self.rho ** 2) * M1 / self.T)
            self.sigma_BS = self.sigma_BS.reshape(-1, 1)
        self.E_F_T = self.E_F_T.reshape(-1, 1)

    def simulate_M1(self):
        '''
        First, we need the characteristic function.
        Second, we direct use the first moment calculated from characteristic function as the proxy for U_T,
        as the RNG from conditional distribution have oscillating output,
        which is not good for the standard deviation of option pricing in AEMC approach.
        Returns:
            M1: first moment as the proxy for U_T
        '''
        chfs = self.charFuncs()
        M1 = self.calOneMoment(chfs)
        M1 = np.array(np.abs(M1).tolist(),dtype=float)
        return M1

    @staticmethod
    def calOneMoment(chfs):
        M1 = derivative(chfs, x0=0, dx=0.00001, n=1)
        return M1

    # def optionPrice_version3(self):
    #     self.simulate_sigma_2_T()
    #     self.simulate_U_T_version3()
    #     self.calForwardAndVolatility_version1()
    #     return self.calOutput()

    # def simulate_U_T_version3(self):
    #     chfs = self.charFuncs()
    #     M1, M2 = self.calTwoMoments(chfs)
    #     mu, scale = self.calInverseGaussianParas(M1, M2)
    #     self.U_T = st.lognorm.rvs(mu, 0, scale)

    # @staticmethod
    # def calInverseGaussianParas(M1, M2):
    #     M1 = np.array(np.abs(M1).tolist(), dtype=float)
    #     M2 = np.array(np.abs(M2).tolist(), dtype=float)
    #     M2 = np.sqrt(np.log(np.array(M2/M1**2, dtype=float)))
    #     M2[np.isnan(M2)] = 0
    #     scale = np.power(M1,3)/(M2-M1**2)
    #     mu = M1/scale
    #     return mu, scale

    def simulate_Heston_outlayer(self):
        '''
        Generate X_T for the preparation for V_T, which is sigma_T^2
        Returns:
            X_T: np.ndarray
                the shape is (1,path_num)
        '''
        X_0 = 1/self.sigma_0**2
        delta = 4 * (self.kappa + self.vov ** 2) / self.vov ** 2
        c_T = self.c_function(self.T)
        alpha = X_0 / c_T
        exp_term = np.exp(self.kappa*self.theta*self.T)
        return np.random.noncentral_chisquare(delta, alpha, size = self.path_num)/exp_term*c_T

    def c_function(self,t):
        '''
        c function in the paper: Exact Simulation of the 3/2 Model. The formula is in page-6
        Args:
            t: float
                t is the independent variable of c function

        Returns:
            float
                c(t)
        '''
        return self.vov**2*(np.exp(self.kappa*self.theta*t)-1)/(4*self.kappa*self.theta)

    @staticmethod
    def BSM(S0,Ks,r,T,sigma,cp):
        '''
        import from pyfeng package, to ues BSM model pricing formula
        Args:
            S0: Stock initial price
            Ks: Array of strike prices
            r: interest rate
            T: time to maturity of options
            sigma: volatility of stock
            cp: call or put

        Returns:
            prices: np,ndarray
        '''
        prices = Bsm.price_formula(Ks,S0,sigma,T,cp,r)
        return prices

    @staticmethod
    def Bachelier(S0, Ks, r, T, sigma, cp):
        '''
        import from pyfeng package, to ues Normal model pricing formula
        Args:
            S0: Stock initial price
            Ks: Array of strike prices
            r: interest rate
            T: time to maturity of options
            sigma: volatility of stock
            cp: call or put

        Returns:
            prices: np,ndarray
        '''
        prices = Norm.price_formula(Ks,S0,sigma,T,cp,r)
        return prices

    def imp_vol_for_bsm(self, K, price):
        '''
        This function is used to calculate the implied volatility in BSM model, given strike prices and option prices.
        spot price in given in self.S0
        '''
        iv_func = lambda _vol: \
            self.BSM(self.S0, K, self.r, self.T, _vol, self.cp) - price
        vol = sopt.brentq(iv_func, 0, 10000)
        return vol

    def imp_vol_for_normal(self, K, price):
        '''
        This function is used to calculate the implied volatility in Normal model, given strike prices and option prices.
        spot price in given in self.S0
        '''
        iv_func = lambda _vol: \
            self.Bachelier(self.S0, K, self.r, self.T, _vol, self.cp) - price
        vol = sopt.brentq(iv_func, 0, 10000)
        return vol

    def impliedVolatility1(self):
        '''
        Calculate the implied volatility with default strike prices, spot prices, and option price.
        Returns:
            output: np.ndarray, the shape is the same as the shape of strike prices.
        '''
        output = []
        if self.beta == 1:
            for k,p in zip(self.Ks.ravel(), self.option_prices.ravel()):
                output.append(self.imp_vol_for_bsm(k, p))
        else:
            for k, p in zip(self.Ks.ravel(), self.option_prices.ravel()):
                output.append(self.imp_vol_for_normal(k, p))
        return np.array(output)

    def impliedVolatility2(self, model_type='bsm'):
        '''
        Calculate the implied volatility with default strike prices, spot prices, and option price.
        Compared with impliedVolatility1, you can assign the type of implied volatility that you want.
        Returns:
            output: np.ndarray, the shape is the same as the shape of strike prices.
        '''
        assert model_type=='bsm' or  model_type=='normal', 'The model_type arg should be bsm or normal'
        output = []
        if model_type=='bsm':
            for k, p in zip(self.Ks.ravel(), self.option_prices.ravel()):
                output.append(self.imp_vol_for_bsm(k, p))
        else:
            for k, p in zip(self.Ks.ravel(), self.option_prices.ravel()):
                output.append(self.imp_vol_for_normal(k, p))
        return np.array(output)