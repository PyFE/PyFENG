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


class Sv32McAe:
    def __init__(
        self,
        S0,
        Ks,
        T,
        r=0.05,
        sigma_0=1,
        beta=1,
        rho=-0.5,
        theta=1.5,
        kappa=2,
        vov=0.2,
        path_num=100,
        cp=1,
    ):
        assert beta == 1 or beta == 0, "Beta must be 0 or 1."
        self.S0 = S0
        if not isinstance(Ks, np.ndarray):
            self.Ks = np.array([Ks])
        else:
            self.Ks = Ks
        self.Ks = self.Ks.reshape(1, -1)
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
        """
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
        """
        self.simulate_sigma_2_T()
        self.simulate_U_T_version1()
        self.calForwardAndVolatility_version1()
        return self.calOutput()

    def simulate_sigma_2_T(self):
        """
        Generate path_num of sigma_T^2 with Heston Model,
        where self.X_T follow Heston Process, in which the parameters have been changed for 3/2 Model.
        And V_T are the inverses of X_T.
        Returns:
        None
        """
        self.X_T = self.simulate_Heston_outlayer()
        self.V_T = 1 / self.X_T

    def simulate_U_T_version1(self):
        """
        With characteristic function and path_num of sigma_T^2, we can get conditional distribution of U_T,
        where U_T is the integral of sigma_t^2 from t=0 to t=T.
        With conditional distribution, we will do one RNG on U_T with each sigma_T^2
        Returns:
            None
        """
        chfs = self.charFuncs()
        M1, M2 = self.calTwoMoments(chfs)
        mu, sigma = self.calLogNormalParas_version1(M1, M2)
        self.U_T = spst.lognorm.rvs(
            mu, sigma
        )  # Generate random numbers from lognormal distribution with mu and sigma

    def calForwardAndVolatility_version1(self):
        """
        Calculate E_F_T, expected forward price, and sigma_N or sigma_BS.
        Returns:
            None
        """
        internal_term = (
            self.rho
            / self.vov
            * (
                np.log(self.V_T / self.sigma_0 ** 2)
                - self.kappa * self.T * self.theta
                + ((self.kappa + self.vov ** 2 / 2) * self.U_T)
            )
        )
        if self.beta == 0:
            self.E_F_T = self.S0 + internal_term
            self.sigma_N = np.sqrt((1 - self.rho ** 2) * self.U_T / self.T)
            self.sigma_N = self.sigma_N.reshape(-1, 1)
        else:
            outside_term = self.rho ** 2 * self.U_T / 2
            self.E_F_T = self.S0 * np.exp(internal_term - outside_term)
            self.sigma_BS = np.sqrt((1 - self.rho ** 2) * self.U_T / self.T)
            self.sigma_BS = self.sigma_BS.reshape(-1, 1)
        self.E_F_T = self.E_F_T.reshape(-1, 1)

    def get_prices(self):
        """
        After running the simulation, we can get prices with this function, without run the simulation again.
        Returns:
            np.ndarray: Mean and Std of price output from AEMC
        """
        try:
            return self.option_prices, self.option_prices_std
        except:
            print(
                "Please run object function optionPrice_version1()/optionPrice_version2()/optionPrice_version3() first."
            )

    def calOutput(self):
        """
        Calculate the mean and std of price output from AEMC
        Returns:
            np.ndarray: Mean and Std of price output from AEMC
        """
        if self.beta == 0:
            prices = self.Bachelier(
                self.E_F_T, self.Ks, self.r, self.T, self.sigma_N, self.cp
            )
        else:
            prices = self.BSM(
                self.E_F_T, self.Ks, self.r, self.T, self.sigma_BS, self.cp
            )
        self.option_prices, self.option_prices_std = prices.mean(axis=0), prices.std(
            axis=0
        )
        return self.option_prices, self.option_prices_std

    @staticmethod
    def calLogNormalParas_version1(M1, M2):
        """
        Calculate mu and sigma of lognormal distribution for first moment and second moment.
        Args:
            M1: np.ndarray, first moment
            M2: np.ndarray, second moment

        Returns:
            M1: np.ndarray, mu of lognormal distribution
            np.sqrt(np.log(M2/M1**2)): np.ndarray, sigma of lognormal distribution
        """
        M1 = np.array(np.abs(M1).tolist(), dtype=float)
        M2 = np.array(np.abs(M2).tolist(), dtype=float)
        M2[np.isnan(M2)] = np.mean(M2)
        return M1, np.sqrt(np.log(M2 / M1 ** 2))

    @staticmethod
    def calTwoMoments(chfs):
        """
        Calculate two moments from the characteristic function.
        Args:
            chfs: function with only one arg_in

        Returns:
            M1: np.ndarray, first moment
            M2: np.ndarray, second moment
        """
        M1 = derivative(chfs, x0=0, dx=0.00001, n=1)
        M2 = -derivative(chfs, x0=0, dx=0.00001, n=2)
        return M1, M2

    def charFuncs(self):
        """
        This is the characteristic function for the integration of sigma_T^2 from t=0 to t=T
        Returns:
            It return the characteristic function with only one independent variable a.
        """
        n = 4 * (self.kappa + self.vov ** 2) / self.vov ** 2
        j = -2 * self.kappa * self.theta / self.vov ** 2
        vega = n / 2 - 1
        delta = self.vov ** 2 * self.T / 4
        X_0 = 1 / self.sigma_0 ** 2
        denominator = np.sinh(j * delta)
        arg_in_Iv = j * np.sqrt(self.X_T * X_0) / denominator
        besseli_ufun = np.frompyfunc(besseli, 2, 1)

        def char_func(a):
            order_1 = np.sqrt(vega ** 2 + 8 * a * (-1j) / self.vov ** 2)
            return besseli_ufun(order_1, arg_in_Iv) / besseli_ufun(vega, arg_in_Iv)

        return char_func

    def optionPrice_version2(self):
        """
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
        """
        self.simulate_sigma_2_T()
        M1 = self.simulate_M1()
        self.calForwardAndVolatility_version2(M1)
        return self.calOutput()

    def calForwardAndVolatility_version2(self, M1):
        """
        Calculate E_F_T, expected forward price, and sigma_N or sigma_BS.
        The slight difference with version 1 is that we substitute U_T with the mean of distribtion of U_T,
        which is M1 in this function.
        Returns:
            None
        """
        internal_term = (
            self.rho
            / self.vov
            * (
                np.log(self.V_T / self.sigma_0 ** 2)
                - self.kappa * self.T * self.theta
                + ((self.kappa + self.vov ** 2 / 2) * M1)
            )
        )
        if self.beta == 0:
            self.E_F_T = self.S0 + internal_term
            self.sigma_N = np.sqrt((1 - self.rho ** 2) * M1 / self.T)
            self.sigma_N = self.sigma_N.reshape(-1, 1)
        else:
            outside_term = self.rho ** 2 * M1 / 2
            self.E_F_T = self.S0 * np.exp(internal_term - outside_term)
            self.sigma_BS = np.sqrt((1 - self.rho ** 2) * M1 / self.T)
            self.sigma_BS = self.sigma_BS.reshape(-1, 1)
        self.E_F_T = self.E_F_T.reshape(-1, 1)

    def simulate_M1(self):
        """
        First, we need the characteristic function.
        Second, we direct use the first moment calculated from characteristic function as the proxy for U_T,
        as the RNG from conditional distribution have oscillating output,
        which is not good for the standard deviation of option pricing in AEMC approach.
        Returns:
            M1: first moment as the proxy for U_T
        """
        chfs = self.charFuncs()
        M1 = self.calOneMoment(chfs)
        M1 = np.array(np.abs(M1).tolist(), dtype=float)
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
        """
        Generate X_T for the preparation for V_T, which is sigma_T^2
        Returns:
            X_T: np.ndarray
                the shape is (1,path_num)
        """
        X_0 = 1 / self.sigma_0 ** 2
        delta = 4 * (self.kappa + self.vov ** 2) / self.vov ** 2
        c_T = self.c_function(self.T)
        alpha = X_0 / c_T
        exp_term = np.exp(self.kappa * self.theta * self.T)
        return (
            np.random.noncentral_chisquare(delta, alpha, size=self.path_num)
            / exp_term
            * c_T
        )

    def c_function(self, t):
        """
        c function in the paper: Exact Simulation of the 3/2 Model. The formula is in page-6
        Args:
            t: float
                t is the independent variable of c function

        Returns:
            float
                c(t)
        """
        return (
            self.vov ** 2
            * (np.exp(self.kappa * self.theta * t) - 1)
            / (4 * self.kappa * self.theta)
        )

    @staticmethod
    def BSM(S0, Ks, r, T, sigma, cp):
        """
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
        """
        prices = Bsm.price_formula(Ks, S0, sigma, T, cp, r)
        return prices

    @staticmethod
    def Bachelier(S0, Ks, r, T, sigma, cp):
        """
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
        """
        prices = Norm.price_formula(Ks, S0, sigma, T, cp, r)
        return prices

    def imp_vol_for_bsm(self, K, price):
        """
        This function is used to calculate the implied volatility in BSM model, given strike prices and option prices.
        spot price in given in self.S0
        """
        iv_func = (
            lambda _vol: self.BSM(self.S0, K, self.r, self.T, _vol, self.cp) - price
        )
        vol = spop.brentq(iv_func, 0, 10000)
        return vol

    def imp_vol_for_normal(self, K, price):
        """
        This function is used to calculate the implied volatility in Normal model, given strike prices and option prices.
        spot price in given in self.S0
        """
        iv_func = (
            lambda _vol: self.Bachelier(self.S0, K, self.r, self.T, _vol, self.cp)
            - price
        )
        vol = spop.brentq(iv_func, 0, 10000)
        return vol

    def impliedVolatility1(self):
        """
        Calculate the implied volatility with default strike prices, spot prices, and option price.
        Returns:
            output: np.ndarray, the shape is the same as the shape of strike prices.
        """
        output = []
        if self.beta == 1:
            for k, p in zip(self.Ks.ravel(), self.option_prices.ravel()):
                output.append(self.imp_vol_for_bsm(k, p))
        else:
            for k, p in zip(self.Ks.ravel(), self.option_prices.ravel()):
                output.append(self.imp_vol_for_normal(k, p))
        return np.array(output)

    def impliedVolatility2(self, model_type="bsm"):
        """
        Calculate the implied volatility with default strike prices, spot prices, and option price.
        Compared with impliedVolatility1, you can assign the type of implied volatility that you want.
        Returns:
            output: np.ndarray, the shape is the same as the shape of strike prices.
        """
        assert (
            model_type == "bsm" or model_type == "normal"
        ), "The model_type arg should be bsm or normal"
        output = []
        if model_type == "bsm":
            for k, p in zip(self.Ks.ravel(), self.option_prices.ravel()):
                output.append(self.imp_vol_for_bsm(k, p))
        else:
            for k, p in zip(self.Ks.ravel(), self.option_prices.ravel()):
                output.append(self.imp_vol_for_normal(k, p))
        return np.array(output)
