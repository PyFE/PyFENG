# -*- coding: utf-8 -*-
"""
Created on Thur, Apr 29, 2021
Last modified on Fri, May 7, 2021
Conditional MC for Heston model based on QE discretization scheme by Andersen(2008)
@author: Xueyang & Xiaoyin
"""
import numpy as np
import pyfeng as pf
import scipy.stats as st
import scipy.integrate as spint
import scipy.optimize as sopt
from tqdm import tqdm


class HestonCondMcQE:
    '''
    Conditional MC for Heston model based on QE discretization scheme by Andersen(2008)

    Underlying price is assumed to follow a geometric Brownian motion.
    Volatility (variance) of the price is assumed to follow a CIR process.

    Example:
        >>> import numpy as np
        >>> import heston_cmc_qe as heston
        >>> strike = [100.0, 140.0, 70.0]
        >>> forward = 100
        >>> delta = [1, 1/2, 1/4, 1/8, 1/16, 1/32]
        >>> vov, kappa, rho, texp, theta, sigma = [1, 0.5, -0.9, 10, 0.04, 0.2]
        >>> heston_cmc_qe = heston.HestonCondMcQE(vov=vov, kappa=kappa, rho=rho, theta=theta)
        >>> price_cmc = np.zeros([len(delta), len(strike)])
        >>> for d in range(len(delta)):
        >>>     price_cmc[d, :] = heston_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], path=1e5, seed=123456)
        >>> price_cmc
        array([[14.52722285,  0.19584722, 37.20591415],
               [13.56691261,  0.26568546, 36.12295964],
               [13.22061601,  0.29003533, 35.9154245 ],
               [13.12057087,  0.29501411, 35.90207168],
               [13.1042753 ,  0.29476289, 35.89245755],
               [13.09047939,  0.29547721, 35.86410028]])
    '''

    def __init__(self, vov=1, kappa=0.5, rho=-0.9, theta=0.04):
        '''
        Initiate a Heston model

        Args:
            vov: volatility of variance, strictly positive
            kappa: speed of variance's mean-reversion, strictly positive
            rho: correlation between BMs of price and vol
            theta: long-term mean (equilibirum level) of the variance, strictly positive
        '''
        self.vov = vov
        self.kappa = kappa
        self.rho = rho
        self.theta = theta

        self.psi_points = None  # for TG scheme only
        self.rx_results = None
        self.dis = 1e-3

    def price(self, strike, spot, texp, sigma, delta, intr=0, divr=0, psi_c=1.5, path=10000, scheme='QE', seed=None):
        '''
        Conditional MC routine for Heston model
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
        '''
        self.sigma = sigma
        self.bsm_model = pf.Bsm(self.sigma, intr=intr, divr=divr)
        self.delta = delta
        self.path = int(path)
        self.step = int(texp / self.delta)

        vt = self.sigma ** 2 * np.ones([self.path, self.step + 1])
        np.random.seed(seed)

        if scheme == 'QE':
            u = np.random.uniform(size=(self.path, self.step))

            expo = np.exp(-self.kappa * self.delta)
            for i in range(self.step):
                # compute m, s_square, psi given vt(i)
                m = self.theta + (vt[:, i] - self.theta) * expo
                s2 = vt[:, i] * (self.vov ** 2) * expo * (1 - expo) / self.kappa + self.theta * (self.vov ** 2) * \
                     ((1 - expo) ** 2) / (2 * self.kappa)
                psi = s2 / m ** 2

                # compute vt(i+1) given psi
                below = np.where(psi <= psi_c)[0]
                ins = 2 * psi[below] ** -1
                b2 = ins - 1 + np.sqrt(ins * (ins - 1))
                b = np.sqrt(b2)
                a = m[below] / (1 + b2)
                z = st.norm.ppf(u[below, i])
                vt[below, i+1] = a * (b + z) ** 2

                above = np.where(psi > psi_c)[0]
                p = (psi[above] - 1) / (psi[above] + 1)
                beta = (1 - p) / m[above]
                for k in range(len(above)):
                    if u[above[k], i] > p[k]:
                        vt[above[k], i+1] = beta[k] ** -1 * np.log((1 - p[k]) / (1 - u[above[k], i]))
                    else:
                        vt[above[k], i+1] = 0

        elif scheme == 'TG':
            if np.all(self.rx_results) == None:
                self.psi_points, self.rx_results = self.prepare_rx()

            expo = np.exp(-self.kappa * self.delta)
            for i in range(self.step):
                # compute m, s_square, psi given vt(i)
                m = self.theta + (vt[:, i] - self.theta) * expo
                s2 = vt[:, i] * (self.vov ** 2) * expo * (1 - expo) / self.kappa + self.theta * (self.vov ** 2) * \
                     ((1 - expo) ** 2) / (2 * self.kappa)
                psi = s2 / m ** 2

                rx = np.array([self.find_rx(j) for j in psi])

                z = np.random.normal(size=(self.path, self.step))
                mu_v = np.zeros_like(z)
                sigma_v = np.zeros_like(z)
                mu_v[:, i] = rx * m / (st.norm.pdf(rx) + rx * st.norm.cdf(rx))
                sigma_v[:, i] = np.sqrt(s2) * psi ** (-0.5) / (st.norm.pdf(rx) + rx * st.norm.cdf(rx))

                vt[:, i+1] = np.fmax(mu_v[:, i] + sigma_v[:, i] * z[:, i], 0)

        elif scheme == 'Euler':
            z = np.random.normal(size=(self.path, self.step))
            for i in range(self.step):
                vt[:, i+1] = vt[:, i] + self.kappa * (self.theta - np.max(vt[:, i], 0)) * self.delta + \
                             self.vov * np.sqrt(np.max(vt[:, i], 0) * self.delta) * z[:, i]
            below_0 = np.where(vt < 0)
            vt[below_0] = 0

        elif scheme == 'Milstein':
            z = np.random.normal(size=(self.path, self.step))
            for i in range(self.step):
                vt[:, i+1] = vt[:, i] + self.kappa * (self.theta - np.max(vt[:, i], 0)) * self.delta + self.vov * \
                             np.sqrt(np.max(vt[:, i], 0) * self.delta) * z[:, i] + \
                             self.vov**2 * 0.25 * (z[:, i]**2 - 1) * self.delta
            below_0 = np.where(vt < 0)
            vt[below_0] = 0

        elif scheme == 'KJ':
            z = np.random.normal(size=(self.path, self.step))
            for i in range(self.step):
                vt[:, i+1] = (vt[:, i] + self.kappa * self.theta * self.delta + self.vov * \
                             np.sqrt(np.max(vt[:, i], 0) * self.delta) * z[:, i] + \
                             self.vov**2 * 0.25 * (z[:, i]**2 - 1) * self.delta) / (1 + self.kappa * self.delta)
            below_0 = np.where(vt < 0)
            vt[below_0] = 0

        # compute integral of vt, equivalent spot and vol
        vt_int = spint.simps(vt, dx=self.delta)
        spot_cmc = spot * np.exp(self.rho * (vt[:, -1] - vt[:, 0] - self.kappa * (self.theta * texp - vt_int))
                                 / self.vov - self.rho ** 2 * vt_int / 2)
        vol_cmc = np.sqrt((1 - self.rho ** 2) * vt_int / texp)

        # compute bsm price vector for the given strike vector
        price_cmc = np.zeros_like(strike)
        for j in range(len(strike)):
            price_cmc[j] = np.mean(self.bsm_model.price_formula(strike[j], spot_cmc, vol_cmc, texp, intr=intr, divr=divr))

        return price_cmc

    def prepare_rx(self):
        '''
        Pre-calculate r(x) and store the result
        for TG scheme only
        '''
        fx = lambda rx: rx * st.norm.pdf(rx) + st.norm.cdf(rx) * (1 + rx ** 2) / \
                        ((st.norm.pdf(rx) + rx * st.norm.cdf(rx)) ** 2) - 1
        rx_results = np.linspace(-2, 100, 10 ** 5)
        psi_points = fx(rx_results)

        return psi_points, rx_results

    def find_rx(self, psi):
        '''
        Return r(psi) according to the pre_calculated results
        '''

        if self.rx_results[self.psi_points >= psi].size == 0:
            print("Caution: input psi too large")
            return self.rx_results[-1]
        elif self.rx_results[self.psi_points <= psi].size == 0:
            print("Caution: input psi too small")
            return self.rx_results[0]
        else:
            return (self.rx_results[self.psi_points >= psi][0] + self.rx_results[self.psi_points <= psi][-1])/2


