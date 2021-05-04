# -*- coding: utf-8 -*-
"""
Created on Thur, Apr 29, 2021
Conditional MC for 3/2 model based on QE discretization scheme by Andersen(2008)
@author: xueyang & xiaoyin
"""
import numpy as np
import pyfeng as pf
import scipy.stats as st
import scipy.integrate as spint
from tqdm import tqdm

class ThreehalfQECondMC:
    '''
    Conditional MC for Heston model based on QE discretization scheme by Andersen(2008)
    Adjusting the model's parameters for 3/2 model
    Underlying price is assumed to follow a geometric Brownian motion.
    Volatility (variance) of the price is assumed to follow a CIR process.

    Example:
        >>> import numpy as np
        >>> strike = [100.0, 140.0, 70.0]
        >>> forward = 100
        >>> delta = [1, 1/2, 1/4, 1/8, 1/16, 1/32]
        >>> vov, kappa, rho, texp, theta, sigma = [1, 0.5, -0.9, 10, 0.04, 0.2]
        >>> threehalf_cmc_qe = threehalf.ThreehalfQECondMC(vov=vov, kappa=kappa, rho=rho, theta=theta)
        >>> price_cmc = np.zeros([len(delta), len(strike)])
        >>> for d in range(len(delta)):
        >>>     price_cmc[d, :] = threehalf_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], path=1e5, seed=123456)
        >>> price_cmc
        array([[22.95314785 10.44093783 38.98364955],
               [23.2425543  10.67162543 39.26731165],
               [23.20965635 10.64143576 39.21865023],
               [22.93527518 10.4758762  38.87971674],
               [22.9298084  10.47613694 38.88556212],
               [23.12806844 10.56484306 39.16893668]])
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

    def price(self, strike, spot, texp, sigma, delta, intr=0, divr=0, psi_c=1.5, path=10000, seed=None):
        '''
        Conditional MC routine for Heston model
        Generate paths for vol only using QE discretization scheme.
        Compute integrated variance and get BSM prices vector for all strikes.

        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry
            sigma: initial volatility
            delta: length of each time step
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            psi_c: critical value for psi, lying in [1, 2]
            path: number of vol paths generated
            seed: random seed for rv generation

        Return:
            BSM price vector for all strikes
        '''
        self.sigma = sigma
        self.bsm_model = pf.Bsm(self.sigma, intr=intr, divr=divr)
        self.delta = delta
        self.path = int(path)
        self.step = int(texp / self.delta)

        xt = self.sigma ** -2 * np.ones([self.path, self.step + 1])
        vt = np.zeros_like(xt)
        np.random.seed(seed)
        u = np.random.uniform(size=(self.path, self.step))

        '''
        For the 3/2 model, first simulate xt using heston model,
        Adjusted args: 
            kappa_ = self.kappa * self.theta
            vov_ = -self.vov
            theta_ = (self.kappa + self.vov ** 2) / (self.kappa * self.theta)
            Others remains unchanged
        Then calculate real 3/2 model vt = 1 / xt
        '''
        kappa_ = self.kappa * self.theta
        vov_ = -self.vov
        theta_ = (self.kappa + self.vov ** 2) / (self.kappa * self.theta)

        expo = np.exp(-kappa_ * self.delta)
        # for i in tqdm(range(self.step)):
        for i in range(self.step):
            # compute m, s_square, psi given vt(i)
            m = theta_ + (xt[:, i] - theta_) * expo
            s2 = xt[:, i] * (vov_ ** 2) * expo * (1 - expo) / kappa_ + theta_ * (vov_ ** 2) * \
                 ((1 - expo) ** 2) / (2 * kappa_)
            psi = s2 / m ** 2

            # compute vt(i+1) given psi
            below = np.where(psi <= psi_c)[0]
            ins = 2 * psi[below] ** -1
            b2 = ins - 1 + np.sqrt(ins * (ins - 1))
            b = np.sqrt(b2)
            a = m[below] / (1 + b2)
            z = st.norm.ppf(u[below, i])
            xt[below, i+1] = a * (b + z) ** 2

            above = np.where(psi > psi_c)[0]
            p = (psi[above] - 1) / (psi[above] + 1)
            beta = (1 - p) / m[above]
            for k in range(len(above)):
                if u[above[k], i] > p[k]:
                    xt[above[k], i+1] = beta[k] ** -1 * np.log((1 - p[k]) / (1 - u[above[k], i]))
                else:
                    xt[above[k], i+1] = 0

        '''
        Calculating vt = 1 / xt
        If vt == float('inf') , let vt = 99999
        '''
        vt = 1 / xt
        infloc = np.where(vt == float('inf'))
        vt[infloc] = 999999

        # compute integral of vt, equivalent spot and vol
        vt_int = spint.simps(vt, dx=self.delta)
        spot_cmc = spot * np.exp(self.rho / self.vov * ((np.log(vt[:, -1]) - np.log(vt[:, 0])) - self.kappa * (self.theta * texp - \
                        (1 + self.vov ** 2 / (2 * self.kappa)) * vt_int))  - self.rho ** 2 * vt_int / 2)
        vol_cmc = np.sqrt((1 - self.rho ** 2) * vt_int / texp)

        # compute bsm price vector for the given strike vector
        price_cmc = np.zeros_like(strike)
        for j in range(len(strike)):
            price_cmc[j] = np.mean(self.bsm_model.price_formula(strike[j], spot_cmc, vol_cmc, texp))
        return price_cmc
