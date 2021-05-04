# -*- coding: utf-8 -*-

"""
Created on Mon, May 3, 2021
Almost exact MC for Heston model
@author: Shang Chencheng & Ning Lei
"""

import numpy as np

from scipy.special import iv
from scipy.misc import derivative
from scipy.stats import invgauss


class HestonMCAe:
    """
    Almost exact MC for Heston model.

    Underlying price is assumed to follow a geometric Brownian motion.
    Volatility (variance) of the price is assumed to follow a CIR process.
    Example:
        >>> strike = 100
        >>> spot = 100
        >>> vov, kappa, rho, texp, theta, sigma = 0.61, 6.21, -0.7, 1, 0.019, 0.10201
        >>> heston_ae = HestonMCAe(vov, kappa, rho, theta, r)
        >>> price_ae = heston_ae.price(strike, spot, texp, sigma_0, intr=0, divr=0)
        >>> price_ae
        8.946951375550809
    """
    def __init__(self, vov=1, kappa=0.5, rho=-0.9, theta=0.04, r=0):
        """
        Initiate a Heston model

        Args:
            vov: volatility of variance, strictly positive
            kappa: speed of variance's mean-reversion, strictly positive
            rho: correlation between BMs of price and vol
            theta: long-term mean (equilibirum level) of the variance, strictly positive
            r：the drift item
        """
        self.vov = vov
        self.theta = theta
        self.kappa = kappa
        self.rho = rho
        self.r = r

    def ch_f(self, texp, sigma_0, sigma_t, chi_dim):
        """

        Args:
            texp: time to expiry
            sigma_0: initial volatility
            sigma_t: volatility at time T
            chi_dim: dimensions of chisquare distribution

        Returns:
            ch_f: characteristic function of the distribution of integral sigma_t

        """
        gamma_f = lambda a: np.sqrt(self.kappa ** 2 - 2 * self.vov**2 * a * 1j)

        temp_f = lambda a: gamma_f(a) * texp

        ch_f_part_1 = lambda a: gamma_f(a) * np.exp(-0.5 * (temp_f(a) - self.kappa * texp)) \
                                * (1 - np.exp(-self.kappa * texp)) / (self.kappa * (1 - np.exp(-temp_f(a))))

        ch_f_part_2 = lambda a: np.exp((sigma_0 + sigma_t) / self.vov ** 2 * \
                                       (self.kappa * (1 + np.exp(-self.kappa * texp)) / (1 - np.exp(-self.kappa * texp))
                                        - gamma_f(a) * (1 + np.exp(-temp_f(a))) / (1 - np.exp(-temp_f(a)))))

        ch_f_part_3 = lambda a: iv(0.5 * chi_dim - 1, np.sqrt(sigma_0 * sigma_t) * 4 * gamma_f(a) *
                                   np.exp(-0.5 * temp_f(a)) / (self.vov ** 2 * (1 - np.exp(-temp_f(a))))) / \
                                iv(0.5 * chi_dim - 1, np.sqrt(sigma_0 * sigma_t) * 4 * self.kappa *
                                   np.exp(-0.5 * self.kappa * texp) / (
                                           self.vov ** 2 * (1 - np.exp(- self.kappa * texp))))

        ch_f = lambda a: ch_f_part_1(a) * ch_f_part_2(a) * ch_f_part_3(a)
        return ch_f

    def gen_vov_t(self, chi_dim, chi_lambda, texp, n_paths):
        """

        Args:
            chi_dim: dimensions of chisquare distribution
            chi_lambda: the skewing item of chisquare distribution
            texp: time to expiry
            n_paths: number of vol paths generated

        Returns:
            sigma_t: volatility at time T

        """
        cof = self.vov ** 2 * (1 - np.exp(-self.kappa * texp)) / (4 * self.kappa)
        sigma_t = cof * np.random.noncentral_chisquare(chi_dim, chi_lambda, n_paths)
        return sigma_t

    def gen_s_t(self, spot, sigma_t, sigma_0, texp, integral_sigma_t, n_paths):
        """

        Args:
            spot: spot (or forward)
            sigma_t: volatility at time T
            sigma_0: initial volatility
            texp: time to expiry
            integral_sigma_t: samples from the distribution of integral sigma_t
            n_paths: number of vol paths generated

        Returns:
            s_t: stock price at time T
        """

        integral_sqrt_sigma_t = (sigma_t - sigma_0 - self.kappa * self.theta * texp + self.kappa * integral_sigma_t)\
                                / self.vov
        mean = np.log(spot) + (self.r * texp - 0.5 * integral_sigma_t + self.rho * integral_sqrt_sigma_t)
        sigma_2 = (1 - self.rho ** 2) * integral_sigma_t
        s_t = np.exp(mean + np.sqrt(sigma_2) * np.random.normal(size=n_paths))
        return s_t

    def price(self, strike, spot, texp, sigma_0, intr=0, divr=0, n_paths=10000, seed=None,
              dis_can="Inverse-Gaussian", call=1):
        """
        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry
            sigma_0: initial volatility
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            n_paths: number of vol paths generated
            seed: random seed for rv generation
        Returns:
            price_ae: option price
        """
        if seed:
            np.random.seed(seed)

        chi_dim = (4 * self.theta * self.kappa) / (self.vov ** 2)
        chi_lambda = (4 * self.kappa * np.exp(-self.kappa * texp)) / \
                     ((self.vov ** 2) * (1 - np.exp(-self.kappa * texp))) * sigma_0

        sigma_t = self.gen_vov_t(chi_dim, chi_lambda, texp, n_paths)

        ch_f = self.ch_f(texp, sigma_0, sigma_t, chi_dim)

        moment_1st = (derivative(ch_f, 0, n=1, dx=1e-5) / 1j).real
        moment_2st = (derivative(ch_f, 0, n=2, dx=1e-5) / (1j ** 2)).real

        if dis_can == "Inverse-Gaussian":
            scale_ig = moment_1st**3 / (moment_2st - moment_1st**2)
            miu_ig = moment_1st / scale_ig
            integral_sigma_t = invgauss.rvs(miu_ig, scale=scale_ig)
            s_t = self.gen_s_t(spot, sigma_t, sigma_0, texp, integral_sigma_t, n_paths)

        elif dis_can == "Log-normal":
            scale_ln = np.sqrt(np.log(moment_2st) - 2 * np.log(moment_1st))
            miu_ln = np.log(moment_1st) - 0.5 * scale_ln ** 2
            integral_sigma_t = np.random.lognormal(miu_ln, scale_ln)
            s_t = self.gen_s_t(spot, sigma_t, sigma_0, texp, integral_sigma_t, n_paths)
        else:
            print("This function is not currently a candidate function!")
            return -1

        if call:
            price_ae = np.fmax(s_t - strike, 0).mean()
        else:
            price_ae = np.fmax(strike - s_t, 0).mean()

        return price_ae
