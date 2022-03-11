# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:09:29 2019

"""

import math
import numpy as np


class Mgf2Mom:
    """
    Choudhury and Lucantoni (1996)'s algorithm for calculating moments from moment generating function (MGF)

    References:
        - Choudhury GL, Lucantoni DM (1996) Numerical Computation of the Moments of a Probability Distribution from its Transform. Operations Research 44:368–381. https://doi.org/10.1287/opre.44.2.368
    """

    mgf = None
    l = 1

    def __init__(self, mgf, l=1):
        self.mgf = mgf
        self.l = l

    @staticmethod
    def radius(n, gamma=11, l=1):
        rr = np.power(10, -gamma / (2 * n * l))
        return rr

    def moment_raw(self, n, alpha, l=None):
        """
        Raw moment of order n. Eq. (33) in the reference.

        Args:
            n: order
            alpha: alpha to use
            l: l

        Returns:

        """

        if l is None:
            l = self.l

        r_n = self.radius(n, l=l)
        denominator = 2.0 * n * l * (alpha*r_n) ** n
        coef_ = math.factorial(n) / denominator

        part1 = self.mgf(alpha*r_n).real
        part2 = (-1) ** n * self.mgf(-alpha*r_n).real

        #calc part3
        kk = np.arange(1, n*l)
        power1 = np.pi * kk * 1j / (n*l)
        power2 = -np.pi * kk * 1j / l
        #calc the value
        part3 = np.sum(np.real(self.mgf(alpha*r_n * np.exp(power1)) * np.exp(power2)))

        mu_n = coef_ * (part1 + part2 + 2.0 * part3)
        return mu_n

    def moments(self, n):
        '''
        Input:
            n: The order of the moment you want to get

        This function is modified based on algo3_mu which uses for loop.
        '''
        # Step 1:
        # pre-compute mu_1 and mu_2 with alpha=1 and l=1
        mu_1 = self.moment_raw(1, alpha=1, l=1)
        mu_2 = self.moment_raw(2, alpha=1.0/mu_1, l=1)

        # Step 2:
        # re-compute mu_1 and mu_2

        alpha_1 = 2.0 * mu_1 / mu_2
        mu = np.zeros(n)
        mu[0] = self.moment_raw(1, alpha=alpha_1)
        mu[1] = self.moment_raw(2, alpha=alpha_1)

        for i in range(3, n+1):
            #calc alpha_i
            alpha_i = (i - 1) * mu[i-3] / mu[i-2]
            #calc mu_i
            mu[i-1] = self.moment_raw(i, alpha=alpha_i)

        return mu
