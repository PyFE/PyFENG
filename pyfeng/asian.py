# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:29:04 2021

@author: cy-wang15 / Zay
"""

import math
import numpy as np
import mpmath as m
import sympy
from scipy.misc import derivative
import pyfeng.multiasset as ma
from . import opt_abc as opt
from . import nsvh


class BsmAsianJsu(ma.OptMaABC):
    """

    Johnson's SU distribution approximation for Asian option pricing under the BSM model.

    Note: Johnson's SU distribution is the solution of NSVh with NSVh with lambda = 1.

    References:
        [1] Posner, S. E., & Milevsky, M. A. (1998). Valuing exotic options by approximating the SPD
        with higher moments. The Journal of Financial Engineering, 7(2). https://ssrn.com/abstract=108539

        [2] Choi, J., Liu, C., & Seo, B. K. (2019). Hyperbolic normal stochastic volatility model.
        Journal of Futures Markets, 39(2), 186–204. https://doi.org/10.1002/fut.21967

    """

    def moments(self, spot, texp, n=1):
        """

        The moments of the Asian option with a log-normal underlying asset.

        Args:
            spot: spot price
            texp: time to expiry
            n: moment order

        References:
            Geman, H., & Yor, M. (1993). Bessel processes, Asian options, and perpetuities.
            Mathematical finance, 3(4), 349-375.

        Returns: the nth moment

        """
        lam = self.sigma[0]
        v = (self.intr - self.divr - lam ** 2 / 2) / lam
        beta = v / lam

        ds = []
        for j in range(n + 1):
            item0 = 2 ** n
            for i in range(n + 1):
                if i != j:
                    item0 *= ((beta + j) ** 2 - (beta + i) ** 2) ** (-1)
            ds.append(item0)
        item1 = 0
        for i in range(n + 1):
            item1 += ds[i] * np.exp((lam ** 2 * i ** 2 / 2 + lam * i * v) * texp)
        moment = (spot / texp) ** n * math.factorial(n) / lam ** (2 * n) * item1

        return moment

    def moment_mvsk(self, spot, texp):
        """

        Return mean, variance, skewness, kurtosis for Asian options.

        Args:
            spot: spot price
            texp: time to expiry

        Returns: mean, variance, skewness, kurtosis of Asian options

        """
        moments = []
        for i in range(1, 5):
            moments.append(self.moments(spot, texp, i))

        m1, m2, m3, m4 = moments[0], moments[1], moments[2], moments[3]
        mu = m1
        var = m2 - m1 ** 2
        skew = (m3 - m1 ** 3 - 3 * m2 * m1 + 3 * m1 ** 3) / var ** (3 / 2)
        kurt = (m4 - 3 * m1 ** 4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2) / var ** 2

        return mu, var, skew, kurt

    def price(self, strike, spot, texp, cp=1):
        """

        Asian options price.
        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
        Returns: Basket options price

        """
        df = np.exp(-texp * self.intr)

        mu, var, skew, kurt = self.moment_mvsk(spot, texp)

        m = nsvh.Nsvh1(sigma=self.sigma)
        m.calibrate_vsk(var, skew, kurt - 3, texp, setval=True)
        price = m.price(strike, mu, texp, cp)

        return df * price


class BsmAsianLinetsky2004(opt.OptABC):

    b = 1.0
    n_eig = 50

    def nu(self):
        nu = 2 * (self.intr - self.divr) / (self.sigma ** 2) - 1
        return nu

    def find_zeros_real(self):
        nu = self.nu()

        eigenval = []
        if nu <= -2:
            for n in range(2, math.floor(np.abs(nu) / 2) + 1 + 1):
                eigenvalue = np.abs(nu) - 2 * n + 2
                eigenval.append(eigenvalue)
        else:
            pass
        return np.array(eigenval)

    @staticmethod
    def positive_eigenvalue(eigval):
        eigval_p = []
        for i in range(len(eigval)):
            real_i = np.array(eigval[i], dtype=complex).real[0]
            if real_i > 0:
                eigval_p.append(real_i)
            else:
                pass
        return eigval_p

    def find_zeros_imag(self, n):
        nu = self.nu()

        p = sympy.symbols("p")
        eigenval = []
        for j in range(1, n):
            p_tilde = sympy.solve(
                p * (sympy.log(4 * p) - 1) - 2 * np.pi * (j + nu / 4 - 1 / 2), p
            )
            eigenval.append(p_tilde)
        eigenval = self.positive_eigenvalue(np.real(eigenval))
        eigenval = np.array(eigenval).transpose()
        return eigenval

    def eta_q(self, eigenval):
        nu = self.nu()

        f = lambda x: m.whitw((1 - nu) / 2, x / 2, 1 / (2 * self.b))
        return complex(-derivative(f, eigenval, dx=1e-12))

    def xi_p(self, eigenval):
        nu = self.nu()
        func = lambda x: m.whitw((1 - nu) / 2, complex(0, x / 2), 1 / (2 * self.b))
        return complex(derivative(func, eigenval, dx=1e-12))

    def price_element_imag(self, tau, p_value, k):
        nu = self.nu()

        p1 = m.exp(-(nu ** 2 + p_value ** 2) * tau / 2)
        p2 = (p_value * m.gamma(complex(nu / 2, p_value / 2))) / (
            4
            * self.xi_p(eigenval=p_value)
            * m.gamma(complex(1, p_value))
        )
        const = (2 * k) ** ((nu + 3) / 2) * m.exp(-1 / (4 * k))
        w1 = m.whitw(-(nu + 3) / 2, complex(0, p_value / 2), 1 / (2 * k))
        m1 = m.whitm((1 - nu) / 2, complex(0, p_value / 2), 1 / (2 * self.b))
        return p1 * p2 * const * w1 * m1

    def price_element_real(self, tau, q_value, k):
        nu = self.nu()

        p1 = m.exp(-(nu ** 2 - q_value ** 2) * tau / 2)
        p2 = (q_value * m.gamma((nu + q_value) / 2)) / (
            4 * self.eta_q(eigenval=q_value) * m.gamma(1 + q_value)
        )
        const = (2 * k) ** ((nu + 3) / 2) * m.exp(-1 / (4 * k))
        w1 = m.whitw(-(nu + 3) / 2, q_value / 2, 1 / (2 * k))
        m1 = m.whitm((1 - nu) / 2, q_value / 2, 1 / (2 * self.b))
        return p1 * p2 * const * w1 * m1

    def price(self, strike, spot, texp, cp=1):
        tau = self.sigma ** 2 * texp / 4
        k = tau * strike / spot

        p = self.find_zeros_imag(n=self.n_eig + 1)
        q = self.find_zeros_real()

        imaginary_terms = []
        for i in range(len(p)):
            imaginary_term = self.price_element_imag(tau=tau, p_value=p[i], k=k)
            imaginary_terms.append(complex(imaginary_term))

        real_terms = []
        for i in range(len(q)):
            real_term = self.price_element_real(tau=tau, q_value=q[i], k=k)
            real_terms.append(complex(real_term))

        P = np.sum(imaginary_terms) + np.sum(real_terms)

        fwd, df, divf = self._fwd_factor(spot, texp)

        # undiscounted put price
        price = (spot / tau) * P.real
        # undiscounted call price
        if cp > 0:
            # Eq. (3)
            if np.abs((self.intr - self.divr) * texp) < 1e-8:
                fac = 1 + (self.intr - self.divr) * texp / 2
            else:
                fac = (divf/df - 1) / ((self.intr - self.divr) * texp)
            price += spot * fac - strike

        return df * price