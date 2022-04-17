import sympy as sp
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import scipy
import matplotlib.pyplot as plt


class ApproxMethod:
    """
    Luca Capriotti (2018).etc applied the Exponent Expansion (EE) to
    the approximation of transition densities of IGBM,
    say GARCH diffusion process.
    """
    def __init__(self, a, b, sigma):
        self.a = a
        self.b = b
        self.sigma = sigma

    def w_discard(self, x, x0, t, n: int = 4):
        """
        W(x,x0,T) is the correction for the ansatz transition desities.
        Python is not as powerful as Matlab to solve this integration.
        """
        # Here w0 is the result of GARCH process
        wn_arr = []
        u, v = sp.symbols('u, v')
        w0 = 1/self.sigma**2 * (self.a * self.b * (sp.exp(-u)-sp.exp(-x0)) +
                                (self.a + 0.5*self.sigma**2)*(u-x0))
        mu = self.a * (self.b - sp.exp(u))/sp.exp(u) - 0.5 * self.sigma**2

        wn_arr.append(w0)
        w_res = float(w0.evalf(subs={u: x}))
        for i in range(1, n+1):
            print('n=', i, '\n')
            lmda_1 = 0.5 * self.sigma ** 2 * sp.diff(wn_arr[i-1], u, 2) - mu * sp.diff(wn_arr[i-1], u)
            lmda_2 = 0
            for j in range(i):
                lmda_2 += sp.diff(wn_arr[j], u) * sp.diff(wn_arr[i - 1 - j], u)
            if i == 1:
                lmda = lmda_1 - 0.5 * self.sigma**2 * lmda_2 + sp.diff(mu, u)
            else:
                lmda = lmda_1 - 0.5 * self.sigma ** 2 * lmda_2

            wn_arr.append(sp.integrate(v ** (i - 1) * lmda.subs(u, x0 + (u - x0) * v), (v, 0, 1)))
            w_res += wn_arr[i].evalf(subs={u: x}) * (t**i)
            print(w_res)
        self.wn_arr = wn_arr

        return w_res

    def w_for_density(self, x, x0, t, n: int = 4):
        """
        Exponent Expansion on the nth order
        to approximately get W(x, x0, t) of transition density.

        Parameters
        ----------
        x : float
            fractile point in terms of transformation, xt = ln(yt).
        x0 : float
            initial point of x.
        t : float
            time to expiry.
        n : int
            the highest order for Exponent Expansion.
        Returns
        -------
        w_res : float
            w = symsum(wn*t**n, 0, inf)
        """
        a = self.a
        b = self.b
        sigma = self.sigma

        w0 = ((x - x0)*(sigma**2/2 + a) + a*b*(np.exp(-x) - np.exp(-x0)))/sigma**2
        if n > 0.5:
            w1 = (
                a/2 + sigma**2/8 + a**2/(2*sigma**2) - (a**2*b**2*np.exp(-2*x))/(4*(sigma**2*x - sigma**2*x0)) +
                (a**2*b**2*np.exp(-2*x0))/(4*(sigma**2*x - sigma**2*x0)) + (a*b*np.exp(-x))/(x - x0) -
                (a*b*np.exp(-x0))/(x - x0) + (a**2*b*np.exp(-x))/(sigma**2*x - sigma**2*x0) -
                (a**2*b*np.exp(-x0))/(sigma**2*x - sigma**2*x0)
            )
            w_res = w0 + w1 * t
        if n > 1:
            w2 = (
                (a**2*b**2*sigma**2*(np.exp(-2*x)/(2*sigma**2*(x - x0)**2) - np.exp(-2*x0)/(2*sigma**2*(x - x0)**2)))/2 -
                (a**2*b*sigma**2*(np.exp(-x)/(sigma**2*(x - x0)**2) - np.exp(-x0)/(sigma**2*(x - x0)**2)))/2 -
                (a*b*sigma**2*(np.exp(-x) - np.exp(-x0)))/(2*(x - x0)**2) -
                (a*b*np.exp(-2*x0)*(np.exp(2*x0 - x)*(4*sigma**2 + 4*a) - np.exp(x0)*(4*a + 4*a*x0 + 4*sigma**2*x0 + 4*sigma**2) +
                                 2*a*b*x0 + 2*a*b*np.exp(x0 - x)*(np.exp(x - x0)/2 - np.exp(x0 - x)/2)))/(4*(x - x0)**3) -
                (a*b*x*np.exp(-2*x0)*(4*np.exp(x0)*sigma**2 - 2*a*b + 4*a*np.exp(x0)))/(4*(x - x0)**3)
            )
            w_res = w0 + w1 * t + w2 * t ** 2
        if n > 2:
            w3 = (
                np.exp(4 * x0 - 4 * x) * ((a ** 4 * b ** 4 * np.exp(-4 * x0)) / (32 * sigma ** 2 * (x - x0) ** 3) +
                                       (a ** 4 * b ** 4 * np.exp(-4 * x0)) / (32 * sigma ** 2 * (x - x0) ** 4)) +
                np.exp(2 * x0 - 2 * x) * ((a ** 2 * b ** 2 * np.exp(-4 * x0) *
                                        (- a ** 2 * b ** 2 + 4 * np.exp(x0) * a ** 2 * b + 8 * np.exp(2 * x0) * a ** 2 +
                                         4 * np.exp(x0) * a * b * sigma ** 2 + 16 * np.exp(2 * x0) * a * sigma ** 2 +
                                         8 * np.exp(2 * x0) * sigma ** 4)) / (16 * sigma ** 2 * (x - x0) ** 4) +
                                       (a ** 2 * b ** 2 * np.exp(-2 * x0) * (sigma ** 2 + a) ** 2) /
                                       (4 * sigma ** 2 * (x - x0) ** 3)) - np.exp(3 * x0 - 3 * x) *
                ((a ** 3 * b ** 3 * np.exp(-3 * x0) * (sigma ** 2 + a)) / (6 * sigma ** 2 * (x - x0) ** 3) +
                 (a ** 3 * b ** 3 * np.exp(-3 * x0) * (sigma ** 2 + a)) / (4 * sigma ** 2 * (x - x0) ** 4)) +
                (a ** 4 * b ** 4 - 8 * np.exp(x0) * a ** 4 * b ** 3 + 16 * np.exp(2 * x0) * a ** 4 * b ** 2 -
                 8 * np.exp(x0) * a ** 3 * b ** 3 * sigma ** 2 + 32 * np.exp(2 * x0) * a ** 3 * b ** 2 * sigma ** 2 +
                 16 * np.exp(2 * x0) * a ** 2 * b ** 2 * sigma ** 4) /
                (32 * sigma ** 2 * x ** 4 * np.exp(4 * x0) + 32 * sigma ** 2 * x0 ** 4 * np.exp(4 * x0) +
                 192 * sigma ** 2 * x ** 2 * x0 ** 2 * np.exp(4 * x0) - 128 * sigma ** 2 * x * x0 ** 3 * np.exp(4 * x0) -
                 128 * sigma ** 2 * x ** 3 * x0 * np.exp(4 * x0)) +
                ((a * b * sigma ** 2 * np.exp(-2 * x0) *
                  (np.exp(x0) * (6 * a * x - 6 * a * x0 + 6 * sigma ** 2 * x - 6 * sigma ** 2 * x0) +
                   np.exp(2 * x0 - x) * (6 * a * x - 6 * a * x0 + 6 * sigma ** 2 * x - 6 * sigma ** 2 * x0) -
                   6 * a * b * x * np.cosh(x - x0) * np.exp(x0 - x) + 6 * a * b * x0 * np.cosh(x - x0) * np.exp(x0 - x))) / 4 +
                 (a * b * sigma ** 2 * np.exp(-2 * x0) * (np.exp(2 * x0 - x) * (12 * sigma ** 2 + 12 * a) - np.exp(x0) * (
                    12 * sigma ** 2 + 12 * a) + 6 * a * b * np.sinh(x - x0) * np.exp(x0 - x))) / 4) / (x - x0) ** 5 + (
                            a * b * sigma ** 4 * (np.exp(-x) - np.exp(-x0))) / (4 * (x - x0) ** 3) - (
                            a ** 2 * b ** 2 * np.exp(-4 * x0) * (
                                3 * a ** 2 * b ** 2 - 16 * np.exp(x0) * a ** 2 * b + 24 * np.exp(2 * x0) * a ** 2 - 16 * np.exp(
                            x0) * a * b * sigma ** 2 + 48 * np.exp(2 * x0) * a * sigma ** 2 + 24 * np.exp(
                            2 * x0) * sigma ** 4)) / (96 * sigma ** 2 * (x - x0) ** 3) + (
                            a * b * sigma ** 2 * np.exp(-2 * x0) * (
                                a * x ** 2 * np.exp(2 * x0 - x) + a * x0 ** 2 * np.exp(2 * x0 - x) - 2 * a * x * x0 * np.exp(
                            2 * x0 - x) - a * b * x ** 2 * np.exp(2 * x0 - 2 * x) - a * b * x0 ** 2 * np.exp(
                            2 * x0 - 2 * x) + 2 * a * b * x * x0 * np.exp(2 * x0 - 2 * x))) / (4 * (x - x0) ** 5) + (
                            a ** 2 * b * sigma ** 2 * np.exp(-2 * x0) * (b - np.exp(x0))) / (4 * (x - x0) ** 3) - (
                            a ** 2 * b ** 2 * np.exp(- x - 2 * x0) * (sigma ** 2 + a) * (
                                4 * np.exp(x0) * sigma ** 2 - a * b + 4 * a * np.exp(x0))) / (4 * sigma ** 2 * (x - x0) ** 4)

            )
            w_res = w0 + w1 * t + w2 * t ** 2 + w3 * t ** 3
        if n > 3:
            w4 = (
                (
                            4 * a ** 4 * b ** 4 * x0 - 4 * a ** 4 * b ** 4 * x + 5 * a ** 4 * b ** 4 * x ** 2 + 5 * a ** 4 * b ** 4 * x0 ** 2 + 240 * a * b * sigma ** 6 * np.exp(
                        3 * x0) + 32 * a ** 4 * b ** 3 * x * np.exp(x0) - 32 * a ** 4 * b ** 3 * x0 * np.exp(
                        x0) + 240 * a ** 2 * b * sigma ** 4 * np.exp(3 * x0) - 64 * a ** 4 * b ** 2 * x * np.exp(
                        2 * x0) + 64 * a ** 4 * b ** 2 * x0 * np.exp(2 * x0) - 28 * a ** 4 * b ** 3 * x ** 2 * np.exp(
                        x0) - 28 * a ** 4 * b ** 3 * x0 ** 2 * np.exp(
                        x0) - 10 * a ** 4 * b ** 4 * x * x0 - 60 * a ** 2 * b ** 2 * sigma ** 4 * np.exp(
                        2 * x0) + 40 * a ** 4 * b ** 2 * x ** 2 * np.exp(2 * x0) + 40 * a ** 4 * b ** 2 * x0 ** 2 * np.exp(
                        2 * x0) - 120 * a ** 2 * b * sigma ** 4 * x * np.exp(
                        3 * x0) + 120 * a ** 2 * b * sigma ** 4 * x0 * np.exp(3 * x0) + 24 * a * b * sigma ** 6 * x ** 2 * np.exp(
                        3 * x0) + 24 * a * b * sigma ** 6 * x0 ** 2 * np.exp(
                        3 * x0) + 32 * a ** 3 * b ** 3 * sigma ** 2 * x * np.exp(
                        x0) - 32 * a ** 3 * b ** 3 * sigma ** 2 * x0 * np.exp(x0) - 80 * a ** 4 * b ** 2 * x * x0 * np.exp(
                        2 * x0) - 128 * a ** 3 * b ** 2 * sigma ** 2 * x * np.exp(
                        2 * x0) + 128 * a ** 3 * b ** 2 * sigma ** 2 * x0 * np.exp(
                        2 * x0) - 4 * a ** 2 * b ** 2 * sigma ** 4 * x * np.exp(
                        2 * x0) + 4 * a ** 2 * b ** 2 * sigma ** 4 * x0 * np.exp(
                        2 * x0) + 24 * a ** 2 * b * sigma ** 4 * x ** 2 * np.exp(
                        3 * x0) + 24 * a ** 2 * b * sigma ** 4 * x0 ** 2 * np.exp(
                        3 * x0) - 28 * a ** 3 * b ** 3 * sigma ** 2 * x ** 2 * np.exp(
                        x0) - 28 * a ** 3 * b ** 3 * sigma ** 2 * x0 ** 2 * np.exp(x0) - 120 * a * b * sigma ** 6 * x * np.exp(
                        3 * x0) + 120 * a * b * sigma ** 6 * x0 * np.exp(3 * x0) + 56 * a ** 4 * b ** 3 * x * x0 * np.exp(
                        x0) + 80 * a ** 3 * b ** 2 * sigma ** 2 * x ** 2 * np.exp(
                        2 * x0) + 80 * a ** 3 * b ** 2 * sigma ** 2 * x0 ** 2 * np.exp(
                        2 * x0) + 16 * a ** 2 * b ** 2 * sigma ** 4 * x ** 2 * np.exp(
                        2 * x0) + 16 * a ** 2 * b ** 2 * sigma ** 4 * x0 ** 2 * np.exp(
                        2 * x0) - 48 * a * b * sigma ** 6 * x * x0 * np.exp(
                        3 * x0) - 48 * a ** 2 * b * sigma ** 4 * x * x0 * np.exp(
                        3 * x0) + 56 * a ** 3 * b ** 3 * sigma ** 2 * x * x0 * np.exp(
                        x0) - 160 * a ** 3 * b ** 2 * sigma ** 2 * x * x0 * np.exp(
                        2 * x0) - 32 * a ** 2 * b ** 2 * sigma ** 4 * x * x0 * np.exp(2 * x0)) / (
                            16 * x ** 7 * np.exp(4 * x0) - 16 * x0 ** 7 * np.exp(4 * x0) - 336 * x ** 2 * x0 ** 5 * np.exp(
                        4 * x0) + 560 * x ** 3 * x0 ** 4 * np.exp(4 * x0) - 560 * x ** 4 * x0 ** 3 * np.exp(
                        4 * x0) + 336 * x ** 5 * x0 ** 2 * np.exp(4 * x0) + 112 * x * x0 ** 6 * np.exp(
                        4 * x0) - 112 * x ** 6 * x0 * np.exp(4 * x0)) - np.exp(x0 - x) * ((a * b * np.exp(-3 * x0) * (
                    sigma ** 2 + a) * (- a ** 2 * b ** 2 + 6 * np.exp(2 * x0) * sigma ** 4)) / (4 * (x - x0) ** 5) + (
                                                                                               a * b * sigma ** 4 * np.exp(
                                                                                           -x0) * (sigma ** 2 + a)) / (
                                                                                               8 * (x - x0) ** 4) + (
                                                                                               15 * a * b * sigma ** 4 * np.exp(
                                                                                           -x0) * (sigma ** 2 + a)) / (
                                                                                               x - x0) ** 7 + (
                                                                                               a * b * np.exp(-3 * x0) * (
                                                                                                   sigma ** 2 + a) * (
                                                                                                           4 * a ** 2 * b ** 2 - 16 * np.exp(
                                                                                                       x0) * a ** 2 * b - 16 * np.exp(
                                                                                                       x0) * a * b * sigma ** 2 + 15 * np.exp(
                                                                                                       2 * x0) * sigma ** 4)) / (
                                                                                               2 * (x - x0) ** 6)) - np.exp(
            2 * x0 - 2 * x) * ((a ** 2 * b ** 2 * np.exp(-2 * x0) * (2 * a ** 2 + 4 * a * sigma ** 2 + sigma ** 4)) / (
                    4 * (x - x0) ** 4) + (a ** 2 * b ** 2 * np.exp(-4 * x0) * (
                    - 2 * a ** 2 * b ** 2 + 8 * np.exp(x0) * a ** 2 * b + 16 * np.exp(2 * x0) * a ** 2 + 8 * np.exp(
                x0) * a * b * sigma ** 2 + 32 * np.exp(2 * x0) * a * sigma ** 2 + np.exp(2 * x0) * sigma ** 4)) / (
                                           4 * (x - x0) ** 6) + (a ** 2 * b ** 2 * np.exp(-3 * x0) * (
                    4 * sigma ** 4 * np.exp(x0) + a ** 2 * b + 10 * a ** 2 * np.exp(x0) + 20 * a * sigma ** 2 * np.exp(
                x0) + a * b * sigma ** 2)) / (4 * (x - x0) ** 5) - (15 * a ** 2 * b ** 2 * sigma ** 4 * np.exp(-2 * x0)) / (
                                           4 * (x - x0) ** 7)) + np.exp(3 * x0 - 3 * x) * (
                            (a ** 3 * b ** 3 * np.exp(-3 * x0) * (sigma ** 2 + a)) / (2 * (x - x0) ** 4) + (
                                7 * a ** 3 * b ** 3 * np.exp(-3 * x0) * (sigma ** 2 + a)) / (4 * (x - x0) ** 5) + (
                                        2 * a ** 3 * b ** 3 * np.exp(-3 * x0) * (sigma ** 2 + a)) / (x - x0) ** 6) - np.exp(
            4 * x0 - 4 * x) * (
                            (a ** 4 * b ** 4 * np.exp(-4 * x0)) / (8 * (x - x0) ** 4) + (5 * a ** 4 * b ** 4 * np.exp(-4 * x0)) / (
                                16 * (x - x0) ** 5) + (a ** 4 * b ** 4 * np.exp(-4 * x0)) / (4 * (x - x0) ** 6)) - (
                            a * b * np.exp(-4 * x0) * (
                                a ** 3 * b ** 3 - 4 * np.exp(x0) * a ** 3 * b ** 2 + 4 * np.exp(2 * x0) * a ** 3 * b - 4 * np.exp(
                            x0) * a ** 2 * b ** 2 * sigma ** 2 + 8 * np.exp(2 * x0) * a ** 2 * b * sigma ** 2 + 2 * np.exp(
                            2 * x0) * a * b * sigma ** 4 + np.exp(3 * x0) * a * sigma ** 4 + np.exp(3 * x0) * sigma ** 6)) / (
                            8 * (x - x0) ** 4)

            )
            w_res = w0 + w1*t + w2*t**2 + w3*t**3 + w4*t**4

        return w_res

    def transition_density_x(self, x, x0, t, n: int = 3):
        pdf_x = (
                1/np.sqrt(2*np.pi*self.sigma**2*t) *
                np.exp(-(x-x0)**2 / (2*self.sigma**2*t) - self.w_for_density(x, x0, t, n))
        )
        return pdf_x

    def transition_density_y(self, y, y0, t, n: int = 3):
        x = np.log(y)
        x0 = np.log(y0)
        pdf_x = (
                1 / np.sqrt(2 * np.pi * self.sigma ** 2 * t) *
                np.exp(-(x - x0) ** 2 / (2 * self.sigma ** 2 * t) - self.w_for_density(x, x0, t, n))
        )
        pdf_y = pdf_x / y
        return pdf_y

    def w_for_ADprice(self, x, x0, t, n: int = 4) -> sp.Symbol:
        """
        Exponent Expansion on the nth order
        to approximately get W(x, x0, t) of Arrow-Debreu Prices.

        Parameters
        ----------
        x : symbol
            fractile point in terms of transformation, xt = ln(yt).
        x0 : float
            initial point of x.
        t : float
            time to expiry.
        n : int
            the highest order for Exponent Expansion.
        Returns
        -------
        w_res : symbol
            w = symsum(wn*t**n, 0, inf)
        """
        a = self.a
        b = self.b
        sigma = self.sigma

        w0 = ((x - x0)*(sigma**2/2 + a) + a*b*(np.exp(-x) - np.exp(-x0)))/sigma**2
        if n > 0.5:
            w1 = (
                    a / 2 + np.exp(x) / (x - x0) - np.exp(x0) / (x - x0) + sigma ** 2 / 8 + a ** 2 / (2 * sigma ** 2) - (
                        a ** 2 * b ** 2 * np.exp(-2 * x)) / (4 * (sigma ** 2 * x - sigma ** 2 * x0)) + (
                                a ** 2 * b ** 2 * np.exp(-2 * x0)) / (4 * (sigma ** 2 * x - sigma ** 2 * x0)) + (
                                a * b * np.exp(-x)) / (x - x0) - (a * b * np.exp(-x0)) / (x - x0) + (a ** 2 * b * np.exp(-x)) / (
                                sigma ** 2 * x - sigma ** 2 * x0) - (a ** 2 * b * np.exp(-x0)) / (
                                sigma ** 2 * x - sigma ** 2 * x0)

            )
            w_res = w0 + w1 * t
        if n > 1:
            w2 = (
                    (np.exp(-2 * x0) * (
                            a ** 2 * b ** 2 - 2 * np.exp(x0) * a ** 2 * b - 2 * np.exp(x0) * a * b * sigma ** 2 + 2 * np.exp(
                                3 * x0) * sigma ** 2)) / (2 * (x - x0) ** 2) + (sigma ** 2 * (np.exp(x) - np.exp(x0))) / (
                        2 * (x - x0) ** 2) + (a ** 2 * b ** 2 * sigma ** 2 * (
                        np.exp(-2 * x) / (2 * sigma ** 2 * (x - x0) ** 2) - np.exp(-2 * x0) / (
                            2 * sigma ** 2 * (x - x0) ** 2))) / 2 - (a ** 2 * b * sigma ** 2 * (
                        np.exp(-x) / (sigma ** 2 * (x - x0) ** 2) - np.exp(-x0) / (sigma ** 2 * (x - x0) ** 2))) / 2 - (
                                np.exp(- 2 * x - 2 * x0) * (np.exp(x) - np.exp(x0)) * (
                                    4 * sigma ** 2 * np.exp(2 * x) * np.exp(2 * x0) + a ** 2 * b ** 2 * np.exp(x) +
                                    a ** 2 * b ** 2 * np.exp(x0) - 4 * a ** 2 * b * np.exp(x) * np.exp(x0) -
                                    4 * a * b * sigma ** 2 * np.exp(x) * np.exp(x0))) / (4 * (x - x0) ** 3) -
                    (a * b * sigma ** 2 * (np.exp(-x) - np.exp(-x0))) / (2 * (x - x0) ** 2)
            )
            w_res += w2 * t**2
        if n > 2:
            w3 = (
                    np.exp(-2 * x) * ((a ** 2 * b ** 2 * (sigma ** 2 + a) ** 2) / (4 * sigma ** 2 * (x - x0) ** 3) + (
                        a ** 2 * b ** 2 * np.exp(-2 * x0) * (- a ** 2 * b ** 2 + 4 * np.exp(x0) * a ** 2 * b +
                                                               8 * np.exp(2 * x0) * a ** 2 +
                                                               4 * np.exp(x0) * a * b * sigma ** 2 +
                                                               16 * np.exp(2 * x0) * a * sigma ** 2 +
                                                               8 * np.exp(2 * x0) * sigma ** 4 +
                                                               4 * np.exp(3 * x0) * sigma ** 2)) /
                                        (16 * sigma ** 2 * (x - x0) ** 4)) - np.exp(2 * x) *
                    (sigma ** 2 / (4 * (x - x0) ** 3) - sigma ** 2 / (2 * (x - x0) ** 4)) + np.exp(-4 * x) * (
                                (a ** 4 * b ** 4) / (32 * sigma ** 2 * (x - x0) ** 3) + (a ** 4 * b ** 4) / (
                                    32 * sigma ** 2 * (x - x0) ** 4)) - np.exp(-3 * x) * (
                                (a ** 3 * b ** 3 * (sigma ** 2 + a)) / (6 * sigma ** 2 * (x - x0) ** 3) + (
                                    a ** 3 * b ** 3 * (sigma ** 2 + a)) / (4 * sigma ** 2 * (x - x0) ** 4)) -
                    (sigma ** 2 * ((12 * sigma ** 2 * np.exp(3 * x0) - 3 * a ** 2 * b ** 2 + 3 * a ** 2 * b ** 2 * x -
                                   3 * a ** 2 * b ** 2 * x0 + 12 * a ** 2 * b * np.exp(x0) +
                                   6 * sigma ** 2 * x * np.exp(3 * x0) - 6 * sigma ** 2 * x0 * np.exp(3 * x0) +
                                   12 * a * b * sigma ** 2 * np.exp(x0) - 6 * a ** 2 * b * x * np.exp(x0) +
                                   6 * a ** 2 * b * x0 * np.exp(x0) - 6 * a * b * sigma ** 2 * x * np.exp(x0) +
                                   6 * a * b * sigma ** 2 * x0 * np.exp(x0)) /
                                   (2 * x ** 5 * np.exp(2 * x0) - 2 * x0 ** 5 * np.exp(2 * x0) -
                                    20 * x ** 2 * x0 ** 3 * np.exp(2 * x0) +
                                    20 * x ** 3 * x0 ** 2 * np.exp(2 * x0) + 10 * x * x0 ** 4 * np.exp(2 * x0) -
                                    10 * x ** 4 * x0 * np.exp(2 * x0)) - np.exp(-x) *
                                   ((a * b * (sigma ** 2 + a)) / (2 * (x - x0) ** 3) +
                                    (3 * a * b * (sigma ** 2 + a)) / (x - x0) ** 4 +
                                    (6 * a * b * (sigma ** 2 + a)) / (x - x0) ** 5) -
                                   np.exp(x) * (sigma ** 2 / (2 * (x - x0) ** 3) - (3 * sigma ** 2) / (x - x0) ** 4 +
                                                  (6 * sigma ** 2) / (x - x0) ** 5) +
                                   np.exp(-2 * x) * ((a ** 2 * b ** 2) / (2 * (x - x0) ** 3) +
                                                       (3 * a ** 2 * b ** 2) / (2 * (x - x0) ** 4) +
                                                       (3 * a ** 2 * b ** 2) / (2 * (x - x0) ** 5)) +
                                   (np.exp(-2 * x0) *
                                    (- a ** 2 * b ** 2 + np.exp(x0) * a ** 2 * b + np.exp(x0) * a * b * sigma ** 2 +
                                     np.exp(3 * x0) * sigma ** 2)) / (2 * (x - x0) ** 3))) / 2 +
                    np.exp(-x) * ((a ** 2 * b ** 2) / (2 * (x - x0) ** 3) - (np.exp(-2 * x0) * (
                        - a ** 4 * b ** 3 + 4 * np.exp(x0) * a ** 4 * b ** 2 - a ** 3 * b ** 3 * sigma ** 2 +
                        8 * np.exp(x0) * a ** 3 * b ** 2 * sigma ** 2 +
                        4 * np.exp(x0) * a ** 2 * b ** 2 * sigma ** 4 +
                        np.exp(2 * x0) * a ** 2 * b ** 2 * sigma ** 2 +
                        4 * np.exp(3 * x0) * a ** 2 * b * sigma ** 2 +
                        4 * np.exp(3 * x0) * a * b * sigma ** 4)) / (4 * sigma ** 2 * (x - x0) ** 4)) +
                    (a ** 4 * b ** 4 - 8 * np.exp(x0) * a ** 4 * b ** 3 + 16 * np.exp(2 * x0) * a ** 4 * b ** 2 -
                     8 * np.exp(x0) * a ** 3 * b ** 3 * sigma ** 2 +
                     32 * np.exp(2 * x0) * a ** 3 * b ** 2 * sigma ** 2 +
                     16 * np.exp(2 * x0) * a ** 2 * b ** 2 * sigma ** 4 -
                     8 * np.exp(3 * x0) * a ** 2 * b ** 2 * sigma ** 2 +
                     64 * np.exp(4 * x0) * a ** 2 * b * sigma ** 2 +
                     64 * np.exp(4 * x0) * a * b * sigma ** 4 + 16 * np.exp(6 * x0) * sigma ** 4) /
                    (32 * sigma ** 2 * x ** 4 * np.exp(4 * x0) + 32 * sigma ** 2 * x0 ** 4 * np.exp(4 * x0) +
                     192 * sigma ** 2 * x ** 2 * x0 ** 2 * np.exp(4 * x0) -
                     128 * sigma ** 2 * x * x0 ** 3 * np.exp(4 * x0) -
                     128 * sigma ** 2 * x ** 3 * x0 * np.exp(4 * x0)) -
                    (np.exp(x - 2 * x0) * (- a ** 2 * b ** 2 + 4 * np.exp(x0) * a ** 2 * b +
                                             4 * np.exp(x0) * a * b * sigma ** 2 +
                                             4 * np.exp(3 * x0) * sigma ** 2)) / (4 * (x - x0) ** 4) -
                    (np.exp(-4 * x0) * (3 * a ** 4 * b ** 4 - 16 * np.exp(x0) * a ** 4 * b ** 3 +
                                          24 * np.exp(2 * x0) * a ** 4 * b ** 2 -
                                          16 * np.exp(x0) * a ** 3 * b ** 3 * sigma ** 2 +
                                          48 * np.exp(2 * x0) * a ** 3 * b ** 2 * sigma ** 2 +
                                          24 * np.exp(2 * x0) * a ** 2 * b ** 2 * sigma ** 4 +
                                          48 * np.exp(3 * x0) * a ** 2 * b ** 2 * sigma ** 2 -
                                          24 * np.exp(6 * x0) * sigma ** 4)) / (96 * sigma ** 2 * (x - x0) ** 3) +
                    (a * b * (sigma ** 2 + a)) / (x - x0) ** 2
            )
            w_res += w_res + w3 * t**3
        if n > 3:
            w4 = (
                    np.exp(-x) * ((a * b * np.exp(-2 * x0) * (
                        a ** 3 * b ** 2 + a ** 2 * b ** 2 * sigma ** 2 - 11 * np.exp(2 * x0) * a * b * sigma ** 2 - 6 * np.exp(
                    2 * x0) * a * sigma ** 4 + 8 * np.exp(3 * x0) * a * sigma ** 2 - 6 * np.exp(2 * x0) * sigma ** 6 + 8 * np.exp(
                    3 * x0) * sigma ** 4)) / (4 * (x - x0) ** 5) + (a * b * np.exp(-2 * x0) * (
                        - 4 * a ** 3 * b ** 2 + 16 * np.exp(x0) * a ** 3 * b - 4 * a ** 2 * b ** 2 * sigma ** 2 + 32 * np.exp(
                    x0) * a ** 2 * b * sigma ** 2 + 16 * np.exp(x0) * a * b * sigma ** 4 +
                        4 * np.exp(2 * x0) * a * b * sigma ** 2 - 15 * np.exp(2 * x0) * a * sigma ** 4 +
                        16 * np.exp(3 * x0) * a * sigma ** 2 - 15 * np.exp(2 * x0) * sigma ** 6 +
                        16 * np.exp(3 * x0) * sigma ** 4)) / (2 * (x - x0) ** 6) -
                               (15 * a * b * sigma ** 4 * (sigma ** 2 + a)) / (x - x0) ** 7 -
                               (a * b * sigma ** 2 * (sigma ** 4 + a * sigma ** 2 + 4 * a * b)) / (8 * (x - x0) ** 4)) -
                    np.exp(-2 * x) * ((a ** 2 * b ** 2 * (2 * a ** 2 + 4 * a * sigma ** 2 + sigma ** 4)) / (4 * (x - x0) ** 4) -
                                   (15 * a ** 2 * b ** 2 * sigma ** 4) / (4 * (x - x0) ** 7) +
                                   (a ** 2 * b ** 2 * np.exp(-2 * x0) *
                                    (- 2 * a ** 2 * b ** 2 + 8 * np.exp(x0) * a ** 2 * b + 16 * np.exp(2 * x0) * a ** 2 +
                                     8 * np.exp(x0) * a * b * sigma ** 2 + 32 * np.exp(2 * x0) * a * sigma ** 2 +
                                     np.exp(2 * x0) * sigma ** 4 + 8 * np.exp(3 * x0) * sigma ** 2)) / (4 * (x - x0) ** 6) +
                                   (a ** 2 * b ** 2 * np.exp(-x0) * (4 * sigma ** 4 * np.exp(x0) + 3 * sigma ** 2 * np.exp(
                                2 * x0) + a ** 2 * b + 10 * a ** 2 * np.exp(x0) + 20 * a * sigma ** 2 * np.exp(
                                x0) + a * b * sigma ** 2)) / (4 * (x - x0) ** 5)) + np.exp(x) * (
                                sigma ** 6 / (8 * (x - x0) ** 4) - (15 * sigma ** 6) / (x - x0) ** 7 - (
                                    sigma ** 2 * np.exp(-2 * x0) * (- 3 * a ** 2 * b ** 2 + 8 * np.exp(x0) * a ** 2 * b + 8 * np.exp(
                                x0) * a * b * sigma ** 2 + 6 * np.exp(2 * x0) * sigma ** 4)) / (4 * (x - x0) ** 5) + (
                                            sigma ** 2 * np.exp(-2 * x0) * (
                                                - 4 * a ** 2 * b ** 2 + 16 * np.exp(x0) * a ** 2 * b + 16 * np.exp(
                                            x0) * a * b * sigma ** 2 + 15 * np.exp(2 * x0) * sigma ** 4 + 16 * np.exp(
                                            3 * x0) * sigma ** 2)) / (2 * (x - x0) ** 6)) + np.exp(-3 * x) * (
                                (a ** 3 * b ** 3 * (sigma ** 2 + a)) / (2 * (x - x0) ** 4) + (
                                    7 * a ** 3 * b ** 3 * (sigma ** 2 + a)) / (4 * (x - x0) ** 5) + (
                                            2 * a ** 3 * b ** 3 * (sigma ** 2 + a)) / (x - x0) ** 6) - np.exp(2 * x) * (
                                sigma ** 4 / (2 * (x - x0) ** 4) - (5 * sigma ** 4) / (2 * (x - x0) ** 5) + (
                                    4 * sigma ** 4) / (x - x0) ** 6) - np.exp(-4 * x) * (
                                (a ** 4 * b ** 4) / (8 * (x - x0) ** 4) + (5 * a ** 4 * b ** 4) / (16 * (x - x0) ** 5) + (
                                    a ** 4 * b ** 4) / (4 * (x - x0) ** 6)) + (
                                240 * sigma ** 6 * np.exp(5 * x0) - 40 * sigma ** 4 * x ** 2 * np.exp(
                            6 * x0) - 40 * sigma ** 4 * x0 ** 2 * np.exp(6 * x0) + 24 * sigma ** 6 * x ** 2 * np.exp(
                            5 * x0) + 24 * sigma ** 6 * x0 ** 2 * np.exp(5 * x0) - 4 * a ** 4 * b ** 4 * x +
                                4 * a ** 4 * b ** 4 * x0 + 5 * a ** 4 * b ** 4 * x ** 2 + 5 * a ** 4 * b ** 4 * x0 ** 2 -
                                64 * sigma ** 4 * x * np.exp(6 * x0) + 64 * sigma ** 4 * x0 * np.exp(6 * x0) +
                                120 * sigma ** 6 * x * np.exp(5 * x0) - 120 * sigma ** 6 * x0 * np.exp(5 * x0) +
                                240 * a * b * sigma ** 6 * np.exp(
                            3 * x0) + 32 * a ** 4 * b ** 3 * x * np.exp(x0) - 32 * a ** 4 * b ** 3 * x0 * np.exp(
                            x0) + 80 * sigma ** 4 * x * x0 * np.exp(6 * x0) - 48 * sigma ** 6 * x * x0 * np.exp(
                            5 * x0) + 240 * a ** 2 * b * sigma ** 4 * np.exp(3 * x0) - 64 * a ** 4 * b ** 2 * x * np.exp(
                            2 * x0) + 64 * a ** 4 * b ** 2 * x0 * np.exp(2 * x0) - 28 * a ** 4 * b ** 3 * x ** 2 * np.exp(
                            x0) - 28 * a ** 4 * b ** 3 * x0 ** 2 * np.exp(
                            x0) - 10 * a ** 4 * b ** 4 * x * x0 - 60 * a ** 2 * b ** 2 * sigma ** 4 * np.exp(
                            2 * x0) + 40 * a ** 4 * b ** 2 * x ** 2 * np.exp(2 * x0) + 40 * a ** 4 * b ** 2 * x0 ** 2 * np.exp(
                            2 * x0) - 256 * a ** 2 * b * sigma ** 2 * x * np.exp(
                            4 * x0) + 256 * a ** 2 * b * sigma ** 2 * x0 * np.exp(
                            4 * x0) - 120 * a ** 2 * b * sigma ** 4 * x * np.exp(
                            3 * x0) + 120 * a ** 2 * b * sigma ** 4 * x0 * np.exp(
                            3 * x0) + 24 * a * b * sigma ** 6 * x ** 2 * np.exp(
                            3 * x0) + 24 * a * b * sigma ** 6 * x0 ** 2 * np.exp(
                            3 * x0) + 32 * a ** 3 * b ** 3 * sigma ** 2 * x * np.exp(
                            x0) - 32 * a ** 3 * b ** 3 * sigma ** 2 * x0 * np.exp(x0) - 80 * a ** 4 * b ** 2 * x * x0 * np.exp(
                            2 * x0) + 32 * a ** 2 * b ** 2 * sigma ** 2 * x * np.exp(
                            3 * x0) - 32 * a ** 2 * b ** 2 * sigma ** 2 * x0 * np.exp(
                            3 * x0) - 128 * a ** 3 * b ** 2 * sigma ** 2 * x * np.exp(
                            2 * x0) + 128 * a ** 3 * b ** 2 * sigma ** 2 * x0 * np.exp(
                            2 * x0) - 4 * a ** 2 * b ** 2 * sigma ** 4 * x * np.exp(
                            2 * x0) + 4 * a ** 2 * b ** 2 * sigma ** 4 * x0 * np.exp(
                            2 * x0) + 24 * a ** 2 * b * sigma ** 4 * x ** 2 * np.exp(
                            3 * x0) + 24 * a ** 2 * b * sigma ** 4 * x0 ** 2 * np.exp(
                            3 * x0) - 28 * a ** 3 * b ** 3 * sigma ** 2 * x ** 2 * np.exp(
                            x0) - 28 * a ** 3 * b ** 3 * sigma ** 2 * x0 ** 2 * np.exp(x0) - 256 * a * b * sigma ** 4 * x * np.exp(
                            4 * x0) + 256 * a * b * sigma ** 4 * x0 * np.exp(4 * x0) - 120 * a * b * sigma ** 6 * x * np.exp(
                            3 * x0) + 120 * a * b * sigma ** 6 * x0 * np.exp(3 * x0) + 56 * a ** 4 * b ** 3 * x * x0 * np.exp(
                            x0) + 44 * a ** 2 * b ** 2 * sigma ** 2 * x ** 2 * np.exp(
                            3 * x0) + 44 * a ** 2 * b ** 2 * sigma ** 2 * x0 ** 2 * np.exp(
                            3 * x0) + 80 * a ** 3 * b ** 2 * sigma ** 2 * x ** 2 * np.exp(
                            2 * x0) + 80 * a ** 3 * b ** 2 * sigma ** 2 * x0 ** 2 * np.exp(
                            2 * x0) + 16 * a ** 2 * b ** 2 * sigma ** 4 * x ** 2 * np.exp(
                            2 * x0) + 16 * a ** 2 * b ** 2 * sigma ** 4 * x0 ** 2 * np.exp(
                            2 * x0) - 48 * a * b * sigma ** 6 * x * x0 * np.exp(
                            3 * x0) - 48 * a ** 2 * b * sigma ** 4 * x * x0 * np.exp(
                            3 * x0) + 56 * a ** 3 * b ** 3 * sigma ** 2 * x * x0 * np.exp(
                            x0) - 88 * a ** 2 * b ** 2 * sigma ** 2 * x * x0 * np.exp(
                            3 * x0) - 160 * a ** 3 * b ** 2 * sigma ** 2 * x * x0 * np.exp(
                            2 * x0) - 32 * a ** 2 * b ** 2 * sigma ** 4 * x * x0 * np.exp(2 * x0)) / (
                                16 * x ** 7 * np.exp(4 * x0) - 16 * x0 ** 7 * np.exp(4 * x0) - 336 * x ** 2 * x0 ** 5 * np.exp(
                            4 * x0) + 560 * x ** 3 * x0 ** 4 * np.exp(4 * x0) - 560 * x ** 4 * x0 ** 3 * np.exp(
                            4 * x0) + 336 * x ** 5 * x0 ** 2 * np.exp(4 * x0) + 112 * x * x0 ** 6 * np.exp(
                            4 * x0) - 112 * x ** 6 * x0 * np.exp(4 * x0)) - (np.exp(-4 * x0) * (
                        a ** 4 * b ** 4 - 4 * np.exp(x0) * a ** 4 * b ** 3 + 4 * np.exp(2 * x0) * a ** 4 * b ** 2 - 4 * np.exp(
                    x0) * a ** 3 * b ** 3 * sigma ** 2 + 8 * np.exp(2 * x0) * a ** 3 * b ** 2 * sigma ** 2 + 2 * np.exp(
                    2 * x0) * a ** 2 * b ** 2 * sigma ** 4 + 4 * np.exp(3 * x0) * a ** 2 * b ** 2 * sigma ** 2 + np.exp(
                    3 * x0) * a ** 2 * b * sigma ** 4 + 32 * np.exp(4 * x0) * a ** 2 * b * sigma ** 2 + np.exp(
                    3 * x0) * a * b * sigma ** 6 + 32 * np.exp(4 * x0) * a * b * sigma ** 4 - np.exp(
                    5 * x0) * sigma ** 6 + 4 * np.exp(6 * x0) * sigma ** 4)) / (8 * (x - x0) ** 4)
            )
            w_res += w_res + w4 * t ** 4

        return w_res

    def bond_price(self, t, y0, n: int = 3) -> float:
        num = 50
        yvals = np.r_[np.linspace(0.001, y0 - 0.005, num), np.linspace(y0 + 0.005, 1.0, num)]
        x0 = np.log(y0)
        x = np.log(yvals)
        phi_y = 1/yvals * (
                1 / np.sqrt(2 * np.pi * self.sigma ** 2 * t) *
                np.exp(-(x - x0) ** 2 / (2 * self.sigma ** 2 * t) - self.w_for_ADprice(x, x0, t, n))
        )
        phi_y_inpld = interp1d(yvals, phi_y)
        price, _ = quad(phi_y_inpld, 0.001, 1)

        return price


def get_bond():
    order = 3
    t = np.array([0.1, 0.5, 1])
    prices = np.zeros(t.size)
    volmodel = ApproxMethod(a=0.1, b=0.04, sigma=0.6)
    for i in range(t.size):
        prices[i] = volmodel.bond_price(t=t[i], y0=0.06, n=order)

    print(prices)


def run():
    num = 50
    order = 3
    f = np.zeros(2*num)
    prob = np.zeros(2 * num)
    w = np.zeros(2 * num)
    z = np.r_[np.linspace(0.001, 0.055, num), np.linspace(0.065, 1.0, num)]
    volmodel = ApproxMethod(a=0.1, b=0.04, sigma=0.6)
    for i in range(z.size):
        print('\n', i)
        w[i] = volmodel.w_for_density(x=np.log(z[i]), x0=np.log(0.06), t=0.5, n=order)
        f[i] = volmodel.transition_density_y(y=z[i], y0=0.06, t=0.5, n=order)
    print('w:\n', w[:5], w[-5:])
    f_intpd = interp1d(z, f)
    for i in range(z.size):
        prob[i], _ = quad(lambda x: f_intpd(x), 0.001, z[i])

    fig, ax = plt.subplots(3, 1)
    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]

    ax1.plot(z, f)
    ax1.set_title('Order n = {:.0f}'.format(order))
    ax1.set_xlabel('z')
    ax1.set_ylabel('Prob Density')
    ax1.grid()

    ax2.plot(z, prob)
    ax2.set_ylabel('Accumulated Prob')
    ax2.grid()

    ax3.plot(np.log(z), w)
    ax3.set_xlabel('x')
    ax3.set_ylabel('W(x,t|x0, 0)')
    ax3.grid()

    plt.show()


if __name__ == '__main__':
    get_bond()
    # run()

