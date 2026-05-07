# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:08:29 2021

@author: Yuze, Lantian
"""
import warnings
from . import multiasset as ma
from .opt_abc import OptABC
import numpy as np
from .util import MathFuncs
import scipy.stats as spst


class BsmBasketAsianJu2002(ma.NormBasket):

    def average_s(self, spot, texp, basket=True):
        # cal the forward price of asset num in the basket
        n = self.n_asset
        if np.isscalar(spot):
            spot = np.full(n, spot)
        if np.isscalar(self.divr):
            self.divr = np.full(n, self.divr)
        av_s = np.zeros(n)
        if basket:
            for num in range(n):
                av_s[num] = (
                    self.weight[num]
                    * spot[num]
                    * np.exp((self.intr - self.divr[num]) * texp)
                )
        else:
            for num in range(n):
                av_s[num] = (
                    self.weight[num]
                    * spot[num]
                    * np.exp(
                        (self.intr - self.divr[num]) * texp / (n - 1) * num
                    )
                )
        self.av_s = av_s

    def average_rho(self, texp, basket=True):
        # cal the rho between asset i and j
        n = self.n_asset
        av_rho = np.zeros((n, n))
        if basket:
            for i in range(n):
                for j in range(n):
                    av_rho[i, j] = self.cor_m[i, j] * self.sigma[i] * self.sigma[j] * texp
        else:
            for i in range(n):
                for j in range(i, n):
                    av_rho[i, j] = self.sigma[0] ** 2 * texp / (n - 1) * i
                    av_rho[j, i] = av_rho[i, j]
        self.av_rho = av_rho

    def u1(self, spot, texp):
        # the first momentum of log normal distribution#
        u1_value = self.av_s.sum()
        # u1_value = self.weight @ (spot * np.exp((self.intr-self.divr)*texp))
        return u1_value

    def u2(self, z):
        # the second momentum of log normal distribution#
        u2_value = 0
        for i in range(self.n_asset):
            for j in range(self.n_asset):
                u2_value += (
                    self.av_s[i] * self.av_s[j] * np.exp(z * z * self.av_rho[i, j])
                )
        return u2_value

    def u2_1st_der(self):
        u2_1st_value = 0
        for i in range(self.n_asset):
            for j in range(self.n_asset):
                u2_1st_value += self.av_s[i] * self.av_s[j] * self.av_rho[i, j]
        return u2_1st_value

    def u2_2nd_der(self):
        u2_2nd_value = 0
        for i in range(self.n_asset):
            for j in range(self.n_asset):
                u2_2nd_value += self.av_s[i] * self.av_s[j] * pow(self.av_rho[i, j], 2)
        return u2_2nd_value

    def u2_3rd_der(self):
        u2_3rd_value = 0
        for i in range(self.n_asset):
            for j in range(self.n_asset):
                u2_3rd_value += self.av_s[i] * self.av_s[j] * pow(self.av_rho[i, j], 3)
        return u2_3rd_value

    def ak_bar(self):
        # calculate the average a, save to self#
        av_a = self.av_rho @ self.av_s
        self.av_a = av_a

    def e_a12_a2(self):
        return 2 * self.av_s @ pow(self.av_a, 2)

    def e_a12_a22(self):
        value = 0
        for i in range(self.n_asset):
            for j in range(self.n_asset):
                value += (
                    self.av_a[i]
                    * self.av_s[i]
                    * self.av_rho[i, j]
                    * self.av_a[j]
                    * self.av_s[j]
                )
        value *= 8
        value += 2 * self.u2_1st_der() * self.u2_2nd_der()
        return value

    def e_a13_a3(self):
        return 6 * self.av_s @ pow(self.av_a, 3)

    def e_a1_a2_a3(self):
        value = 0
        for i in range(self.n_asset):
            for j in range(self.n_asset):
                value += (
                    self.av_s[i]
                    * pow(self.av_rho[i, j], 2)
                    * self.av_a[j]
                    * self.av_s[j]
                )
        value *= 6
        return value

    def e_a23(self):
        value = 0
        n = self.n_asset
        temp = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                temp[i, j] = (
                    pow(self.av_s[i], 0.5) * self.av_rho[i, j] * pow(self.av_s[j], 0.5)
                )
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    value += temp[i, j] * temp[j, k] * temp[k, i]
        value *= 8
        return value

    def func_a1(self, z):
        return -pow(z, 2) * self.u2_1st_der() / 2 / self.u2(0)

    def func_a2(self, z):
        return 2 * pow(self.func_a1(z), 2) - pow(
            z, 4
        ) * self.u2_2nd_der() / 2 / self.u2(0)

    def func_a3(self, z):
        return (
            6 * self.func_a1(z) * self.func_a2(z)
            - 4 * pow(self.func_a1(z), 3)
            - pow(z, 6) * self.u2_3rd_der() / 2 / self.u2(0)
        )

    def func_b1(self, spot, texp, z):
        return pow(z, 4) * self.e_a12_a2() / 4 / pow(self.u1(spot, texp), 3)

    def func_b2(self, z):
        return pow(self.func_a1(z), 2) - self.func_a2(z) / 2

    def func_c1(self, spot, texp, z):
        return -self.func_a1(z) * self.func_b1(spot, texp, z)

    def func_c2(self, spot, texp, z):
        return (
            pow(z, 6)
            * (9 * self.e_a12_a22() + 4 * self.e_a13_a3())
            / 144
            / pow(self.u1(spot, texp), 4)
        )

    def func_c3(self, spot, texp, z):
        return (
            pow(z, 6)
            * (4 * self.e_a1_a2_a3() + self.e_a23())
            / 48
            / pow(self.u1(spot, texp), 3)
        )

    def func_c4(self, z):
        return (
            self.func_a1(z) * self.func_a2(z)
            - 2 * pow(self.func_a1(z), 3) / 3
            - self.func_a3(z) / 6
        )

    def func_d1(self, spot, texp, z):
        return 0.5 * (
            6 * pow(self.func_a1(z), 2)
            + self.func_a2(z)
            - 4 * self.func_b1(spot, texp, z)
            + 2 * self.func_b2(z)
        ) - 1 / 6 * (
            120 * pow(self.func_a1(z), 3)
            - self.func_a3(z)
            + 6
            * (
                24 * self.func_c1(spot, texp, z)
                - 6 * self.func_c2(spot, texp, z)
                + 2 * self.func_c3(spot, texp, z)
                - self.func_c4(z)
            )
        )

    def func_d2(self, spot, texp, z):
        return 0.5 * (
            10 * pow(self.func_a1(z), 2)
            + self.func_a2(z)
            - 6 * self.func_b1(spot, texp, z)
            + 2 * self.func_b2(z)
        ) - (
            128 * pow(self.func_a1(z), 3) / 3
            - self.func_a3(z) / 6
            + 2 * self.func_a1(z) * self.func_b1(spot, texp, z)
            - self.func_a1(z) * self.func_b2(z)
            + 50 * self.func_c1(spot, texp, z)
            - 11 * self.func_c2(spot, texp, z)
            + 3 * self.func_c3(spot, texp, z)
            - self.func_c4(z)
        )

    def func_d3(self, spot, texp, z):
        return (
            2 * pow(self.func_a1(z), 2)
            - self.func_b1(spot, texp, z)
            - 1
            / 3
            * (
                88 * pow(self.func_a1(z), 3)
                + 3
                * self.func_a1(z)
                * (5 * self.func_b1(spot, texp, z) - 2 * self.func_b2(z))
                + 3
                * (
                    35 * self.func_c1(spot, texp, z)
                    - 6 * self.func_c2(spot, texp, z)
                    + self.func_c3(spot, texp, z)
                )
            )
        )

    def func_d4(self, spot, texp, z):
        return (
            -20 * pow(self.func_a1(z), 3) / 3
            + self.func_a1(z) * (-4 * self.func_b1(spot, texp, z) + self.func_b2(z))
            - 10 * self.func_c1(spot, texp, z)
            + self.func_c2(spot, texp, z)
        )

    def price(self, strike, spot, texp, cp=1, basket=True):
        if np.isscalar(spot):
            spot = np.full(self.n_asset, spot)
        if np.isscalar(self.divr):
            self.divr = np.full(self.n_asset, self.divr)
        if basket:
            self.average_s(spot, texp)
            self.average_rho(texp)
        else:
            self.average_s(spot, texp, False)
            self.average_rho(texp, False)
        self.ak_bar()
        m1 = 2 * np.log(self.u1(spot, texp)) - 0.5 * np.log(self.u2(1))
        v1 = np.log(self.u2(1)) - 2 * np.log(self.u1(spot, texp))
        sqrtv1 = np.sqrt(v1)
        y = np.log(strike)
        y1 = (m1 - y) / np.sqrt(v1) + sqrtv1
        y2 = y1 - sqrtv1
        z1 = (
            self.func_d2(spot, texp, 1)
            - self.func_d3(spot, texp, 1)
            + self.func_d4(spot, texp, 1)
        )
        z2 = self.func_d3(spot, texp, 1) - self.func_d4(spot, texp, 1)
        z3 = self.func_d4(spot, texp, 1)
        bc = (
            self.u1(spot, texp)
            * np.exp(-self.intr * texp)
            * spst.norm.cdf(y1, loc=0, scale=1)
            - strike * np.exp(-self.intr * texp) * spst.norm.cdf(y2, loc=0, scale=1)
            + np.exp(-self.intr * texp)
            * strike
            * (
                z1 * spst.norm.pdf(y, loc=m1, scale=sqrtv1)
                + z2 * spst.norm.pdf(y, loc=m1, scale=sqrtv1) * (m1 - y) / v1
                + z3
                * ((y - m1) * (y - m1) / v1 / v1 - 1 / v1)
                * spst.norm.pdf(y, loc=m1, scale=sqrtv1)
            )
        )
        if cp == 1:
            return bc
        elif cp == -1:
            return np.exp(-self.intr * texp) * (strike - self.u1(spot, texp)) + bc
        else:
            return -1


class BsmContinuousAsianJu2002(OptABC):
    def price(self, strike, spot, texp, cp=1):

        if np.isscalar(spot) == False:
            raise ValueError("spot must be a scalar.")
        elif np.isscalar(self.divr) == False:
            raise ValueError("dividend (divr) must be a scalar.")
            return 0
        else:
            g = self.intr - self.divr
            gt = g * texp
            u1 = spot * MathFuncs.avg_exp(gt)
            g2 = 2 * g + self.sigma ** 2
            u2 = (
                2 * spot ** 2
                * (MathFuncs.avg_exp(g2 * texp) - MathFuncs.avg_exp(gt))
                / texp
                / (g + self.sigma ** 2)
            )
            z1 = -pow(self.sigma, 4) * texp ** 2 * (
                1 / 45
                + gt / 180
                - 11 * gt ** 2 / 15120
                - pow(gt, 3) / 2520
                + pow(gt, 4) / 113400
            ) - pow(self.sigma, 6) * pow(texp, 3) * (
                1 / 11340
                - 13 * gt / 30240
                - 17 * gt ** 2 / 226800
                + 23 * pow(gt, 3) / 453600
                + 59 * pow(gt, 4) / 5987520
            )
            z2 = -pow(self.sigma, 4) * texp ** 2 * (
                1 / 90
                + gt / 360
                - 11 * gt ** 2 / 30240
                - pow(gt, 3) / 5040
                + pow(gt, 4) / 226800
            ) - pow(self.sigma, 6) * pow(texp, 3) * (
                31 / 22680
                - 11 * gt / 60480
                - 37 * gt ** 2 / 151200
                - 19 * pow(gt, 3) / 302400
                + 953 * pow(gt, 4) / 59875200
            )
            z3 = (
                pow(self.sigma, 6)
                * pow(texp, 3)
                * (
                    2 / 2835
                    - gt / 60480
                    - 2 * gt ** 2 / 14175
                    - 17 * pow(gt, 3) / 907200
                    + 13 * pow(gt, 4) / 1247400
                )
            )
            m1 = 2 * np.log(u1) - 0.5 * np.log(u2)
            v1 = np.log(u2) - 2 * np.log(u1)
            sqrtv1 = np.sqrt(v1)
            y = np.log(strike)
            y1 = (m1 - y) / np.sqrt(v1) + sqrtv1
            y2 = y1 - sqrtv1
            bc = (
                u1 * np.exp(-self.intr * texp) * spst.norm.cdf(y1, loc=0, scale=1)
                - strike * np.exp(-self.intr * texp) * spst.norm.cdf(y2, loc=0, scale=1)
                + np.exp(-self.intr * texp)
                * strike
                * (
                    z1 * spst.norm.pdf(y, loc=m1, scale=sqrtv1)
                    + z2 * spst.norm.pdf(y, loc=m1, scale=sqrtv1) * (m1 - y) / v1
                    + z3
                    * ((y - m1) * (y - m1) / v1 / v1 - 1 / v1)
                    * spst.norm.pdf(y, loc=m1, scale=sqrtv1)
                )
            )
        if cp == 1:
            return bc
        elif cp == -1:
            return np.exp(-self.intr * texp) * (strike - u1) + bc
        else:
            return -1
