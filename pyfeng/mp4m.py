import math
import numpy as np
from itertools import product, combinations
from . import nsvh


class MP4M(nsvh.Nsvh1):
    """

    Johnson's SU distribution approximation for basket/Asian options.

    Note: Johnson's SU distribution is the solution of NSVh with NSVh with lambda = 1.

    References:
        [1] Posner, S. E., & Milevsky, M. A. (1998). Valuing exotic options by approximating the SPD
        with higher moments. The Journal of Financial Engineering, 7(2). https://ssrn.com/abstract=108539

        [2] Choi, J., Liu, C., & Seo, B. K. (2019). Hyperbolic normal stochastic volatility model.
        Journal of Futures Markets, 39(2), 186–204. https://doi.org/10.1002/fut.21967

    """

    def __init__(self, sigma):
        """
        Args:
            sigma: model volatility at t=0
        """
        super().__init__(sigma)

    def asian_moments(self, spot, texp, sigma, r, q, n=1):
        """

        The moments of the Asian option with a log-normal underlying asset.

        Args:
            spot: spot price
            texp: time to expiry
            n: moment order
            sigma: volatility
            r: risk-free rate
            q: dividend

        References:
            Geman, H., & Yor, M. (1993). Bessel processes, Asian options, and perpetuities.
            Mathematical finance, 3(4), 349-375.

        Returns: the nth moment

        """
        lam = sigma
        v = (r - q - sigma**2 / 2) / sigma
        beta = v / lam

        ds = []
        for j in range(n+1):
            item0 = 2 ** n
            for i in range(n+1):
                if i != j:
                    item0 *= ((beta + j)**2 - (beta + i)**2)**(-1)
            ds.append(item0)
        item1 = 0
        for i in range(n+1):
            item1 += (ds[i] * np.exp((lam**2 * i**2 / 2 + lam * i * v) * texp))
        moment = (spot / texp)**n * math.factorial(n) / lam ** (2 * n) * item1

        return moment

    def asian_mvsk(self, spot, texp, sigma, r, q):
        """

        Return mean, variance, skewness, kurtosis for Asian options.

        Args:
            spot: spot price
            texp: time to expiry
            sigma: volatility
            r: risk-free rate
            q: dividend

        Returns: mean, variance, skewness, kurtosis of Asian options

        """
        moments = []
        for i in range(1, 5):
            moments.append(self.asian_moments(spot, texp, sigma, r, q, i))

        m1, m2, m3, m4 = moments[0], moments[1], moments[2], moments[3]
        mu = m1
        var = m2 - m1 ** 2
        skew = (m3 - m1 ** 3 - 3 * m2 * m1 + 3 * m1 ** 3) / var**(3 / 2)
        kurt = (m4 - 3 * m1 ** 4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2) / var**2

        return mu, var, skew, kurt

    def basket_mvsk(self, spot, w, texp, sigma, rho):
        """

        Return mean, variance, skewness, kurtosis for Basket options.

        Args:
            spot: spot price
            w: weights for stocks
            texp: time to expiry
            sigma: volatility
            rho: correlation

        Returns: mean, variance, skewness, kurtosis of Basket options

        """
        n = len(w)

        m1 = sum(w[i] * spot[i] for i in range(n))

        m2_index = [i for i in product(np.arange(n), repeat=2)]
        m2 = sum(w[i] * w[j] * spot[i] * spot[j] * np.exp(sigma[i] * sigma[j] * rho[i][j] * texp) for
                 i, j in m2_index)

        m3_index = [i for i in product(np.arange(n), repeat=3)]
        m3 = sum(w[i] * w[j] * w[l] * spot[i] * spot[j] * spot[l] *
                 np.exp(sum(sigma[ii] * sigma[jj] * rho[ii][jj] for ii, jj in
                            combinations(np.array([i, j, l]), 2)) * texp) for i, j, l in m3_index)

        m4_index = [i for i in product(np.arange(n), repeat=4)]
        m4 = sum(w[i] * w[j] * w[l] * w[k] * spot[i] * spot[j] * spot[l] * spot[k] *
                 np.exp(sum(sigma[ii] * sigma[jj] * rho[ii][jj] for ii, jj in
                            combinations(np.array([i, j, l, k]), 2)) * texp) for i, j, l, k in m4_index)
        mu = m1
        var = m2 - m1 ** 2
        skew = (m3 - m1 ** 3 - 3 * m2 * m1 + 3 * m1 ** 3) / var ** (3 / 2)
        kurt = (m4 - 3 * m1 ** 4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2) / var ** 2

        return mu, var, skew, kurt

    def asian_price(self, strike, spot, texp, sigma, r, q, cp=1):
        """

        Asian options price.

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            sigma: volatility
            r: risk-free rate
            q: dividend
            cp: 1/-1 for call/put option

        Returns: Asian options price

        """
        mu, var, skew, kurt = self.asian_mvsk(spot, texp, sigma, r, q)
        self.calibrate_vsk(var, skew, kurt-3, texp, setval=True)
        return self.price(strike, mu, texp, cp)

    def basket_price(self, strike, spot, w, texp, sigma, rho, cp=1):
        """

        Basket options price.

        Args:
            strike: strike price
            spot: spot price
            w: weights for stocks
            texp: time to expiry
            sigma: volatility
            rho: correlation
            cp: 1/-1 for call/put option

        Returns: Basket options price

        """
        mu, var, skew, kurt = self.basket_mvsk(spot, w, texp, sigma, rho)
        self.calibrate_vsk(var, skew, kurt-3, texp, setval=True)
        return self.price(strike, mu, texp, cp)

