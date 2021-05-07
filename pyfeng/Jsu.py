import math
import numpy as np
from itertools import product, combinations
from . import opt_abc as opt
from . import NormBasket
from . import nsvh


class BsmBasketJsu(NormBasket):
    """

    Johnson's SU distribution approximation for Basket option pricing under the multiasset BSM model.

    Note: Johnson's SU distribution is the solution of NSVh with NSVh with lambda = 1.

    References:
        [1] Posner, S. E., & Milevsky, M. A. (1998). Valuing exotic options by approximating the SPD
        with higher moments. The Journal of Financial Engineering, 7(2). https://ssrn.com/abstract=108539

        [2] Choi, J., Liu, C., & Seo, B. K. (2019). Hyperbolic normal stochastic volatility model.
        Journal of Futures Markets, 39(2), 186–204. https://doi.org/10.1002/fut.21967

    """

    weight = None

    def __init__(self, sigma, cor=None, weight=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatilities of `n_asset` assets. (n_asset, ) array
            cor: correlation. If matrix, used as it is. (n_asset, n_asset)
                If scalar, correlation matrix is constructed with all same off-diagonal values.
            weight: asset weights, If None, equally weighted as 1/n_asset
                If scalar, equal weights of the value
                If 1-D array, uses as it is. (n_asset, )
            intr: interest rate (domestic interest rate)
            divr: vector of dividend/convenience yield (foreign interest rate) 0-D or (n_asset, ) array
            is_fwd: if True, treat `spot` as forward price. False by default.
        """

        super().__init__(sigma, cor=cor, weight=weight, intr=intr, divr=divr, is_fwd=is_fwd)

    def moment_vsk(self, fwd, texp):
        """

        Return variance, skewness, kurtosis for Basket options.

        Args:
            fwd: forward price
            texp: time to expiry

        Returns: variance, skewness, kurtosis of Basket options

        """
        n = len(self.weight)

        m1 = sum(self.weight[i] * fwd[i] for i in range(n))

        m2_index = [i for i in product(np.arange(n), repeat=2)]
        m2 = sum(self.weight[i] * self.weight[j] * fwd[i] * fwd[j] *
                 np.exp(self.sigma[i] * self.sigma[j] * self.cor_m[i][j] * texp) for i, j in m2_index)

        m3_index = [i for i in product(np.arange(n), repeat=3)]
        m3 = sum(self.weight[i] * self.weight[j] * self.weight[l] * fwd[i] * fwd[j] * fwd[l] *
                 np.exp(sum(self.sigma[ii] * self.sigma[jj] * self.cor_m[ii][jj] for ii, jj in
                            combinations(np.array([i, j, l]), 2)) * texp) for i, j, l in m3_index)

        m4_index = [i for i in product(np.arange(n), repeat=4)]
        m4 = sum(self.weight[i] * self.weight[j] * self.weight[l] * self.weight[k] * fwd[i] * fwd[j] * fwd[l] *
                 fwd[k] * np.exp(sum(self.sigma[ii] * self.sigma[jj] * self.cor_m[ii][jj] for ii, jj in
                                      combinations(np.array([i, j, l, k]), 2)) * texp) for i, j, l, k in m4_index)

        var = m2 - m1 ** 2
        skew = (m3 - m1 ** 3 - 3 * m2 * m1 + 3 * m1 ** 3) / var ** (3 / 2)
        kurt = (m4 - 3 * m1 ** 4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2) / var ** 2

        return var, skew, kurt

    def price(self, strike, spot, texp, cp=1):
        """

        Basket options price.
        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
        Returns: Basket options price

        """
        df = np.exp(-texp * self.intr)
        fwd = np.array(spot) * (1.0 if self.is_fwd else np.exp(-texp * np.array(self.divr)) / df)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd @ self.weight

        var, skew, kurt = self.moment_vsk(fwd, texp)

        m = nsvh.Nsvh1(sigma=self.sigma)
        m.calibrate_vsk(var, skew, kurt-3, texp, setval=True)
        price = m.price(strike, fwd_basket, texp, cp)

        return df * price


class BsmAsianJsu(opt.OptMaABC):
    """

    Johnson's SU distribution approximation for Asian option pricing under the BSM model.

    Note: Johnson's SU distribution is the solution of NSVh with NSVh with lambda = 1.

    References:
        [1] Posner, S. E., & Milevsky, M. A. (1998). Valuing exotic options by approximating the SPD
        with higher moments. The Journal of Financial Engineering, 7(2). https://ssrn.com/abstract=108539

        [2] Choi, J., Liu, C., & Seo, B. K. (2019). Hyperbolic normal stochastic volatility model.
        Journal of Futures Markets, 39(2), 186–204. https://doi.org/10.1002/fut.21967

    """

    def __init__(self, sigma, cor=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatilities of `n_asset` assets. (n_asset, ) array
            cor: correlation. If matrix, used as it is. (n_asset, n_asset)
                If scalar, correlation matrix is constructed with all same off-diagonal values.
            intr: interest rate (domestic interest rate)
            divr: vector of dividend/convenience yield (foreign interest rate) 0-D or (n_asset, ) array
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        super().__init__(sigma, cor=cor, intr=intr, divr=divr, is_fwd=is_fwd)

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
        v = (self.intr - self.divr - lam**2 / 2) / lam
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
        skew = (m3 - m1 ** 3 - 3 * m2 * m1 + 3 * m1 ** 3) / var**(3 / 2)
        kurt = (m4 - 3 * m1 ** 4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2) / var**2

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
        m.calibrate_vsk(var, skew, kurt-3, texp, setval=True)
        price = m.price(strike, mu, texp, cp)

        return df * price


