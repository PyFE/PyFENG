import abc
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve


class FmABC(abc.ABC):
    """

    Johnson's SU distribution approximation for basket/Asian options.

    References:

        Posner, S. E., & Milevsky, M. A. (1998). Valuing exotic options by approximating the SPD with higher moments.
        The Journal of Financial Engineering, 7(2). https://ssrn.com/abstract=108539

    """

    def __init__(self, spot, texp, sigma, r):
        self.spot = spot
        self.texp = texp
        self.sigma = sigma
        self.r = r

    @abc.abstractmethod
    def price(self, strike, spot, texp):
        pass

    # obtain first four moments
    def get_four_moments(self, s0, T, r, sigma):
        a = r + sigma ** 2
        b = 2 * r + sigma ** 2
        c = 2 * r + 3 * sigma ** 2

        moment1 = s0 / (r * T) * (np.exp(r * T) - 1)

        item1 = r * np.exp(b * T) - b * np.exp(r * T) + a
        moment2 = 2 * s0 ** 2 / (r * a * b * T ** 2) * item1

        item2 = 1 / c * 1 / (3 * a) * np.exp(3 * a * T)
        item3 = -1 / a * 1 / b * np.exp(b * T)
        item4 = (1 / (r * a) - (1 / (r * c))) * np.exp(r * T)
        item5 = 1 / (r * c) + 1 / (a * b)
        item6 = -1 / (3 * a * c) - 1 / (r * a)
        moment3 = 6 * s0 ** 3 / ((r + 2 * sigma ** 2) * T ** 3) * (item2 + item3 + item4 + item5 + item6)

        d = r + 2 * sigma ** 2
        e = 3 * r + 3 * sigma ** 2
        f = 2 * r + 5 * sigma ** 2
        g = 4 * r + 6 * sigma ** 2
        h = np.exp(g * T) - 1
        item7 = 1 / (f * e * g) * h - 1 / (f * e * r) * h
        item8 = - 1 / (f * a * b) * h + 1 / (f * a * r) * h
        item9 = - 1 / (d * c * e) * h + 1 / (d * c * r) * h
        item10 = 1 / (d * a * b) * h - 1 / (d * a * r) * h
        moment4 = 24 * s0 ** 4 / ((r + 3 * sigma ** 2) * T ** 4) * (item7 + item8 + item9 + item10)

        mu = moment1
        var = moment2 - moment1 ** 2
        skew = (moment3 - moment1 ** 3 - 3 * moment2 * moment1 + 3 * moment1 ** 3) / np.power(var, 3 / 2)
        kurt = (moment4 - 3 * moment1 ** 4 - 4 * moment3 * moment1 + 6 * moment2 * moment1 ** 2) / var ** 2

        return mu, var, skew, kurt

    def moment_matching(self, mu, var, skew, kurt):
        w0 = np.power(8 + 4 * skew ** 2 + 4 * np.sqrt(4 * skew ** 2 + skew ** 4), 1 / 3)
        w = 1 / 2 * w0 + 2 / w0 - 1

        k = w ** 4 + 2 * w ** 3 + 3 * w ** 2 - 3
        # Type I
        if np.isclose(kurt - k, 0):
            b = np.power(np.log(w), -1 / 2)
            a = 0.5 * b * np.log(w * (w - 1) / var)
            d = np.sign(skew)
            c = - np.exp((1 / 2 * b - a) / b)
            return a, b, c, d, 1
        # Type II
        elif np.isclose(skew, 0):
            w = np.sqrt(np.sqrt(2 * kurt - 2) - 1)
            b = 1 / np.sqrt(np.log(w))
            a = 0
        else:
            w = np.sqrt(np.sqrt(2 * kurt - 2.8 * skew ** 2 - 2) - 1)
            a, b = self.Johnson_iteration(w, skew**2, kurt)
        d = np.sqrt(2 * var / ((w - 1) * (w * np.cosh(2 * a / b) + 1)))
        c = mu + d * np.sqrt(w) * np.sinh(a / b)
        return a, b, c, d, 0

    def Johnson_iteration(self, w, b1, b2):

        def helper(w):
            A2 = 8 * (w ** 3 + 3 * w ** 2 + 6 * w + 6)
            A1 = 8 * (w ** 4 + 3 * w ** 3 + 6 * w ** 2 + 7 * w + 3)
            A0 = w ** 5 + 3 * w ** 4 + 6 * w ** 3 + 10 * w ** 2 + 9 * w + 3

            def func_m(m):
                return b2 - 3 - (w - 1) * (A2 * m ** 2 + A1 * m + A0) / (2 * (2 * m + w + 1) ** 2)
            m_root = fsolve(func_m, 5)[0]
            b1_iter = m_root * (w - 1) * (4 * (w + 2) * m_root + 3 * (w + 1) ** 2) ** 2 / (2 * (2 * m_root + w + 1) ** 3)
            return b1_iter, m_root

        w_root = w
        for _ in range(10):
            b1_iter, m_root = helper(w_root)
            print(b1_iter, b1)
            def func_w(w2):
                return b1 / b1_iter - (b2 - 1 / 2 * (w2 ** 4 + 2 * w2 ** 2 + 3)) / (
                            b2 - 1 / 2 * (w ** 4 + 2 * w ** 2 + 3))
            w_root = fsolve(func_w, 5)[0]

        delta = np.power(np.log(w_root), -1 / 2)

        def func_gamma(gamma):
            return m_root - w_root * (np.sinh(gamma / delta)) ** 2

        gamma_root = fsolve(func_gamma, 5)[0]
        return gamma_root, delta


class FmCall(FmABC):
    """

    Johnson's SU distribution approximation for basket/Asian call option.

    """

    def __init__(self, strike, spot, texp, sigma, r):
        super(FmCall, self).__init__(spot=spot, texp=texp, sigma=sigma, r=r)
        self.strike = strike

    # Type I (log) Johnson distribution.
    @staticmethod
    def _asian_call_price_tpye1(a, b, c, d, mu, K, r, T):
        item0 = (K - c) * norm.cdf(a + b * np.log((K - c) / d)) - d * np.exp((1 - 2 * a * b) / (2 * b ** 2)) * norm.cdf(
            a + b * np.log((K - c) / d) - 1 / b)
        return np.exp(-r * T) * (mu - K + item0)

    # Type II (inverse hyperbolic sine) Johnson distribution.
    @staticmethod
    def _asian_call_price_tpye2(a, b, c, d, mu, K, r, T):
        Q = a + b * np.arcsinh((K - c) / d)
        item0 = (K - c) * norm.cdf(Q) + d / 2 * np.exp(1 / (2 * b ** 2)) * (
                    np.exp(a / b) * norm.cdf(Q + 1 / b) - np.exp(-a / b) * norm.cdf(Q - 1 / b))
        return np.exp(-r * T) * (mu - K + item0)

    def price(self, strike, spot, texp):
        mu, var, skew, kurt = self.get_four_moments(spot, texp, self.r, self.sigma)
        a, b, c, d, is_type1 = self.moment_matching(mu, var, skew, kurt)
        if is_type1:
            return self._asian_call_price_tpye1(a, b, c, d, mu, strike, self.r, texp)
        else:
            return self._asian_call_price_tpye2(a, b, c, d, mu, strike, self.r, texp)


if __name__ == "__main__":
    r = 0.09
    q = 0
    sigma = 0.50
    s0 = 100
    K = 90
    T = 1

    fmCall = FmCall(K, s0, T, sigma, r)
    # price = fmCall.price(K, s0, T)
    print(fmCall.get_four_moments(s0, T, r, sigma))