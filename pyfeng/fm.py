import math
import numpy as np
from scipy.stats import norm
import scipy.optimize as spop


class Fm:
    """

    Johnson's SU distribution approximation for basket/Asian options.

    Args:
        sigma: volatility
        r: risk-free rate
        q: dividend

    References:
        Posner, S. E., & Milevsky, M. A. (1998). Valuing exotic options by approximating the SPD with higher moments.
        The Journal of Financial Engineering, 7(2). https://ssrn.com/abstract=108539

    """
    def __init__(self, sigma, r, q):
        self.sigma = sigma
        self.r = r
        self.q = q

    def moments_fm(self, spot, texp, n=1):
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
        lam = self.sigma
        v = (self.r - self.q - self.sigma**2 / 2) / self.sigma
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

    def get_moments(self, spot, texp):
        """

        Return mean, variance, skewness, kurtosis.

        Args:
            spot: spot price
            texp: time to expiry

        Returns: mean, variance, skewness, kurtosis

        """
        moments = []
        for i in range(1, 5):
            moments.append(self.moments_fm(spot, texp, i))

        m1, m2, m3, m4 = moments[0], moments[1], moments[2], moments[3]
        mu = m1
        var = m2 - m1 ** 2
        skew = (m3 - m1 ** 3 - 3 * m2 * m1 + 3 * m1 ** 3) / var**(3 / 2)
        kurt = (m4 - 3 * m1 ** 4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2) / var**2

        return mu, var, skew, kurt

    def calibrate_fm(self, mu, var, skew, kurt):
        """

        Calibrate parameters to the moments: mean, variance, skewness, kurtosis.

        References:
            Tuenter, H. J. H. (2001). An algorithm to determine the parameters of SU-curves in the johnson system of
            probability distributions by moment matching. Journal of Statistical Computation and Simulation, 70(4),
            325–347. https://doi.org/10.1080/00949650108812126

        Args:
            mu: mean
            var: variance
            skew: skewness
            kurt: kurtosis. should be > 0

        Returns: parameters of Johnson distribution

        """
        assert kurt > 0
        beta1 = skew**2
        beta2 = kurt

        # min of w search
        roots = np.roots(np.array([1, 2, 3, 0, -3-beta2]))
        roots = roots[(roots.real > 0) & np.isclose(roots.imag, 0)]
        assert len(roots) == 1
        w_min = roots.real[0]
        w_max = np.sqrt(-1 + np.sqrt(2*(beta2 - 1)))

        def f_beta1(w):
            term1 = np.sqrt(4 + 2*(w*w - (beta2 + 3) / (w*w + 2*w + 3)))
            return (w + 1 - term1)*(w + 1 + 0.5*term1)**2 - beta1

        assert f_beta1(w_min) >= 0

        # root finding
        w_root = spop.brentq(f_beta1, w_min, w_max)
        m = -2 + np.sqrt(4 + 2*(w_root**2 - (beta2 + 3)/(w_root**2 + 2*w_root + 3)))
        term = (w_root + 1)/(2*w_root)*((w_root - 1)/m - 1)

        # if term is slightly negative, next line error in sqrt
        if abs(term) < np.finfo(float).eps*100:
            term = 0.0

        omega = -np.sign(skew) * np.log(term + np.sqrt(term**2 + 1))
        gamma = omega / np.sqrt(np.log(w_root))
        delta = 1 / np.sqrt(np.log(w_root))

        mu1 = - np.sqrt(w_root) * np.sinh(omega)
        var1 = 1/2 * (w_root - 1) * (w_root * np.cosh(2 * omega) + 1)
        d = np.sqrt(var / var1)  # sign?
        c = mu - mu1 * d

        return gamma, delta, c, d, 0

    def price(self, strike, spot, texp, cp=1):
        """

        Call option price (cp == 1), or put option price (cp == 0).

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns: call/put option price

        """
        mu, var, skew, kurt = self.get_moments(spot, texp)
        a, b, c, d, is_type1 = self.calibrate_fm(mu, var, skew, kurt)

        Q = a + b * np.arcsinh((strike - c) / d)
        item0 = (strike - c) * norm.cdf(Q) + d / 2 * np.exp(1 / (2 * b ** 2)) * (
                    np.exp(a / b) * norm.cdf(Q + 1 / b) - np.exp(-a / b) * norm.cdf(Q - 1 / b))
        if cp:
            return np.exp(-self.r * texp) * (mu - strike + item0)
        else:
            return np.exp(-self.r * texp) * item0
