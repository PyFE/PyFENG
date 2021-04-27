import abc
import nsvh
import numpy as np
from scipy.stats import norm


class FmABC(abc.ABC):
    """

    Johnson's SU distribution approximation for basket/Asian options.

    References:

        Posner, S. E., & Milevsky, M. A. (1998). Valuing exotic options by approximating the SPD with higher moments.
        The Journal of Financial Engineering, 7(2). https://ssrn.com/abstract=108539

    """

    @abc.abstractmethod
    def integral_CDF(self, a, b, c, d, K):
        pass


class FmTypeI(FmABC):
    """

    Type I (log) Johnson distribution.

    """

    def integral_CDF(self, a, b, c, d, K):
        return -d * np.exp((1 - 2 * a * b) / (2 * b**2)) * norm.cdf(a + b * np.log((K - c) / d) - 1 / b)



class FmTypeII(FmABC):
    """

    Type II (inverse hyperbolic sine) Johnson distribution.

    """

    def integral_CDF(self, a, b, c, d, K):
        Q = a + b * np.arcsinh((K - c) / d)
        return (K - c) * norm.cdf(Q) + d / 2 * np.exp(1 / (2 * b**2)) * \
               (np.exp(a / b) * norm.cdf(Q + 1 / b) - np.exp(-a / b) * norm.cdf(Q - 1 / b))
