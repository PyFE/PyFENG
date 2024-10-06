import numpy as np
import scipy.special as spsp

class MathConsts:
    """

    References:
        https://github.com/lattera/glibc/blob/master/math/math.h
    """

    M_E         = 2.7182818284590452354   # e
    M_LOG2E     = 1.4426950408889634074   # log_2 e
    M_LOG10E    = 0.43429448190325182765  # log_10 e
    M_LN2       = 0.69314718055994530942  # log_e 2
    M_LN10      = 2.30258509299404568402  # log_e 10
    M_PI        = 3.14159265358979323846  # pi
    M_PI_2      = 1.57079632679489661923  # pi/2
    M_PI_4      = 0.78539816339744830962  # pi/4
    M_1_PI      = 0.31830988618379067154  # 1/pi
    M_2_PI      = 0.63661977236758134308  # 2/pi
    M_2_SQRTPI  = 1.12837916709551257390  # 2/sqrt(pi)
    M_SQRT2     = 1.41421356237309504880  # sqrt(2)
    M_SQRT1_2   = 0.70710678118654752440  # 1/sqrt(2)

    # values below are from https://keisan.casio.com/calculator
    M_SQRT2PI   = 2.506628274631000502416  # sqrt(2pi)
    M_1_SQRT2PI = 0.3989422804014326779399  # 1/sqrt(2pi)
    M_SQRT_PI_2 = 1.253314137315500251208  # sqrt(pi/2)
    M_LN2PI_2   = 0.9189385332046727417803  # log(sqrt(2pi)) = log(2pi)/2


class MathFuncs:

    @staticmethod
    def mills_ratio(x):
        """
        Mills Ratio: R(x) = N(-x)/n(x) = sqrt(pi/2) erfcx(x/sqrt(2))

        Args:
            x: argument

        Returns:

        """
        return MathConsts.M_SQRT_PI_2 * spsp.erfcx(x * MathConsts.M_SQRT1_2)

    @staticmethod
    def avg_exp(x):
        """
        Integral_0^x exp(x) dx / x = ( exp(x) - 1 ) / x

        Args:
            x: argument

        Returns:
            value
        """

        rv = np.ones_like(x)
        np.divide(np.expm1(x),  x, out=rv, where=(x != 0.0))
        return rv

    @staticmethod
    def avg_inv(x):
        """
        [Integarl 1/x from 1 to 1+x] / x = log(1+x) / x

        Args:
            x: argument

        Returns:

        """
        assert np.all(x > -1.0)

        rv = np.ones_like(x)
        np.divide(np.log1p(x),  x, out=rv, where=(x != 0.0))
        return rv

    @staticmethod
    def avg_pow(x, a):
        """
        (int from 1 to (1+x) t^a dt) / x
            = [(1+x)^(1+a) - 1] / [(1+a)*x]   if a != -1
            = log(1+x) / x                    if a = -1
        Args:
            x: argument
            a: exponent

        Returns:

        """

        assert np.all(x > -1.0)
        a1p = 1.0 + a
        rv = np.expm1(a1p * np.log1p(x))   # rv = 1 when x = 0
        np.divide(rv,  a1p * x, out=rv, where=(x != 0.0) & (a1p != 0.0))
        np.divide(np.log1p(x),  x, out=rv, where=(x != 0.0) & (a1p == 0.0))
        return rv


class DistHelperLnShift:
    """
    Shifted lognormal distribution:
        Y ~ mu [(1-lam) + lam * exp(sig*Z - sig^2/2)] and Z ~ N(0,1)

    If lam=1, the distribution is reduced the lognormal distribution
    """

    mu = 1.0
    sig = 0.0
    lam = 1.0
    _ww = 0.0  # := exp(sig^2) - 1 (normalized variance)

    def __init__(self, sig=1.0, lam=1.0, mu=1.0):
        self.sig = sig; self.lam = lam; self.mu = mu
        self._ww = MathFuncs.avg_exp(sig ** 2) * sig**2

    def mvsk(self):
        """
        mean, variance, skewness and ex-kurtosis

        Returns:
            (m, v, s, k)
        """
        var = (self.mu * self.lam)**2 * self._ww
        skew = np.sqrt(self._ww) * (self._ww + 3)
        exkur = self._ww * (16 + self._ww * (15 + self._ww * (6 + self._ww)))

        return self.mu, var, skew, exkur

    def fit(self, mvs, lam=None, validate=False):
        """
        Fits the parameter given mvs (mean, variance, skewness)
        Args:
            mvs: (m, v, s) If (m, v) is given, lam or self.lam is used. If (m, v, s), lam is calibrated
            lam: lambda. Used only when (m, v) is given. Ignored when (m, v, s) is fully speficied.

        Returns:
            None
        """

        assert len(mvs) >= 2

        self.mu = mvs[0]
        coef_var_sq = mvs[1] / mvs[0]**2  # squared coefficient of variance

        if len(mvs) == 2 or lam is not None:  # use (m, v) only
            if lam is None:
                # if lam is None, self.lam should be specified
                assert self.lam is not None
            else:
                # if lam is specified, store it to self.lam
                self.lam = lam

            self._ww = coef_var_sq / self.lam**2
            self.sig = np.sqrt(np.log1p(self._ww))
        else:
            assert len(mvs) > 2
            s = mvs[2]
            sqrt_w = 2*np.sinh(np.arccosh(1 + 0.5*s**2)/6)
            self.lam = np.sqrt(coef_var_sq) / sqrt_w
            self._ww = sqrt_w ** 2
            self.sig = np.sqrt(np.log1p(self._ww))

        if validate:
            n = 2 if len(mvs) == 2 or lam is not None else 3
            mvsk2 = self.mvsk()
            for (i,v) in enumerate(mvs[:n]):
                assert np.isclose(v, mvsk2[i])