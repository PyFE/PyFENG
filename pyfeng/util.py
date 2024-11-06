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
        rv = np.ones_like(x + a)   # rv = 1 when x = 0
        np.divide(np.expm1(a1p * np.log1p(x)), a1p * x, out=rv, where=(x != 0.0) & (a1p != 0.0))
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
    validate = False
    _ww = 0.0  # := exp(sig^2) - 1 (normalized variance)

    def __init__(self, sig=1.0, lam=1.0, mu=1.0, validate=False):
        self.sig = sig; self.lam = lam; self.mu = mu
        self.validate = validate
        self._ww = MathFuncs.avg_exp(sig ** 2) * sig**2

    def mvsk(self):
        """
        mean, scaled var (var/mean^2), skewness and ex-kurtosis

        Returns:
            (m, v, s, k)
        """
        var_scaled = self.lam**2 * self._ww
        skew = np.sqrt(self._ww) * (self._ww + 3)
        exkur = self._ww * (16 + self._ww * (15 + self._ww * (6 + self._ww)))

        return self.mu, var_scaled, skew, exkur

    def mc4(self):
        """
        First four central moments

        Returns:
            (m1, mc2, mc3, mc4)
        """
        m1, v, s, k = self.mvsk()
        mc2 = v * m1**2
        mc3 = s * mc2**1.5
        mc4 = (k + 3.) * mc2**2

        return m1, mc2, mc3, mc4

    def fit(self, mvs, lam=None):
        """
        Fits the parameter given mvs (mean, scaled variance, skewness)
        Args:
            mvs: (m, v, s) If (m, v) is given, lam or self.lam is used. If (m, v, s), lam is calibrated
            lam: lambda. Used only when (m, v) is given. Ignored when (m, v, s) is fully speficied.

        Returns:
            None
        """

        assert len(mvs) >= 2

        self.mu = mvs[0]

        if len(mvs) == 2 or lam is not None:  # use (m, v) only
            if lam is None:
                # if lam is None, self.lam should be specified
                assert self.lam is not None
            else:
                # if lam is specified, store it to self.lam
                self.lam = lam

            self._ww = mvs[1] / self.lam**2
            self.sig = np.sqrt(np.log1p(self._ww))
        else:
            assert len(mvs) > 2
            s = mvs[2]
            sqrt_w = 2*np.sinh(np.arccosh(1 + 0.5*s**2)/6)
            self.lam = np.sqrt(mvs[1]) / sqrt_w
            self._ww = sqrt_w ** 2
            self.sig = np.sqrt(np.log1p(self._ww))

        if self.validate:
            n = 2 if len(mvs) == 2 or lam is not None else 3
            mvsk2 = self.mvsk()
            for (i,v) in enumerate(mvs[:n]):
                assert np.isclose(v, mvsk2[i])

    def quad(self, n_quad):
        """
        Quadrature points and weights

        Args:
            n_quad: number of points

        Returns:
            (points, weights)
        """

        z, w, w_sum = spsp.roots_hermitenorm(n_quad, mu=True)
        w /= w_sum  # 1/np.sqrt(2.0 * np.pi)
        z = self.mu * (1 + self.lam * np.expm1(self.sig*(z - self.sig/2)))
        return z, w


class ChebInterp:
    """
    Chebyshev interpolator at the Chebyshev nodes of thd 2nd kind, x_k = cos(k/(n-1) pi) for k = 0, ..., n-1

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> interp = pf.ChebInterp(n=8, inc=True)
        >>> y = np.exp(interp.x)
        >>> interp.fit(y)
        >>> y2 = interp.eval(interp.x)
        >>> np.max(np.abs(y - y2))
    """

    inc = False
    coef = []
    x = []

    def __init__(self, n_pts, inc=False):
        """

        Args:
            n_pts: number of points
            inc: True if x_k is increasing, i.e., x_k = -cos(k/(n-1) pi). False by default.

        """
        self.inc = inc
        self.x = (-1 if inc else 1) * np.cos(np.pi * np.arange(n_pts) / (n_pts - 1))

    def fit(self, y):
        """
        Fit Chebyshev interpolator

        Args:
            y: y values at x_k = cos(k/(n-1) pi) for k = 0, ..., n-1
        """
        n = len(y)
        assert n == len(self.x)

        if self.inc:
            y = np.flip(y)

        coef = np.fft.rfft(np.concatenate((y, y[-2:0:-1]))).real / (n-1)
        coef[[0, -1]] /= 2
        self.coef = coef

    def eval(self, x):
        """
        Interplate at x using the fitted coefficients.
        `numpy.polynomial.chebyshev.chebval` uses Clenshaw recursion algorithm.

        Args:
            x: values to evaluate

        References:
            - https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebval.html

        Returns:
            Interpolated y values at x
        """

        ###
        y = np.polynomial.chebyshev.chebval(x, self.coef)
        return y