import numpy as np
import scipy.special as spsp
from statsmodels.stats.moment_helpers import cum2mc

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
        Numerically stable computation of ``(exp(x) - 1) / x``.

        This equals the average of ``exp`` over the interval ``[0, x]``:

        .. math::

            \\frac{1}{x} \\int_0^x e^t \\, dt = \\frac{e^x - 1}{x}

        Uses ``np.expm1(x)`` internally to avoid catastrophic cancellation
        when ``x`` is close to zero (where ``exp(x) - 1`` loses nearly all
        significant digits). The limit as ``x → 0`` is ``1``, which is
        returned exactly for ``x = 0``.

        Works for both scalar and ndarray inputs. Always returns a float
        array (or float scalar) regardless of the input dtype.

        Args:
            x: argument (scalar or ndarray, any numeric dtype)

        Returns:
            ``(exp(x) - 1) / x``, with the limiting value ``1.0`` at ``x = 0``.
            Shape and type match those of ``x``, but dtype is always ``float``.

        Examples:
            >>> MathFuncs.avg_exp(0.0)          # limit at zero → 1
            1.0
            >>> MathFuncs.avg_exp(1.0)          # (e - 1) / 1 ≈ 1.7183
            1.7182818284590453
            >>> MathFuncs.avg_exp(np.array([0.0, 1.0, -1.0]))
            array([1.        , 1.71828183, 0.63212056])
        """

        rv = np.ones_like(x, dtype=float)
        np.divide(np.expm1(x),  x, out=rv, where=(x != 0.0))
        return rv

    @staticmethod
    def avg_inv(x):
        """
        Numerically stable computation of ``log(1 + x) / x``.

        This equals the average of ``1/t`` over the interval ``[1, 1+x]``:

        .. math::

            \\frac{1}{x} \\int_1^{1+x} \\frac{1}{t} \\, dt = \\frac{\\ln(1+x)}{x}

        Uses ``np.log1p(x)`` internally to avoid catastrophic cancellation
        when ``x`` is close to zero (where ``log(1+x)`` loses significant digits
        if computed naively). The limit as ``x → 0`` is ``1``, which is
        returned exactly for ``x = 0``.

        Works for both scalar and ndarray inputs. Always returns a float
        array (or float scalar) regardless of the input dtype.

        Args:
            x: argument (scalar or ndarray, any numeric dtype). Must satisfy
               ``x > -1`` (so that ``1 + x > 0``); raises ``ValueError`` otherwise.

        Returns:
            ``log(1 + x) / x``, with the limiting value ``1.0`` at ``x = 0``.
            Shape and type match those of ``x``, but dtype is always ``float``.

        Examples:
            >>> MathFuncs.avg_inv(0.0)          # limit at zero → 1
            1.0
            >>> MathFuncs.avg_inv(1.0)          # log(2) / 1 ≈ 0.6931
            0.6931471805599453
            >>> MathFuncs.avg_inv(np.array([0.0, 1.0, -0.5]))
            array([1.        , 0.69314718, 1.38629436])
        """
        if not np.all(x > -1.0):
            raise ValueError("x must be greater than -1.0.")

        rv = np.ones_like(x, dtype=float)
        np.divide(np.log1p(x),  x, out=rv, where=(x != 0.0))
        return rv

    @staticmethod
    def avg_pow(x, a):
        """
        Numerically stable computation of the average of ``t^a`` over ``[1, 1+x]``.

        This equals:

        .. math::

            \\frac{1}{x} \\int_1^{1+x} t^a \\, dt =
            \\begin{cases}
            \\dfrac{(1+x)^{a+1} - 1}{(a+1)\\,x} & a \\neq -1 \\\\[6pt]
            \\dfrac{\\ln(1+x)}{x}               & a = -1
            \\end{cases}

        The case ``a = -1`` reduces to :meth:`avg_inv`. In all cases the limit
        as ``x → 0`` is ``1``, which is returned exactly for ``x = 0``.

        Numerical stability is achieved by rewriting ``(1+x)^{a+1} - 1`` as
        ``expm1((a+1) * log1p(x))``, avoiding cancellation for small ``x``
        or for ``a+1`` close to zero.

        Works for both scalar and ndarray inputs. Always returns a float
        array (or float scalar) regardless of the input dtype.

        Args:
            x: argument (scalar or ndarray, any numeric dtype). Must satisfy
               ``x > -1`` (so that ``1 + x > 0``); raises ``ValueError`` otherwise.
            a: power exponent (scalar or ndarray broadcastable with ``x``).

        Returns:
            Average of ``t^a`` over ``[1, 1+x]``, with the limiting value ``1.0``
            at ``x = 0``. Shape is broadcast shape of ``x`` and ``a``, dtype
            is always ``float``.

        Examples:
            >>> MathFuncs.avg_pow(0.0, 2.0)     # limit at zero → 1
            1.0
            >>> MathFuncs.avg_pow(1.0, 2.0)     # (2^3 - 1) / (3 * 1) = 7/3 ≈ 2.3333
            2.3333333333333335
            >>> MathFuncs.avg_pow(1.0, -1.0)    # reduces to avg_inv: log(2) ≈ 0.6931
            0.6931471805599453
        """

        if not np.all(x > -1.0):
            raise ValueError("x must be greater than -1.0.")
        a1p = 1.0 + a
        rv = np.ones_like(x + a, dtype=float)   # rv = 1 when x = 0
        np.divide(np.expm1(a1p * np.log1p(x)), a1p * x, out=rv, where=(x != 0.0) & (a1p != 0.0))
        np.divide(np.log1p(x),  x, out=rv, where=(x != 0.0) & (a1p == 0.0))
        return rv


class StatFuncs:

    def gramcharA_coefs(
            cumuls: np.ndarray,
            sigma: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Gram-Charlier A series coefficients in terms of standardised variable z.

        The expansion is:
            f(z) = phi(z) * sum_n  c[n] * He_{n+1}(z)

        where z = (x - mu) / sigma and phi(z) is the standard normal density.

        Convention: cumuls[k] = cumul_{k+1}, k = 0, ..., m-1
            cumuls[0] = cumul_1  (mean,     absorbed into mu)
            cumuls[1] = cumul_2  (variance, used for sigma)
            cumuls[2] = cumul_3  (skewness driver)
            ...

        Output: c[k] = tilde_c_{k+1}, k = 0, ..., m-1
            c[0] = 0                           (c_1, centering)
            c[1] = 0                           (c_2, centering)
            c[2] = cumul_3 / (6 sigma^3)       (c_3)
            c[3] = cumul_4 / (24 sigma^4)      (c_4)
            ...
            c[k] = cumul_{k+1} / ((k+1)! sigma^{k+1})

        Parameters
        ----------
        cumuls : np.ndarray, shape (m,)
            Cumulants. cumuls[k] = cumul_{k+1}.
        sigma  : float or None
            Defaults to sqrt(cumuls[1]) = sqrt(cumul_2).

        Returns
        -------
        c     : np.ndarray, shape (m,)
            Gram-Charlier coefficients in z-space.
        sigma : float
            Sigma actually used.
        """

        m = len(cumuls)

        # --- Resolve sigma ---
        if sigma is None:
            if cumuls[1] <= 0:
                raise ValueError(f"cumul_2 = cumuls[1] = {cumuls[1]} must be positive.")
            sigma = np.sqrt(cumuls[1])

        # --- sigma powers: sigma_pow[k] = sigma^{-(k+1)} ---
        cumul_std = np.cumprod(np.full(m, 1/sigma))
        # --- Standardised excess cumulants for cum2mc ---
        cumul_std[0:2] = 0.0
        cumul_std[2:] *= cumuls[2:]

        # --- B[k] = 1/(k+1)! ---
        c = 1.0/np.cumprod(np.arange(1, m + 1, dtype=float))
        # --- Bell polynomials via cum2mc: returns [B_1, ..., B_m] ---
        # --- c[k] = B[k] / (k+1)!  (no sigma division in z-space) ---
        c *= np.array(cum2mc(cumul_std))

        return c, sigma
