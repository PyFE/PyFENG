import numpy as np
import scipy.special as spsp
import scipy.optimize as spop
from .util import MathFuncs


class DistLognormal:
    """
    Shifted lognormal distribution:
        Y ~ mu [(1-lam) + lam * exp(sig*Z - sig^2/2)],  Z ~ N(0,1)

    If lam=1 the distribution reduces to the plain lognormal.
    """

    mu = 1.0
    sig = 0.0
    lam = 1.0
    validate = False
    _ww = 0.0  # exp(sig^2) - 1  (normalized variance)

    def __init__(self, sig=1.0, lam=1.0, mu=1.0, validate=False):
        self.sig = sig
        self.lam = lam
        self.mu = mu
        self.validate = validate
        self._ww = MathFuncs.avg_exp(sig**2) * sig**2

    @classmethod
    def from_mv(cls, mean, coef_var, lam=1.0):
        """
        Construct from mean and coefficient of variation (coef_var = variance/mean²).

        Args:
            mean: distribution mean
            coef_var: coefficient of variation squared (variance / mean²)
            lam: lambda parameter (default 1.0, plain lognormal)

        Returns:
            DistLognormal instance calibrated to (mean, coef_var)
        """
        obj = cls()
        obj.fit([mean, coef_var], lam=lam)
        return obj

    def mvsk(self):
        """
        Mean, coefficient of variation squared (coef_var = var/mean²), skewness, excess kurtosis.

        Returns:
            (mean, coef_var, skewness, excess_kurtosis)
        """
        coef_var = self.lam**2 * self._ww
        skew = np.sqrt(self._ww) * (self._ww + 3)
        exkur = self._ww * (16 + self._ww * (15 + self._ww * (6 + self._ww)))
        return self.mu, coef_var, skew, exkur

    def mc4(self):
        """
        First four central moments: (mean, variance, mc3, mc4).

        Returns:
            (m1, mc2, mc3, mc4)
        """
        m1, cv, s, k = self.mvsk()
        mc2 = cv * m1**2
        mc3 = s * mc2**1.5
        mc4 = (k + 3.0) * mc2**2
        return m1, mc2, mc3, mc4

    def fit(self, mvs, lam=None):
        """
        Calibrate parameters from moments.

        Args:
            mvs: (mean, coef_var) or (mean, coef_var, skewness).
                 coef_var = variance / mean².
                 If skewness is provided, lam is calibrated; otherwise self.lam is used.
            lam: override for lam when len(mvs) == 2.
        """
        if len(mvs) < 2:
            raise ValueError("mvs must have at least 2 elements.")

        self.mu = mvs[0]

        if len(mvs) == 2 or lam is not None:
            if lam is None:
                if self.lam is None:
                    raise ValueError("lam must be specified when mvs has only 2 elements.")
            else:
                self.lam = lam
            self._ww = mvs[1] / self.lam**2
            self.sig = np.sqrt(np.log1p(self._ww))
        else:
            s = mvs[2]
            sqrt_w = 2 * np.sinh(np.arccosh(1 + 0.5 * s**2) / 6)
            self.lam = np.sqrt(mvs[1]) / sqrt_w
            self._ww = sqrt_w**2
            self.sig = np.sqrt(np.log1p(self._ww))

        if self.validate:
            n = 2 if len(mvs) == 2 or lam is not None else 3
            mvsk2 = self.mvsk()
            for i, v in enumerate(mvs[:n]):
                if not np.isclose(v, mvsk2[i]):
                    raise ValueError(
                        f"Moment validation failed at index {i}: expected {v}, got {mvsk2[i]}."
                    )

    def quad(self, n_quad):
        """
        Gauss–Hermite quadrature nodes and weights.

        Args:
            n_quad: number of quadrature points.

        Returns:
            (nodes, weights)
        """
        z, w, w_sum = spsp.roots_hermitenorm(n_quad, mu=True)
        w /= w_sum
        z = self.mu * (1 + self.lam * np.expm1(self.sig * (z - self.sig / 2)))
        return z, w


class DistGamma:
    """
    Gamma distribution.

    The PDF is:
        f(x) = rate^shape / Gamma(shape) * x^(shape-1) * exp(-rate * x),  x > 0

    Parameters:
        shape (α > 0): shape parameter
        rate  (β > 0): rate parameter; scale θ = 1/β

    References:
        https://en.wikipedia.org/wiki/Gamma_distribution
    """

    shape = 1.0
    rate = 1.0

    def __init__(self, shape=1.0, rate=1.0):
        self.shape = shape
        self.rate = rate

    @classmethod
    def from_mv(cls, mean, coef_var):
        """
        Construct from mean and coefficient of variation (coef_var = variance/mean²).

        Args:
            mean: distribution mean (> 0)
            coef_var: coefficient of variation squared (variance / mean²); equals 1/shape

        Returns:
            DistGamma instance with shape = 1/coef_var, rate = 1/(coef_var * mean)
        """
        obj = cls()
        obj.fit(mean, coef_var)
        return obj

    @property
    def scale(self):
        return 1.0 / self.rate

    def mvsk(self):
        """
        Mean, coefficient of variation squared (coef_var = var/mean²), skewness, excess kurtosis.

        Returns:
            (mean, coef_var, skewness, excess_kurtosis)
        """
        mean = self.shape / self.rate
        coef_var = 1.0 / self.shape
        skew = 2.0 / np.sqrt(self.shape)
        exkurt = 6.0 / self.shape
        return mean, coef_var, skew, exkurt

    def mc4(self):
        """
        First four central moments: (mean, variance, mc3, mc4).

        Returns:
            (m1, mc2, mc3, mc4)
        """
        m1, cv, skew, exkurt = self.mvsk()
        var = cv * m1**2
        mc3 = skew * var**1.5
        mc4 = (exkurt + 3.0) * var**2
        return m1, var, mc3, mc4

    def fit(self, mean, coef_var):
        """
        Calibrate shape and rate from mean and coefficient of variation.

        Args:
            mean: distribution mean (> 0)
            coef_var: coefficient of variation squared (variance / mean²); > 0
        """
        self.shape = 1.0 / coef_var
        self.rate = 1.0 / (coef_var * mean)

    def quad(self, n_quad):
        """
        Gauss–Laguerre quadrature nodes and weights.

        The quadrature rule exactly integrates polynomials of degree < 2*n_quad
        against the Gamma(shape, rate) density.

        Args:
            n_quad: number of quadrature points.

        Returns:
            (nodes, weights)

        Examples:
            >>> import scipy.special as spsp
            >>> alpha, beta = 2, 2
            >>> x, w = DistGamma(shape=alpha, rate=beta).quad(9)
            >>> sum(x*w), alpha/beta  # mean of Gamma
            >>> sum(w/x), beta/(alpha-1)  # mean of 1/X
        """
        assert self.shape > 0
        x, w, w_sum = spsp.roots_genlaguerre(n_quad, self.shape - 1, mu=True)
        x /= self.rate
        w /= w_sum
        return x, w


class DistInvGauss:
    """
    Inverse Gaussian (IG) distribution.

    The PDF is:
        f(x) = sqrt(lam / (2*pi*x^3)) * exp(-lam*(x-mu)^2 / (2*mu^2*x)),  x > 0

    Parameters:
        mu  (> 0): mean parameter
        lam (> 0): shape (precision) parameter

    References:
        https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
        Choi J, Du Y, Song Q (2021) Inverse Gaussian quadrature and finite
        normal-mixture approximation of the generalized hyperbolic distribution.
        Journal of Computational and Applied Mathematics 388:113302.
        https://doi.org/10.1016/j.cam.2020.113302
    """

    mu = 1.0
    lam = 1.0

    def __init__(self, mu=1.0, lam=1.0):
        self.mu = mu
        self.lam = lam

    @classmethod
    def from_mv(cls, mean, coef_var):
        """
        Construct from mean and coefficient of variation (coef_var = variance/mean²).

        Args:
            mean: distribution mean (> 0)
            coef_var: coefficient of variation squared (variance / mean²); equals mu/lam

        Returns:
            DistInvGauss instance with mu=mean, lam=mean/coef_var
        """
        obj = cls()
        obj.fit(mean, coef_var)
        return obj

    @classmethod
    def from_gig(cls, gamma, delta):
        """
        Construct from the GIG-style (γ, δ) parametrization used in Choi et al. (2021).

        IG(γ, δ) is the special case GIG(γ, δ, p=−½), with density

            f(x | γ, δ) = δ / √(2π x³)  exp(−(γx − δ)² / (2x)),   γ ≥ 0, δ > 0.

        The first-passage-time interpretation: γ is the drift of the drifted Brownian motion
        γt + B_t, and δ is the level it must reach.

        Conversion to the (μ, λ) parametrization:
            μ = δ / γ   (mean)
            λ = δ²      (shape / precision)

        Args:
            gamma: drift parameter (γ ≥ 0)
            delta: level parameter (δ > 0)

        Returns:
            DistInvGauss instance with mu=delta/gamma, lam=delta**2
        """
        return cls(mu=delta / gamma, lam=delta**2)

    def mvsk(self):
        """
        Mean, coefficient of variation squared (coef_var = var/mean²), skewness, excess kurtosis.

        Returns:
            (mean, coef_var, skewness, excess_kurtosis)
        """
        mean = self.mu
        coef_var = self.mu / self.lam
        skew = 3.0 * np.sqrt(self.mu / self.lam)
        exkurt = 15.0 * self.mu / self.lam
        return mean, coef_var, skew, exkurt

    def mc4(self):
        """
        First four central moments: (mean, variance, mc3, mc4).

        Returns:
            (m1, mc2, mc3, mc4)
        """
        m1, cv, skew, exkurt = self.mvsk()
        var = cv * m1**2
        mc3 = skew * var**1.5
        mc4 = (exkurt + 3.0) * var**2
        return m1, var, mc3, mc4

    def fit(self, mean, coef_var):
        """
        Calibrate mu and lam from mean and coefficient of variation.

        Args:
            mean: distribution mean (> 0)
            coef_var: coefficient of variation squared (variance / mean²); equals mu/lam
        """
        self.mu = mean
        self.lam = mean / coef_var

    def quad(self, n_quad):
        """
        Inverse Gaussian quadrature nodes and weights (Choi et al. 2021).

        Args:
            n_quad: number of quadrature points.

        Returns:
            (nodes, weights)

        Examples:
            >>> mu, lam = 2, 1.5
            >>> x, w = DistInvGauss(mu=mu, lam=lam).quad(9)
            >>> sum(x*w), mu           # mean of IG
            >>> sum(w/x), 1/mu + 1/lam  # mean of 1/X
        """
        z, w, w_sum = spsp.roots_hermitenorm(n_quad, mu=True)

        fac = 0.5 * self.mu / self.lam
        y_hat = np.square(z) * fac

        # Always compute the large root first (both terms positive → no cancellation).
        # For z < 0 the direct formula gives the small root via subtraction, which
        # loses precision for large |z|.  Use x_small = 1/x_large instead
        # (the two IG roots satisfy x_small · x_large = 1).
        x = 1.0 + y_hat + np.abs(z) * np.sqrt(fac * (2.0 + y_hat))  # large root, stable
        neg = z < 0
        x[neg] = 1.0 / x[neg]                                         # invert to small root

        w *= 2.0 / (w_sum * (1.0 + x))
        x *= self.mu

        return x, w


class DistGig:
    """
    Generalized Inverse Gaussian (GIG) distribution.

    The PDF is:
        f(x) ∝ x^(p-1) * exp(-(gamma^2*x + delta^2/x) / 2),  x > 0

    Parameters:
        gamma (> 0): scale parameter
        delta (> 0): scale parameter
        p: index parameter (real)

    Special cases:
        p = -1/2 : Inverse Gaussian (IG)
        delta → 0: Gamma distribution (p > 0)
        gamma → 0: Inverse Gamma distribution (p < 0)

    References:
        https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution
        Choi J, Du Y, Song Q (2021) Inverse Gaussian quadrature and finite
        normal-mixture approximation of the generalized hyperbolic distribution.
        Journal of Computational and Applied Mathematics 388:113302.
        https://doi.org/10.1016/j.cam.2020.113302
    """

    gamma = 1.0
    delta = 1.0
    p = -0.5

    def __init__(self, gamma=1.0, delta=1.0, p=-0.5):
        self.gamma = gamma
        self.delta = delta
        self.p = p

    @classmethod
    def from_mv(cls, mean, coef_var, p=-0.5):
        """
        Construct from mean and coefficient of variation (coef_var = variance/mean²) with fixed p.

        Args:
            mean: distribution mean (> 0)
            coef_var: coefficient of variation squared (variance / mean²); > 0
            p: index parameter (default -0.5, i.e. Inverse Gaussian)

        Returns:
            DistGig instance calibrated to (mean, coef_var) with given p
        """
        obj = cls(p=p)
        obj.fit(mean, coef_var)
        return obj

    def _moment_r(self, r):
        """r-th raw moment: (delta/gamma)^r * K_{p+r}(gamma*delta) / K_p(gamma*delta).

        Uses kve (exponentially scaled Bessel) so the exp(q) factors cancel,
        giving numerically stable ratios for all q = gamma*delta > 0.
        """
        q = self.gamma * self.delta
        return (self.delta / self.gamma) ** r * spsp.kve(self.p + r, q) / spsp.kve(self.p, q)

    def mvsk(self):
        """
        Mean, coefficient of variation squared (coef_var = var/mean²), skewness, excess kurtosis.

        Returns:
            (mean, coef_var, skewness, excess_kurtosis)
        """
        m1 = self._moment_r(1)
        m2 = self._moment_r(2)
        m3 = self._moment_r(3)
        m4 = self._moment_r(4)
        var = m2 - m1**2
        mc3 = m3 - 3 * m2 * m1 + 2 * m1**3
        mc4 = m4 - 4 * m3 * m1 + 6 * m2 * m1**2 - 3 * m1**4
        skew = mc3 / var**1.5
        exkurt = mc4 / var**2 - 3.0
        coef_var = var / m1**2
        return m1, coef_var, skew, exkurt

    def mc4(self):
        """
        First four central moments: (mean, variance, mc3, mc4).

        Returns:
            (m1, mc2, mc3, mc4)
        """
        m1, cv, skew, exkurt = self.mvsk()
        var = cv * m1**2
        mc3 = skew * var**1.5
        mc4 = (exkurt + 3.0) * var**2
        return m1, var, mc3, mc4

    def fit(self, mean, coef_var):
        """
        Calibrate gamma and delta from mean and coefficient of variation with p fixed.

        coef_var = variance/mean² depends only on the product η = gamma*delta (for fixed p),
        which is found by a scalar root-find.  The scale rho = delta/gamma is then set from
        the mean.

        Args:
            mean: distribution mean (> 0)
            coef_var: coefficient of variation squared (variance / mean²); > 0
        """
        def cv2_err(eta):
            # kve = kv * exp(eta); exp factors cancel in ratios → numerically stable
            kve_p = spsp.kve(self.p, eta)
            r1 = spsp.kve(self.p + 1, eta) / kve_p
            r2 = spsp.kve(self.p + 2, eta) / kve_p
            return (r2 - r1**2) / r1**2 - coef_var

        # coef_var is decreasing in η; bracket: large η → coef_var ≈ 0, small η → coef_var large
        eta = spop.brentq(cv2_err, 1e-6, 1e4)

        kve_p = spsp.kve(self.p, eta)
        r1 = spsp.kve(self.p + 1, eta) / kve_p
        rho = mean / r1  # delta / gamma
        self.gamma = np.sqrt(eta / rho)
        self.delta = np.sqrt(eta * rho)

    def quad(self, n_quad, correct=False):
        """
        GIG quadrature nodes and weights (Choi et al. 2021).

        Delegates to DistInvGauss quadrature on the reparametrized IG with
        mu = delta/gamma and lam = delta^2, then reweights by x^(p+0.5).

        Args:
            n_quad: number of quadrature points.
            correct: if True, normalize weights to sum to 1.

        Returns:
            (nodes, weights)

        Examples:
            >>> import scipy.special as spsp
            >>> gamma, delta, p = 1, 1, 0.2
            >>> x, w = DistGig(gamma=gamma, delta=delta, p=p).quad(8)
            >>> r = 0.3  # r-th moment
            >>> mom_r = (delta/gamma)**r * spsp.kv(p+r, gamma*delta) / spsp.kv(p, gamma*delta)
            >>> mom_r, np.sum(x**r * w)  # should match
        """
        x, w = DistInvGauss(mu=self.delta / self.gamma, lam=self.delta**2).quad(n_quad)
        w *= np.power(x, self.p + 0.5)

        if correct:
            w /= np.sum(w)
        else:
            # Use kve (exponentially scaled) so exp(+η) in kve cancels the exp(-η) factor,
            # giving a numerically stable ratio for all η = gamma*delta > 0.
            ratio = (
                np.power(self.gamma / self.delta, self.p)
                / self.delta
                / np.sqrt(2.0 / np.pi)
                / spsp.kve(self.p, self.gamma * self.delta)
            )
            w *= ratio

        return x, w
