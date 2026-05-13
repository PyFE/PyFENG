import numpy as np
import scipy.special as spsp
import scipy.optimize as spop
import scipy.stats as spst
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

    def __init__(self, sig=1.0, lam=1.0, mu=1.0, validate=False):
        self.sig = sig
        self.lam = lam
        self.mu = mu
        self.validate = validate

    @classmethod
    def from_mv(cls, mean, var_scaled, lam=1.0):
        """
        Construct from mean and scaled variance (var_scaled = var/mean²  = (std/mean)²).

        Args:
            mean: distribution mean
            var_scaled: scaled variance (var / mean²  = (std/mean)²)
            lam: lambda parameter (default 1.0, plain lognormal)

        Returns:
            DistLognormal instance calibrated to (mean, var_scaled)
        """
        obj = cls()
        obj.fit([mean, var_scaled], lam=lam)
        return obj

    def mvsk(self):
        """
        Mean, scaled variance (var_scaled = var/mean²  = (std/mean)²), skewness, excess kurtosis.

        Returns:
            (mean, var_scaled, skewness, excess_kurtosis)
        """
        ww = np.expm1(self.sig**2)
        var_scaled = self.lam**2 * ww
        skew = np.sqrt(ww) * (ww + 3)
        exkur = ww * (16 + ww * (15 + ww * (6 + ww)))
        return self.mu, var_scaled, skew, exkur

    def mc4(self):
        """
        First four central moments: (mean, variance, mc3, mc4).

        Returns:
            (m1, mc2, mc3, mc4)
        """
        m1, vs, s, k = self.mvsk()
        mc2 = vs * m1**2
        mc3 = s * mc2*np.sqrt(mc2)
        mc4 = (k + 3.0) * mc2**2
        return m1, mc2, mc3, mc4

    def fit(self, mvs, lam=None):
        """
        Calibrate parameters from moments.

        Args:
            mvs: (mean, var_scaled) or (mean, var_scaled, skewness).
                 var_scaled = var / mean²  = (std/mean)².
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
            ww = mvs[1] / self.lam**2
            self.sig = np.sqrt(np.log1p(ww))
        else:
            s = mvs[2]
            sqrt_w = 2 * np.sinh(np.arccosh(1 + 0.5 * s**2) / 6)
            self.lam = np.sqrt(mvs[1]) / sqrt_w
            self.sig = np.sqrt(np.log1p(sqrt_w**2))

        if self.validate:
            n = 2 if len(mvs) == 2 or lam is not None else 3
            mvsk2 = self.mvsk()
            for i, v in enumerate(mvs[:n]):
                if not np.isclose(v, mvsk2[i]):
                    raise ValueError(
                        f"Moment validation failed at index {i}: expected {v}, got {mvsk2[i]}."
                    )

    def quad(self, n_quad, return_zhat=False):
        """
        Gauss-Hermite quadrature nodes and weights with a Girsanov shift.

        The lognormal is parameterised by Z ~ N(0,1):
            Y = mu * (1 - lam + lam * exp(sig*Z - sig^2/2))

        Standard Gauss-Hermite quadrature approximates E_{N(0,1)}[f(Z)] using nodes and
        weights w_i with sum(w_i) = 1. A naive application would place the nodes at Z = z_i
        with x_i = mu * exp(sig*z_i - sig^2/2) and unmodified weights, which is correct but
        numerically inefficient for large sig (the integrand mass is far from the N(0,1) nodes).

        Instead we apply a Girsanov shift nu = sig/2, defining the shifted variable
            Zhat = Z - sig/2,
        so that under the new measure Zhat ~ N(0,1) and Hermite nodes zhat_i place the
        evaluation points where the lognormal integrand has most of its mass.

        Substituting Z = Zhat + sig/2 into the lognormal:
            Y = mu * (1 - lam + lam * exp(sig*Z - sig^2/2))
              = mu * (1 - lam + lam * exp(sig*Zhat))         [sig^2/2 cancels]

        The weight correction (RN derivative dN(0,1)/dN(sig/2,1)) is:
            w_tilde_i = w_i * exp(-sig*zhat_i/2 - sig^2/8)

        Before normalisation, using the MGF of N(0,1) and holding to quadrature accuracy:
            sum(w_tilde_i)       = exp(-sig^2/8) * E[exp(-sig*Zhat/2)] = exp(-sig^2/8)*exp(+sig^2/8) = 1
            sum(w_tilde_i * x_i) = mu * exp(-sig^2/8) * E[exp(+sig*Zhat/2)] = mu*exp(-sig^2/8)*exp(+sig^2/8) = mu

        The normalisation w /= sum(w) makes both properties hold EXACTLY for any n_quad.
        The key is that Hermite nodes are symmetric (zhat_i and -zhat_i paired with equal weights), so:
            C := sum(w_i * exp(-sig*zhat_i/2)) = sum(w_i * exp(+sig*zhat_i/2))
        giving sum(w_tilde_i) = C and sum(w_tilde_i * x_i) = mu * C. After dividing by C:
            sum(w_i)       = 1
            sum(w_i * x_i) = mu

        Args:
            n_quad: number of quadrature points.
            return_zhat: if True, also return the shifted standard-normal nodes
                zhat_i = Z_i - sig/2 (Hermite nodes). Default False.

        Returns:
            (x, w) if return_zhat is False, else (x, w, zhat). Always
            sum(w_i) = 1 and sum(w_i * x_i) = mu.
        """
        zhat, w = spsp.roots_hermitenorm(n_quad)

        # Girsanov shift nu = sig/2: tmp = exp(sig*zhat/2), so tmp^2 = exp(sig*zhat)
        # sig may be an array (e.g. shape (n0, 1)), so use out-of-place ops for broadcasting.
        tmp = np.exp(0.5 * self.sig * zhat)               # (..., n_quad) via broadcasting
        w = w * (np.exp(-self.sig**2/8) / tmp)            # RN derivative: exp(-sig*zhat/2 - sig^2/8)
        # Normalisation is the key: Hermite symmetry gives sum(w*exp(-sig*zhat/2)) = sum(w*exp(+sig*zhat/2)) = C,
        # so sum(w_tilde) = C and sum(w_tilde * x) = mu*C. Dividing by C enforces BOTH exactly.
        w /= np.sum(w, axis=-1, keepdims=True)            # keepdims supports both scalar and array sig
        x = self.mu * (1 - self.lam + self.lam * tmp**2)  # mu * (1 - lam + lam*exp(sig*zhat))
        if return_zhat:
            return x, w, zhat
        return x, w

    def draw(self, n_sample=None, rng=None):
        """
        Draw random samples from the shifted lognormal distribution.

        Distribution parameters (sig, mu, lam) may be numpy arrays and are
        broadcast against the generated standard-normal draws.

        Args:
            n_sample: number of samples (int or shape tuple passed to standard_normal)
            rng: numpy Generator (e.g. np.random.default_rng(seed)). If None, a fresh one is created.

        Returns:
            array of shape (*param_broadcast_shape, n_sample)
        """
        if rng is None:
            rng = np.random.default_rng()
        z = rng.standard_normal(n_sample)
        return self.mu * (1 - self.lam + self.lam * np.exp(self.sig * z - 0.5 * self.sig**2))


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
    def from_mv(cls, mean, var_scaled):
        """
        Construct from mean and scaled variance (var_scaled = var/mean²  = (std/mean)²).

        Args:
            mean: distribution mean (> 0)
            var_scaled: scaled variance (var / mean²  = (std/mean)²); equals 1/shape

        Returns:
            DistGamma instance with shape = 1/var_scaled, rate = 1/(var_scaled * mean)
        """
        obj = cls()
        obj.fit(mean, var_scaled)
        return obj

    @property
    def scale(self):
        return 1.0 / self.rate

    def mvsk(self):
        """
        Mean, scaled variance (var_scaled = var/mean²  = (std/mean)²), skewness, excess kurtosis.

        Returns:
            (mean, var_scaled, skewness, excess_kurtosis)
        """
        mean = self.shape / self.rate
        var_scaled = 1.0 / self.shape
        skew = 2.0 / np.sqrt(self.shape)
        exkurt = 6.0 / self.shape
        return mean, var_scaled, skew, exkurt

    def mc4(self):
        """
        First four central moments: (mean, variance, mc3, mc4).

        Returns:
            (m1, mc2, mc3, mc4)
        """
        m1, var_scaled, skew, exkurt = self.mvsk()
        var = var_scaled * m1**2
        mc3 = skew * var*np.sqrt(var)
        mc4 = (exkurt + 3.0) * var**2
        return m1, var, mc3, mc4

    def fit(self, mean, var_scaled):
        """
        Calibrate shape and rate from mean and scaled variance.

        Args:
            mean: distribution mean (> 0)
            var_scaled: scaled variance (var / mean²  = (std/mean)²); > 0
        """
        self.shape = 1.0 / var_scaled
        self.rate = 1.0 / (var_scaled * mean)

    def scipy_stats(self):
        """
        Frozen scipy.stats.gamma distribution equivalent to this instance.

        Returns:
            scipy.stats.gamma(a=shape, scale=1/rate)
        """
        return spst.gamma(a=self.shape, scale=1.0 / self.rate)

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

    def draw(self, n_sample=None, rng=None):
        """
        Draw random samples from the Gamma distribution.

        Distribution parameters (shape, rate) may be numpy arrays; numpy
        broadcasts them against size and raises if they are inconsistent.

        Args:
            n_sample: number of samples (int or shape tuple passed to standard_gamma)
            rng: numpy Generator (e.g. np.random.default_rng(seed)). If None, a fresh one is created.

        Returns:
            array of shape (*param_broadcast_shape, n_sample)
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.standard_gamma(self.shape, size=n_sample) / self.rate


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
    def from_mv(cls, mean, var_scaled):
        """
        Construct from mean and scaled variance (var_scaled = var/mean²  = (std/mean)²).

        Args:
            mean: distribution mean (> 0)
            var_scaled: scaled variance (var / mean²  = (std/mean)²); equals mu/lam

        Returns:
            DistInvGauss instance with mu=mean, lam=mean/var_scaled
        """
        obj = cls()
        obj.fit(mean, var_scaled)
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
        Mean, scaled variance (var_scaled = var/mean²  = (std/mean)²), skewness, excess kurtosis.

        Returns:
            (mean, var_scaled, skewness, excess_kurtosis)
        """
        mean = self.mu
        var_scaled = self.mu / self.lam
        skew = 3.0 * np.sqrt(self.mu / self.lam)
        exkurt = 15.0 * self.mu / self.lam
        return mean, var_scaled, skew, exkurt

    def mc4(self):
        """
        First four central moments: (mean, variance, mc3, mc4).

        Returns:
            (m1, mc2, mc3, mc4)
        """
        m1, var_scaled, skew, exkurt = self.mvsk()
        var = var_scaled * m1**2
        mc3 = skew * var*np.sqrt(var)
        mc4 = (exkurt + 3.0) * var**2
        return m1, var, mc3, mc4

    def fit(self, mean, var_scaled):
        """
        Calibrate mu and lam from mean and scaled variance.

        Args:
            mean: distribution mean (> 0)
            var_scaled: scaled variance (var / mean²  = (std/mean)²); equals mu/lam
        """
        self.mu = mean
        self.lam = mean / var_scaled

    def scipy_stats(self):
        """
        Frozen scipy.stats.invgauss distribution equivalent to this instance.

        scipy.stats.invgauss(mu_sp, scale=s) has mean = s * mu_sp and
        variance = s² * mu_sp³.  Setting mu_sp = mu/lam and s = lam recovers
        the standard IG(mu, lam) moments: E = mu, Var = mu³/lam.

        Returns:
            scipy.stats.invgauss(mu=mu/lam, scale=lam)
        """
        return spst.invgauss(mu=self.mu / self.lam, scale=self.lam)

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

    def draw(self, n_sample=None, rng=None):
        """
        Draw random samples from the Inverse Gaussian distribution.

        Distribution parameters (mu, lam) may be numpy arrays; numpy
        broadcasts them against size and raises if they are inconsistent.

        Args:
            n_sample: number of samples (int or shape tuple passed to wald)
            rng: numpy Generator (e.g. np.random.default_rng(seed)). If None, a fresh one is created.

        Returns:
            array of shape (*param_broadcast_shape, n_sample)
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.wald(self.mu, self.lam, size=n_sample)


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
    def from_mv(cls, mean, var_scaled, p=-0.5):
        """
        Construct from mean and scaled variance (var_scaled = var/mean²  = (std/mean)²) with fixed p.

        Args:
            mean: distribution mean (> 0)
            var_scaled: scaled variance (var / mean²  = (std/mean)²); > 0
            p: index parameter (default -0.5, i.e. Inverse Gaussian)

        Returns:
            DistGig instance calibrated to (mean, var_scaled) with given p
        """
        obj = cls(p=p)
        obj.fit(mean, var_scaled)
        return obj

    def mnc(self, r):
        """Non-central (raw) moments at orders r: (delta/gamma)^r * K_{p+r}(gamma*delta) / K_p(gamma*delta).

        r may be a scalar or array.  Uses kve (exponentially scaled Bessel) so the
        exp(q) factors cancel, giving numerically stable ratios for all q = gamma*delta > 0.
        """
        q = self.gamma * self.delta
        r = np.asarray(r)
        return (self.delta / self.gamma) ** r * spsp.kve(self.p + r, q) / spsp.kve(self.p, q)

    def mvsk(self):
        """
        Mean, scaled variance (var_scaled = var/mean²  = (std/mean)²), skewness, excess kurtosis.

        Returns:
            (mean, var_scaled, skewness, excess_kurtosis)
        """
        m1, m2, m3, m4 = self.mnc([1, 2, 3, 4])
        var = m2 - m1**2
        mc3 = m3 - 3 * m2 * m1 + 2 * m1**3
        mc4 = m4 - 4 * m3 * m1 + 6 * m2 * m1**2 - 3 * m1**4
        skew = mc3 / (var*np.sqrt(var))
        exkurt = mc4 / var**2 - 3.0
        var_scaled = var / m1**2
        return m1, var_scaled, skew, exkurt

    def mc4(self):
        """
        First four central moments: (mean, variance, mc3, mc4).

        Returns:
            (m1, mc2, mc3, mc4)
        """
        m1, var_scaled, skew, exkurt = self.mvsk()
        var = var_scaled * m1**2
        mc3 = skew * var*np.sqrt(var)
        mc4 = (exkurt + 3.0) * var**2
        return m1, var, mc3, mc4

    def fit(self, mean, var_scaled):
        """
        Calibrate gamma and delta from mean and scaled variance with p fixed.

        var_scaled = var/mean² depends only on the product η = gamma*delta (for fixed p),
        which is found by a scalar root-find.  The scale rho = delta/gamma is then set from
        the mean.

        Args:
            mean: distribution mean (> 0)
            var_scaled: scaled variance (var / mean²  = (std/mean)²); > 0
        """
        def cv2_err(eta):
            # kve = kv * exp(eta); exp factors cancel in ratios → numerically stable
            kve_p = spsp.kve(self.p, eta)
            r1 = spsp.kve(self.p + 1, eta) / kve_p
            r2 = spsp.kve(self.p + 2, eta) / kve_p
            return (r2 - r1**2) / r1**2 - var_scaled

        # var_scaled is decreasing in η; bracket: large η → var_scaled ≈ 0, small η → var_scaled large
        eta = spop.brentq(cv2_err, 1e-6, 1e4)

        kve_p = spsp.kve(self.p, eta)
        r1 = spsp.kve(self.p + 1, eta) / kve_p
        rho = mean / r1  # delta / gamma
        self.gamma = np.sqrt(eta / rho)
        self.delta = np.sqrt(eta * rho)

    def scipy_stats(self):
        """
        Frozen scipy.stats.geninvgauss distribution equivalent to this instance.

        scipy.stats.geninvgauss(p, b, scale=s) has PDF ∝ x^(p−1) exp(−b(x+1/x)/2)
        scaled by s.  Setting b = gamma*delta and scale = delta/gamma matches the
        GIG(gamma, delta, p) PDF ∝ x^(p−1) exp(−(gamma²x + delta²/x)/2).

        Returns:
            scipy.stats.geninvgauss(p=p, b=gamma*delta, scale=delta/gamma)
        """
        return spst.geninvgauss(p=self.p, b=self.gamma * self.delta, scale=self.delta / self.gamma)

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


class DistGh:
    """
    Generalized Hyperbolic (GH) distribution.

    Normal variance-mean mixture:
        X = mu + beta*W + sqrt(W)*Z,  Z ~ N(0,1),  W ~ GIG(gamma, delta, p)

    Uses the (gamma, delta) parametrization from Choi et al. (2021), where
    gamma and delta are the GIG parameters directly.  The Wikipedia-style
    steepness parameter alpha is a derived property: alpha = sqrt(gamma^2 + beta^2).

    Parameters:
        mu    (real):  location shift
        beta  (real):  asymmetry / skewness weight
        gamma (> 0):   GIG concentration; controls tail decay
        delta (> 0):   GIG scale
        p     (real):  GIG index; p = -1/2 gives NIG, p = 1 gives hyperbolic

    References:
        Choi J, Du Y, Song Q (2021) Inverse Gaussian quadrature and finite
        normal-mixture approximation of the generalized hyperbolic distribution.
        Journal of Computational and Applied Mathematics 388:113302.
        https://doi.org/10.1016/j.cam.2020.113302
    """

    mu = 0.0
    beta = 0.0
    gamma = 1.0
    delta = 1.0
    p = -0.5
    _n_quad = 32  # backing store for n_quad property
    quad_x = None  # cached quadrature nodes; recomputed when n_quad is set
    quad_w = None  # cached quadrature weights; recomputed when n_quad is set

    def __init__(self, mu=0.0, beta=0.0, gamma=1.0, delta=1.0, p=-0.5, n_quad=32):
        self.mu = mu
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.p = p
        self.n_quad = n_quad  # triggers setter → computes quad_x, quad_w

    @property
    def n_quad(self):
        return self._n_quad

    @n_quad.setter
    def n_quad(self, n):
        self._n_quad = n
        self.quad_x, self.quad_w = self.mixer().quad(n)

    @property
    def alpha(self):
        """Steepness: alpha = sqrt(gamma^2 + beta^2)."""
        return np.sqrt(self.gamma**2 + self.beta**2)

    @property
    def sigma(self):
        """Concentration: sigma = sqrt(gamma * delta)."""
        return np.sqrt(self.gamma * self.delta)

    @property
    def beta_tilde(self):
        """Normalised asymmetry: beta_tilde = beta * sqrt(delta / gamma)."""
        return self.beta * np.sqrt(self.delta / self.gamma)

    def mixer(self):
        """GIG mixing distribution DistGig(gamma, delta, p)."""
        return DistGig(gamma=self.gamma, delta=self.delta, p=self.p)

    def scipy_stats(self):
        """
        Frozen scipy.stats.genhyperbolic distribution equivalent to this instance.

        The scipy parametrization (p, a, b, loc, scale) maps to Barndorff-Nielsen
        (λ, α, β, δ, μ) via: λ=p, α=a/scale, β=b/scale, δ=scale, μ=loc.
        With scale=delta, loc=mu, a=alpha*delta, b=beta*delta.

        Returns:
            scipy.stats.genhyperbolic(p=p, a=alpha*delta, b=beta*delta, loc=mu, scale=delta)
        """
        return spst.genhyperbolic(
            p=self.p,
            a=self.alpha * self.delta,
            b=self.beta * self.delta,
            loc=self.mu,
            scale=self.delta,
        )

    def mvsk(self):
        """
        Analytic mean, variance, skewness, excess kurtosis via NVM formulas.

        For X = mu + beta*W + sqrt(W)*Z with W ~ GIG(gamma, delta, p) and Z ~ N(0,1):

            E[X]    = mu + beta * m1
            Var[X]  = m1 + beta² (m2 - m1²)
            mc3(X)  = 3 beta (m2 - m1²) + beta³ (m3 - 3 m2 m1 + 2 m1³)
            mc4(X)  = 3 m2 + 6 beta² (m3 - 2 m1 m2 + m1³)
                           + beta⁴ (m4 - 4 m3 m1 + 6 m2 m1² - 3 m1⁴)

        where m_r = E[W^r] are the raw moments of the GIG mixer.

        References:
            https://en.wikipedia.org/wiki/Generalised_hyperbolic_distribution

        Returns:
            (mean, variance, skewness, excess_kurtosis)
        """
        m1, m2, m3, m4 = self.mixer().mnc([1, 2, 3, 4])
        beta = self.beta

        mc2_W = m2 - m1**2
        mc3_W = m3 - 3*m2*m1 + 2*m1**3
        mc4_W = m4 - 4*m3*m1 + 6*m2*m1**2 - 3*m1**4

        mean = self.mu + beta * m1
        mc2  = m1 + beta**2 * mc2_W
        mc3  = 3*beta*mc2_W + beta**3*mc3_W
        mc4  = 3*m2 + 6*beta**2*(m3 - 2*m1*m2 + m1**3) + beta**4*mc4_W

        return mean, mc2, mc3 / (mc2*np.sqrt(mc2)), mc4 / mc2**2 - 3

    def mvsk_quad(self):
        """
        Mean, variance, skewness, excess kurtosis via GIG quadrature (numerical).

        Computes central moments directly using the conditional normal formulas.
        For X = mu + beta*W + sqrt(W)*Z with c = E[X|W] - E[X]:

            mc2 = E[c² + W]
            mc3 = E[c³ + 3cW]
            mc4 = E[c⁴ + 6c²W + 3W²]

        Returns:
            (mean, variance, skewness, excess_kurtosis)
        """
        xw, ww = self.quad_x, self.quad_w
        a    = self.mu + self.beta * xw   # conditional mean E[X | W=xw]
        mean = ww @ a
        c    = a - mean                   # deviation from overall mean

        mc2 = ww @ (c**2 + xw)
        mc3 = ww @ (c**3 + 3*c*xw)
        mc4 = ww @ (c**4 + 6*c**2*xw + 3*xw**2)

        return mean, mc2, mc3 / (mc2*np.sqrt(mc2)), mc4 / mc2**2 - 3

    def mc4(self):
        """
        First four central moments: (mean, variance, mc3, mc4).

        Returns:
            (m1, mc2, mc3, mc4)
        """
        m1, mc2, skew, exkurt = self.mvsk()
        return m1, mc2, skew * mc2*np.sqrt(mc2), (exkurt + 3.0) * mc2**2

    def cdf(self, y):
        """
        CDF via GIG quadrature (Eq. 3 in Choi et al. 2021):
            F(y) = sum_k w_k * Phi((y - mu - beta*x_k) / sqrt(x_k))

        Args:
            y: scalar or array of evaluation points

        Returns:
            CDF values (same shape as y)
        """
        x, w = self.quad_x, self.quad_w
        scalar = np.ndim(y) == 0
        y = np.atleast_1d(y)
        z = (y[:, None] - self.mu - self.beta * x[None, :]) / np.sqrt(x[None, :])
        out = spst.norm._cdf(z) @ w
        return out[0] if scalar else out

    def ppf(self, q):
        """
        Percent-point function (inverse CDF) at probability level q.

        Uses brentq; accuracy governed by self.n_quad.

        Args:
            q: probability level (0 < q < 1), scalar

        Returns:
            y such that F(y) ≈ q
        """
        x_r, w_r = self.quad_x, self.quad_w
        mu, beta = self.mu, self.beta
        def cdf_f(y):
            return w_r @ spst.norm._cdf((y - mu - beta * x_r) / np.sqrt(x_r))
        m1  = mu + beta * (w_r @ x_r)
        mc2 = (w_r @ x_r) + beta**2 * ((w_r @ x_r**2) - (w_r @ x_r)**2)
        std = np.sqrt(max(mc2, 1e-30))
        # Expand bracket until it straddles q (needed for heavy-tailed distributions)
        lo, hi = m1 - 20*std, m1 + 20*std
        for factor in [50, 100, 200, 500]:
            if cdf_f(lo) < q < cdf_f(hi):
                break
            lo, hi = m1 - factor*std, m1 + factor*std
        return spop.brentq(lambda y: cdf_f(y) - q, lo, hi, xtol=1e-12)


class DistNig(DistGh):
    """
    Normal Inverse Gaussian (NIG) distribution.

    Special case of the GH distribution with p = -1/2.  The GIG mixing
    distribution reduces to the inverse Gaussian IG(gamma, delta).

    Parameters:
        mu    (real):  location shift
        beta  (real):  asymmetry parameter
        gamma (> 0):   concentration; gamma = sqrt(alpha^2 - beta^2)
        delta (> 0):   scale

    References:
        Barndorff-Nielsen OE (1997) Normal inverse Gaussian distributions and
        stochastic volatility modelling. Scandinavian Journal of Statistics 24:1–13.
        https://doi.org/10.1111/1467-9469.00045
    """

    def __init__(self, mu=0.0, beta=0.0, gamma=1.0, delta=1.0, n_quad=32):
        super().__init__(mu=mu, beta=beta, gamma=gamma, delta=delta, p=-0.5, n_quad=n_quad)

    def mixer(self):
        """IG mixing distribution (exact NIG mixer; p = -1/2 special case of GIG)."""
        return DistInvGauss.from_gig(self.gamma, self.delta)

    def mvsk(self):
        """
        Analytic mean, variance, skewness, excess kurtosis for NIG.

        Uses closed-form expressions in terms of alpha = sqrt(gamma^2 + beta^2)
        and gamma = sqrt(alpha^2 - beta^2).

        References:
            https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution

        Returns:
            (mean, variance, skewness, excess_kurtosis)
        """
        alpha = self.alpha          # sqrt(gamma^2 + beta^2)
        gamma = self.gamma          # sqrt(alpha^2 - beta^2)
        beta  = self.beta
        delta = self.delta

        mean    = self.mu + delta * beta / gamma
        var     = delta * alpha**2 / gamma**3
        skew    = 3 * beta / (alpha * np.sqrt(delta * gamma))
        exkurt  = 3 * (1 + 4*beta**2 / alpha**2) / (delta * gamma)

        return mean, var, skew, exkurt
