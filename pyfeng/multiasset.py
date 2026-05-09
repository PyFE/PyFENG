import abc
import warnings

import scipy.stats as spst
import numpy as np
from .bsm import Bsm
from .norm import Norm
from .gamma import InvGam
from .nsvh import Nsvh1
from .opt_abc import OptABC
from .params import MaParams, SpreadParams
from pyfeng.quad import NdGHQ  # Not sure


class BsmSpreadKirk(SpreadParams, OptABC):
    """
    Kirk's approximation for spread option.

    References:
        - Kirk E (1995) Correlation in the energy markets. In: Managing Energy Price Risk, First. Risk Publications, London, pp 71–78

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmSpreadKirk((0.2, 0.3), rho=-0.5)
        >>> m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        array([22.15632247, 17.18441817, 12.98974214,  9.64141666,  6.99942072])
    """

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)

        fwd1 = fwd[..., 0] - np.minimum(strike, 0)
        fwd2 = fwd[..., 1] + np.maximum(strike, 0)

        sig1 = self.sigma[0] * fwd[..., 0] / fwd1
        sig2 = self.sigma[1] * fwd[..., 1] / fwd2
        sig_spd = np.sqrt(sig1 * (sig1 - 2.0 * self.rho * sig2) + sig2 ** 2)
        price = Bsm.price_formula(fwd2, fwd1, sig_spd, texp, cp=cp, is_fwd=True)
        return df * price


class BsmSpreadBjerksund2014(SpreadParams, OptABC):
    """
    Bjerksund & Stensland (2014)'s approximation for spread option.

    References:
        - Bjerksund P, Stensland G (2014) Closed form spread option valuation. Quantitative Finance 14:1785–1794. https://doi.org/10.1080/14697688.2011.617775

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmSpreadBjerksund2014((0.2, 0.3), rho=-0.5)
        >>> m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        array([22.13172022, 17.18304247, 12.98974214,  9.54431944,  6.80612597])
    """

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)

        fwd1 = fwd[..., 0]
        fwd2 = fwd[..., 1]

        std11 = self.sigma[0] ** 2 * texp
        std12 = self.sigma[0] * self.sigma[1] * texp
        std22 = self.sigma[1] ** 2 * texp

        aa = fwd2 + strike
        bb = fwd2 / aa
        std = np.sqrt(std11 - 2 * bb * self.rho * std12 + bb ** 2 * std22)

        d3 = np.log(fwd1 / aa)
        d1 = (d3 + 0.5 * std11 - bb * (self.rho * std12 - 0.5 * bb * std22)) / std
        d2 = (d3 - 0.5 * std11 + self.rho * std12 + bb * (0.5 * bb - 1) * std22) / std
        d3 = (d3 - 0.5 * std11 + 0.5 * bb ** 2 * std22) / std

        price = cp * (
            fwd1 * spst.norm.cdf(cp * d1)
            - fwd2 * spst.norm.cdf(cp * d2)
            - strike * spst.norm.cdf(cp * d3)
        )

        return df * price


class NormBasket(MaParams, OptABC):
    """
    Basket option pricing under the multiasset Bachelier model
    """

    @classmethod
    def init_spread(cls, sigma, rho=None, cor_m=None, cov_m=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Initialize an instance for spread option pricing.
        Convenience constructor that sets ``weight = [1, -1]``.

        Examples:
            >>> import numpy as np
            >>> import pyfeng as pf
            >>> m = pf.NormBasket.init_spread((20, 30), rho=-0.5, intr=0.05)
            >>> m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
            array([17.95676186, 13.74646821, 10.26669936,  7.47098719,  5.29057157])
        """
        return cls(sigma, rho=rho, cor_m=cor_m, cov_m=cov_m,
                   weight=np.array([1.0, -1.0]), intr=intr, divr=divr, is_fwd=is_fwd)

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        if fwd.shape[-1] != self.n_asset:
            raise ValueError(f"fwd last dimension {fwd.shape[-1]} does not match n_asset={self.n_asset}.")

        fwd_basket = fwd @ self.weight
        vol_basket = np.sqrt(self.weight @ self.cov_m @ self.weight)

        price = Norm.price_formula(
            strike, fwd_basket, vol_basket, texp, cp=cp, is_fwd=True
        )
        return df * price


class BsmBasketABC(MaParams, OptABC):
    """
    Abstract base class for BSM basket option models.

    Provides :meth:`price_mvsk` — the shared moment machinery for the basket
    $B = \\sum_i w_i F_i(T)$ under the multiasset Black-Scholes-Merton model.
    """

    def price_mvsk(self, fwd, texp, order=4):
        """
        Mean, coefficient of variation, skewness, and excess kurtosis of the basket.

        Each asset follows $F_i(T) = v_i \\exp(\\sigma_i Z_i \\sqrt{T} - \\frac{1}{2}\\sigma_i^2 T)$
        where $v_i = w_i F_i(0)$ and $Z_i$ are correlated standard normals with covariance
        $C_{ij} = \\sigma_i \\sigma_j \\rho_{ij}$.

        The $k$-th raw moment is:

        .. math::
            m_k = \\sum_{i_1,\\ldots,i_k} \\prod_p v_{i_p}
                  \\cdot \\exp\\!\\Bigl(\\sum_{p < q} C_{i_p i_q} \\cdot T\\Bigr)

        The $-\\mathrm{Var}(X_i)/2$ drift correction in each asset and the $+\\mathrm{Var}/2$
        diagonal of the MGF cancel exactly, leaving only the pairwise cross-covariance sum
        $\\sum_{p<q} C_{i_p i_q} T$.  Setting $P_{ij} = \\exp(C_{ij} T)$ (the full covariance
        matrix including diagonal $P_{ii} = \\exp(\\sigma_i^2 T)$), the moments become:

        .. math::
            m_2 &= \\sum_{ij} v_i v_j P_{ij} \\\\
            m_3 &= \\sum_{ijk} v_i v_j v_k P_{ij} P_{ik} P_{jk} \\\\
            m_4 &= \\sum_{ijkl} v_i v_j v_k v_l P_{ij} P_{ik} P_{il} P_{jk} P_{jl} P_{kl}

        $m_2$ is a single matrix product; $m_3$ uses two; $m_4$ uses a reshape-matmul-reshape
        on $N_{ijk} = P_{ik} P_{jk}$.  All are $O(n^k)$ but fully vectorised.

        The second return value is the coefficient of variation,
        $\\widetilde{\\beta} = \\sqrt{\\mathrm{Var}(B)} / E[B] = \\sqrt{m_2/m_1^2 - 1}$,
        consistent with :class:`DistLognormal`.

        Args:
            fwd: forward price array (n_asset,)
            texp: time to expiry
            order: 2 returns ``(mean, coef_var)``; 4 (default) returns ``(mean, coef_var, skew, exkurt)``

        Returns:
            ``(mean, coef_var)`` if *order* = 2, else ``(mean, coef_var, skew, exkurt)``
        """
        P = np.exp(self.cov_m * texp)                       # P[i,j] = exp(σ_i σ_j ρ_ij T)
        v = np.asarray(fwd).ravel() * self.weight           # weighted forwards, shape (n_asset,)

        m1 = v.sum()
        M2 = np.outer(v, v) * P             # M2[i,j] = v_i v_j P_ij
        m2 = M2.sum()
        coef_var = np.sqrt(m2 / m1**2 - 1)  # std(B) / E[B]

        if order == 2:
            return m1, coef_var

        var = (coef_var * m1)**2

        # m3 = Σ_{ijk} v_i v_j v_k P_ij P_ik P_jk
        #    = sum( (P * v) ⊙ (M2 @ P) )
        m3 = np.sum((P * v) * (M2 @ P))

        # m4 = Σ_{ijkl} v_i v_j v_k v_l P_ij P_ik P_il P_jk P_jl P_kl
        #    = Σ_{ij} M2_ij H_ij,  H_ij = Σ_{kl} N_ijk M2_kl N_ijl,  N[i,j,k] = P_ik P_jk
        n = len(v)
        N = P[:, None, :] * P[None, :, :]
        H = (N * (N.reshape(n * n, n) @ M2).reshape(n, n, n)).sum(-1)
        m4 = np.sum(M2 * H)

        skew = (m3 - m1 ** 3 - 3 * m2 * m1 + 3 * m1 ** 3) / var ** 1.5
        exkurt = (m4 - 3 * m1 ** 4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2) / var ** 2 - 3

        return m1, coef_var, skew, exkurt


class BsmBasketJu2002(BsmBasketABC):
    """
    Basket option pricing using Ju's (2002) Taylor-expansion of the characteristic
    function ratio around zero volatility, accurate to O(σ⁶).

    When ``correction_ju2002 = False`` the higher-order correction is skipped and
    the price reduces to the log-normal moment-matching approximation of
    Levy & Turnbull (1992), which is what :class:`BsmBasketLevy1992` uses.

    References:
        - Ju E (2002) Pricing Asian and Basket Options Via Taylor Expansion. Journal of
          Computational Finance 5(3):79–103
        - Levy E, Turnbull S (1992) Average intelligence. Risk 1992:53–57
        - Krekel M, de Kock J, Korn R, Man T-K (2004) An analysis of pricing methods for
          basket options. Wilmott Magazine 2004:82–89

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.arange(50, 151, 10)
        >>> m = pf.BsmBasketJu2002(sigma=0.4*np.ones(4), rho=0.5)
        >>> m.price(strike, spot=100*np.ones(4), texp=5)
        array([54.31, 47.48, 41.52, 36.36, 31.88, 28.01, 24.67, 21.77, 19.26, 17.07, 15.17])
    """

    correction_ju2002 = True

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)

        # Weighted forwards and covariance matrix scaled by texp
        fw = fwd * self.weight               # w_i * F_i,  shape (n_asset,)
        cov_t = self.cov_m * texp            # σ_i σ_j ρ_ij T,  shape (n,n)

        # First two moments of the basket (Levy 1992)
        m1, coef_var = self.price_mvsk(fwd, texp, order=2)
        std = np.sqrt(np.log1p(coef_var))
        base = df * Bsm.price_formula(strike, m1, std, texp=1.0, cp=cp, is_fwd=True)

        if not self.correction_ju2002:
            return base

        # ---- Ju (2002) correction terms -------------------------------------
        av_a = fw @ cov_t                    # Σ_j cov_t[i,j] F_j,  shape (..., n)

        u2_0 = m1**2                         # u2(z=0) = (Σ w_i F_i)²
        u2_1 = np.sum(av_a * fw, axis=-1)              # d u2/d(z²) at z=0
        u2_2 = np.sum((fw @ cov_t**2) * fw, axis=-1)   # d²/d(z²)² (elementwise **)
        u2_3 = np.sum((fw @ cov_t**3) * fw, axis=-1)   # d³         (elementwise **)

        # Taylor coefficients a1, a2, a3 of the CF ratio at z=1
        a1 = -u2_1 / (2 * u2_0)
        a2 = 2 * a1**2 - u2_2 / (2 * u2_0)
        a3 = 6 * a1 * a2 - 4 * a1**3 - u2_3 / (2 * u2_0)

        # Composite moment quantities
        e_a12_a2 = 2 * np.sum(fw * av_a**2, axis=-1)
        e_a13_a3 = 6 * np.sum(fw * av_a**3, axis=-1)
        e_a12_a22 = 8 * np.sum(((av_a * fw) @ cov_t) * (av_a * fw), axis=-1) + 2 * u2_1 * u2_2
        e_a1_a2_a3 = 6 * np.sum((fw @ cov_t**2) * (av_a * fw), axis=-1)
        sqrt_fw = np.sqrt(fw)
        temp  = sqrt_fw[..., :, None] * cov_t * sqrt_fw[..., None, :]  # (..., n, n)
        e_a23 = 8 * np.einsum('...ij,...jk,...ki->...', temp, temp, temp)

        # b coefficients at z=1
        b1 = e_a12_a2 / (4 * m1**3)
        b2 = a1**2 - a2 / 2

        # c coefficients at z=1
        c1 = -a1 * b1
        c2 = (9 * e_a12_a22 + 4 * e_a13_a3) / (144 * m1**4)
        c3 = (4 * e_a1_a2_a3 + e_a23) / (48 * m1**3)
        c4 = a1 * a2 - 2 * a1**3 / 3 - a3 / 6

        # jd2/jd3/jd4: Ju's d_i(z) coefficients evaluated at z=1  (d1 unused)
        jd2 = (0.5 * (10 * a1**2 + a2 - 6 * b1 + 2 * b2)
               - (128 * a1**3 / 3 - a3 / 6 + 2 * a1 * b1 - a1 * b2
                  + 50 * c1 - 11 * c2 + 3 * c3 - c4))
        jd3 = (2 * a1**2 - b1
               - (88 * a1**3 + 3 * a1 * (5 * b1 - 2 * b2)
                  + 3 * (35 * c1 - 6 * c2 + c3)) / 3)
        jd4 = -20 * a1**3 / 3 + a1 * (-4 * b1 + b2) - 10 * c1 + c2

        z1 = jd2 - jd3 + jd4
        z2 = jd3 - jd4
        z3 = jd4

        d2 = np.log(m1/strike)/std - 0.5 * std             # BSM d₂ = (m(1) - log K) / √v(1)
        p_y = spst.norm.pdf(d2) / std
        dp_y = p_y * d2 / std
        d2p_y = p_y * (d2**2 - 1) / std**2

        return base + df * strike * (z1 * p_y + z2 * dp_y + z3 * d2p_y)


class BsmBasketLevy1992(BsmBasketJu2002):
    """
    Basket option pricing with the log-normal moment-matching approximation of
    Levy & Turnbull (1992).  Equivalent to :class:`BsmBasketJu2002` with the
    higher-order correction disabled.

    References:
        - Levy E, Turnbull S (1992) Average intelligence. Risk 1992:53–57
        - Krekel M, de Kock J, Korn R, Man T-K (2004) An analysis of pricing methods for
          basket options. Wilmott Magazine 2004:82–89

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.arange(50, 151, 10)
        >>> m = pf.BsmBasketLevy1992(sigma=0.4*np.ones(4), rho=0.5)
        >>> m.price(strike, spot=100*np.ones(4), texp=5)
        array([54.34281026, 47.521086  , 41.56701301, 36.3982413 , 31.92312156,
               28.05196621, 24.70229571, 21.800801  , 19.28360474, 17.09570196,
               15.19005654])
    """

    correction_ju2002 = False


class BsmBasketMilevsky1998(BsmBasketABC):
    """
    Basket option pricing with the inverse gamma distribution of Milevsky & Posner (1998)

    References:
        - Milevsky MA, Posner SE (1998) A Closed-Form Approximation for Valuing Basket Options. The Journal of Derivatives 5:54–61. https://doi.org/10.3905/jod.1998.408005
        - Krekel M, de Kock J, Korn R, Man T-K (2004) An analysis of pricing methods for basket options. Wilmott Magazine 2004:82–89

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.arange(50, 151, 10)
        >>> m = pf.BsmBasketMilevsky1998(sigma=0.4*np.ones(4), rho=0.5)
        >>> m.price(strike, spot=100*np.ones(4), texp=5)
        array([51.93069524, 44.40986   , 38.02596564, 32.67653542, 28.21560931,
               24.49577509, 21.38543199, 18.77356434, 16.56909804, 14.69831445,
               13.10186928])
    """

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        if fwd.shape[-1] != self.n_asset:
            raise ValueError(f"fwd last dimension {fwd.shape[-1]} does not match n_asset={self.n_asset}.")

        m1, coef_var = self.price_mvsk(fwd, texp, order=2)

        alpha = 1 / coef_var + 2
        beta = (alpha - 1) * m1

        price = InvGam.price_formula(
            strike, m1, texp, alpha, beta, cp=cp, is_fwd=True
        )
        return df * price


class BsmMax2(SpreadParams, OptABC):
    """
    Option on the max of two assets.
    Payout = max( max(F_1, F_2) - K, 0 ) for all or  max( K - max(F_1, F_2), 0 ) for put option

    References:
        - Rubinstein M (1991) Somewhere Over the Rainbow. Risk 1991:63–66

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmMax2(0.2*np.ones(2), rho=0, divr=0.1, intr=0.05)
        >>> m.price(strike=[90, 100, 110], spot=100*np.ones(2), texp=3)
        array([15.86717049, 11.19568103,  7.71592217])
    """

    m_switch = None

    def __post_init__(self):
        super().__post_init__()
        self.m_switch = BsmSpreadKirk(self.sigma, rho=self.rho, is_fwd=True)

    def price(self, strike, spot, texp, cp=1):
        sig = self.sigma
        fwd, df, _ = self._fwd_factor(spot, texp)

        sig_std = sig * np.sqrt(texp)
        spd_rho = np.sqrt(np.dot(sig, sig) - 2 * self.rho * sig[0] * sig[1])
        spd_std = spd_rho * np.sqrt(texp)

        # -x and y as rows
        # supposed to be -log(fwd/strike) but strike is added later
        xx = -np.log(fwd) / sig_std - 0.5 * sig_std

        fwd_ratio = fwd[0] / fwd[1]
        yy = np.log([fwd_ratio, 1 / fwd_ratio]) / spd_std + 0.5 * spd_std

        rho12 = (
            np.array([self.rho * sig[1] - sig[0], self.rho * sig[0] - sig[1]]) / spd_rho
        )

        cor_m = np.array([[1.0, self.rho], [self.rho, 1.0]])
        mu0 = np.zeros(2)
        cor_m1 = rho12[0] + (1 - rho12[0]) * np.eye(2)
        cor_m2 = rho12[1] + (1 - rho12[1]) * np.eye(2)

        strike_isscalar = np.isscalar(strike)
        strike = np.atleast_1d(strike)
        cp = cp * np.ones_like(strike)
        n_strike = len(strike)

        # this is the price of max(S1, S2) = max(S1-S2, 0) + S2
        # Used that Kirk approximation strike = 0 is Margrabe's switch option price
        parity = 0 if np.all(cp > 0) else self.m_switch.price(0, fwd, texp) + fwd[1]
        price = np.zeros_like(strike, float)
        for k in range(n_strike):
            xx_ = xx + np.log(strike[k]) / sig_std
            term1 = fwd[0] * (
                spst.norm.cdf(yy[0])
                - spst.multivariate_normal.cdf(np.array([xx_[0], yy[0]]), mu0, cor_m1)
            )
            term2 = fwd[1] * (
                spst.norm.cdf(yy[1])
                - spst.multivariate_normal.cdf(np.array([xx_[1], yy[1]]), mu0, cor_m2)
            )
            term3 = strike[k] * np.array(
                spst.multivariate_normal.cdf(xx_ + sig_std, mu0, cor_m)
            )

            if term1 + term2 + term3 < strike[k]:
                raise ValueError(f"Pricing error: lower bound violated at strike[{k}]={strike[k]}.")
            price[k] = term1 + term2 + term3 - strike[k]

            if cp[k] < 0:
                price[k] += strike[k] - parity
        price *= df

        return price[0] if strike_isscalar else price


class BsmBasket1Bm(BsmBasketABC):
    """
    Multiasset BSM model for pricing basket/Spread options when all asset prices are driven by a single Brownian motion (BM).

    """

    rho: float = 1.0

    @staticmethod
    def root(fac, std, strike):
        """
        Calculate the root x of f(x) = sum(fac * exp(std*x)) - strike = 0 using Newton's method

        Each fac and std should have the same signs so that f(x) is a monotonically increasing function.

        fac: factor to the exponents. (n_asset, ) or (n_strike, n_asset). Asset takes the last dimension.
        std: total standard variance. (n_asset, )
        strike: strike prices. scalar or (n_asset, )
        """

        if not np.all(fac * std >= 0.0):
            raise ValueError("fac * std must be non-negative for all assets.")
        log = np.min(fac) > 0  # Basket if log=True, spread if otherwise.
        scalar_output = np.isscalar(np.sum(fac * std, axis=-1) - strike)
        strike = np.atleast_1d(strike)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_k = np.where(strike > 0, np.log(strike), 1)

            # Initial guess with linearlized assmption
            x = (strike - np.sum(fac, axis=-1)) / np.sum(fac * std, axis=-1)
            if log:
                np.fmin(x, np.amin(np.log(strike[:, None] / fac) / std, axis=-1), out=x)
            else:
                np.clip(x, -3, 3, out=x)

        # Test x=-9 and 9 for min/max values.
        y_max = np.exp(9 * std)
        y_min = np.sum(fac / y_max, axis=-1) - strike
        y_max = np.sum(fac * y_max, axis=-1) - strike

        x[y_min >= 0] = -np.inf
        x[y_max <= 0] = np.inf
        ind = ~((y_min >= 0) | (y_max <= 0))

        if np.all(~ind):
            return x[0] if scalar_output else x

        for k in range(32):
            y_vec = fac * np.exp(std * x[ind, None])
            y = (
                np.log(np.sum(y_vec, axis=-1)) - log_k[ind]
                if log
                else np.sum(y_vec, axis=-1) - strike[ind]
            )
            dy = (
                np.sum(std * y_vec, axis=-1) / np.sum(y_vec, axis=-1)
                if log
                else np.sum(std * y_vec, axis=-1)
            )
            x[ind] -= y / dy
            if len(y) == 0:
                warnings.warn(f"Implied vol solver: unexpected empty residual at ind={ind}.")
            y_err_max = np.amax(np.abs(y))
            if y_err_max < BsmBasket1Bm.IMPVOL_TOL:
                break

        if y_err_max > BsmBasket1Bm.IMPVOL_TOL:
            warn_msg = (
                f"root did not converge within {k} iterations: max error = {y_err_max}"
            )
            warnings.warn(warn_msg, Warning)

        return x[0] if scalar_output else x

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        if fwd.shape[-1] != self.n_asset:
            raise ValueError(f"fwd last dimension {fwd.shape[-1]} does not match n_asset={self.n_asset}.")

        fwd_basket = fwd * self.weight
        sigma_std = self.sigma * np.sqrt(texp)
        cp = np.array(cp)
        d2 = -cp * self.root(
            fwd_basket * np.exp(-(sigma_std ** 2) / 2), sigma_std, strike
        )

        if np.isscalar(d2):
            d1 = d2 + cp * sigma_std
        else:
            d1 = d2[:, None] + np.atleast_1d(cp)[:, None] * sigma_std

        price = np.sum(fwd_basket * spst.norm.cdf(d1), axis=-1)
        price -= strike * spst.norm.cdf(d2)
        price *= cp * df
        return price


class BsmBasketJsu(BsmBasketABC):
    """

    Johnson's SU distribution approximation for Basket option pricing under the multiasset BSM model.

    Note: Johnson's SU distribution is the solution of NSVh with NSVh with lambda = 1.

    References:
        - Posner, S. E., & Milevsky, M. A. (1998). Valuing exotic options by approximating the SPD
        with higher moments. The Journal of Financial Engineering, 7(2). https://ssrn.com/abstract=108539

        - Choi, J., Liu, C., & Seo, B. K. (2019). Hyperbolic normal stochastic volatility model.
        Journal of Futures Markets, 39(2), 186–204. https://doi.org/10.1002/fut.21967

    """

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        if fwd.shape[-1] != self.n_asset:
            raise ValueError(f"fwd last dimension {fwd.shape[-1]} does not match n_asset={self.n_asset}.")

        fwd_basket = fwd @ self.weight
        m1, coef_var, skew, exkurt = self.price_mvsk(fwd, texp)

        m = Nsvh1(sigma=self.sigma)
        m.calibrate_vsk(coef_var * m1**2, skew, exkurt, texp, setval=True)
        price = m.price(strike, fwd_basket, texp, cp)

        return df * price


class BsmBasketChoi2018(BsmBasketABC):
    """
    Choi (2018)'s pricing method for Basket/Spread/Asian options

    References
        - Choi J (2018) Sum of all Black-Scholes-Merton models: An efficient pricing method for spread, basket, and Asian options. Journal of Futures Markets 38:627–644. https://doi.org/10.1002/fut.21909
    """

    n_quad = None
    lam = 4.0

    @classmethod
    def init_lowerbound(cls, sigma, rho=None, cor_m=None, cov_m=None, weight=None, intr=0.0, divr=0.0, is_fwd=False):
        m = cls(sigma, rho=rho, cor_m=cor_m, cov_m=cov_m, weight=weight, intr=intr, divr=divr, is_fwd=is_fwd)
        m.n_quad = 0
        return m

    def set_num_params(self, n_quad=None, lam=3.0):
        self.n_quad = n_quad
        self.lam = lam

    @staticmethod
    def householder(vv0):
        """
        Returns a Householder reflection (orthonormal matrix) that maps (1,0,...0) to vv0

        Args:
            vv0: vector

        Returns:
            Reflection matrix

        References
            - https://en.wikipedia.org/wiki/Householder_transformation
        """
        vv1 = vv0 / np.linalg.norm(vv0)
        vv1[0] -= 1.0

        if abs(vv1[0]) < np.finfo(float).eps*100:
            return np.eye(len(vv1))
        else:
            return np.eye(len(vv1)) + vv1[:, None] * vv1 / vv1[0]

    def v_mat(self, fwd):
        """
        Construct the V matrix

        Args:
            fwd: forward vector of assets

        Returns:
            V matrix
        """

        fwd_wts_unit = fwd * self.weight
        fwd_wts_unit /= np.linalg.norm(fwd_wts_unit)

        v1 = self.cov_m @ fwd_wts_unit
        v1 /= np.sqrt(np.sum(v1 * fwd_wts_unit))

        thres = 0.01 * self.sigma
        idx = (np.sign(fwd_wts_unit) * v1 < thres)

        if np.any(idx):
            v1[idx] = (np.sign(fwd_wts_unit) * thres)[idx]
            q1 = np.linalg.solve(self.chol_m, v1)
            q1norm = np.linalg.norm(q1)
            q1 /= q1norm
            v1 /= q1norm
        else:
            q1 = self.chol_m.T @ fwd_wts_unit
            q1 /= np.linalg.norm(q1)

        r_mat = self.householder(q1)

        chol_r_mat = self.chol_m @ r_mat[:, 1:]
        svd_u, svd_d, _ = np.linalg.svd(chol_r_mat, full_matrices=False)

        v_mat = np.hstack((v1[:, None], svd_u @ np.diag(svd_d)))
        len_scale = svd_d / np.sum(fwd_wts_unit * v1)

        if self.n_quad is None:
            n_quad = np.rint(self.lam * len_scale + 1).astype(int).tolist()
        else:
            n_quad = self.n_quad

        return v_mat, n_quad

    def v1_fwd_weight(self, fwd, texp):
        """
        Construct v1, forward array, and weights

        Args:
            fwd: forward vector of assets
            texp: time to expiry

        Returns:
            (v1, f_k, ww)
        """

        v_mat, n_quad = self.v_mat(fwd)
        v_mat *= np.sqrt(texp)

        v1 = v_mat[:, 0]

        if n_quad == 0:
            # 1 factor BM model for lower bound
            f_k = np.ones((1, self.n_asset))
            ww = np.array([1.0])
        else:
            v_mat = v_mat[:, 1:len(n_quad)+1]
            quad = NdGHQ(n_quad)
            zz, ww = quad.z_vec_weight()
            f_k = np.exp(zz @ v_mat.T - 0.5*np.sum(v_mat**2, axis=1))

        return v1, f_k, ww

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)

        v1, f_k, ww = self.v1_fwd_weight(fwd, texp)

        fw = fwd * self.weight                       # w_k F_k, shape (n_asset,)
        v1_half_sq = 0.5 * v1**2                     # ½ V_k1²

        scalar_strike = np.isscalar(strike)
        strike = np.atleast_1d(np.asarray(strike, dtype=float))

        price = np.zeros(len(strike))
        delta = np.zeros((len(strike), self.n_asset))  # call Δ_k (Eq. 14), shape (n_strike, n_asset)
        f_tilde = np.zeros(self.n_asset)               # f̃_k = Σ_m h_m f_k(ž_m)

        for f_k_row, wm in zip(f_k, ww):
            fac = f_k_row * fw * np.exp(-v1_half_sq)          # Eq. (8) LHS coefficients
            d_m = -BsmBasket1Bm.root(fac, v1, strike)          # d(ž_m) = −z₁*

            nd1 = spst.norm.cdf(d_m[:, None] + v1)             # N(d+V_k1), (n_strike, n_asset)
            nd1_cp = nd1 if cp == 1 else 1.0 - nd1             # N(cp·(d+V_k1))
            nd2_cp = spst.norm.cdf(cp * d_m)                    # N(cp·d), (n_strike,)

            # Price contribution: cp·(Σ_k fw_k f_k N(cp·(d+V_k1)) − K·N(cp·d))
            price += wm * cp * (np.sum(f_k_row * fw * nd1_cp, axis=-1) - strike * nd2_cp)

            # Call delta accumulation (Eq. 14): D_k = w_k Σ_m h_m f_k(ž_m) N(d+V_k1)
            delta += wm * f_k_row[None, :] * self.weight[None, :] * nd1

            f_tilde += wm * f_k_row

        # Forward price control variate correction (Eq. 17)
        fwd_err = fwd * (f_tilde - 1)                 # F_k (f̃_k − 1), shape (n_asset,)
        if cp == 1:
            # C' = C − Σ_k D_k F_k (f̃_k − 1)
            price -= np.sum(delta * fwd_err, axis=-1)
        else:
            # P' = P − Σ_k (D_k − w_k) F_k (f̃_k − 1)
            price -= np.sum((delta - self.weight) * fwd_err, axis=-1)

        price = df * price
        return price[0] if scalar_strike else price