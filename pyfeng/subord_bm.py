import numpy as np
from .bsm import Bsm
from .opt_abc import OptABC
from .params import VarGammaParams, NigParams
from .disthelper import DistGamma, DistInvGauss

class VarGammaABC(VarGammaParams, OptABC):
    """
    Abstract base for Variance Gamma models — provides ``logp_mgf`` and
    analytic ``logp_cum4``.

    Parameters, constraints, and the precomputed ``_mgf1_correction`` are
    defined in :class:`~pyfeng.params.VarGammaParams`.

    References:
        - Madan DB, Carr PP, Chang EC (1998) The Variance Gamma Process and Option
          Pricing. European Finance Review 2:79–105.
          https://doi.org/10.1023/A:1009703431535
        - Madan DB, Seneta E (1990) The Variance Gamma (V.G.) Model for Share
          Market Returns. Journal of Business 63:511–524.
          https://doi.org/10.1086/296519
    """

    def logp_mgf(self, uu, texp):
        """
        MGF of log price under the Variance Gamma model.

        The MGF of :math:`\\log(S_T/F_T)` at argument :math:`u` is

        .. math::

            \\exp\\!\\left(\\frac{T}{\\nu}\\left[
                u\\ln\\!\\left(1 - \\theta\\nu - \\tfrac{1}{2}\\sigma^2\\nu\\right)
                - \\ln\\!\\left(1 - \\theta\\nu u - \\tfrac{1}{2}\\sigma^2\\nu u^2\\right)
            \\right]\\right).

        Args:
            uu: MGF argument (scalar or array).
            texp: time to expiry.

        Returns:
            MGF value(s) at ``uu``, same shape as ``uu``.
        """
        # [Verified: Claude Sonnet 4.6, 2026-05-08]
        # VG log price: log(S_T/F_T) = ωT + θV_T + σW_{V_T},  V_T ~ Gamma(T/ν, ν).
        # E[exp(aV)] = (1 - aν)^{-T/ν}  (Gamma MGF), so:
        #   MGF(u) = exp(uωT) · (1 - θνu - σ²νu²/2)^{-T/ν}
        #          = exp(T/ν · [uων - log(1 - θνu - σ²νu²/2)])
        # _mgf1_correction = -ων = -log(1 - θν - σ²ν/2),  so -_mgf1_correction = ων.
        # rv = ων·u - log(1 - θνu - σ²νu²/2)  [via log1p for numerical stability]  ✓
        volvar = self.nu * self.sigma**2
        rv = -self._mgf1_correction * uu - np.log1p((-self.theta * self.nu - 0.5 * volvar * uu) * uu)
        np.exp(texp/self.nu*rv, out=rv)
        return rv

    def logp_cum4(self, texp):
        """
        Analytic cumulants of :math:`\\log(S_T/F_T)` for the Variance Gamma model.

        From the CGF :math:`K(u) = (T/\\nu)[-\\omega\\nu u - \\log(1-\\theta\\nu u - \\tfrac{1}{2}\\sigma^2\\nu u^2)]`,
        differentiating via the reciprocal-root power-sum identity:

        .. math::

            \\kappa_1 &= T\\bigl(\\theta + \\tfrac{1}{\\nu}\\log(1-\\theta\\nu-\\tfrac{1}{2}\\sigma^2\\nu)\\bigr) \\\\
            \\kappa_2 &= T(\\sigma^2 + \\nu\\theta^2) \\\\
            \\kappa_3 &= T\\nu\\,\\theta\\,(3\\sigma^2 + 2\\nu\\theta^2) \\\\
            \\kappa_4 &= 3T\\nu\\,(\\sigma^4 + 4\\nu\\theta^2\\sigma^2 + 2\\nu^2\\theta^4)

        Returns:
            ``(c1, c2, c3, c4)``
        """
        # [Verified: Claude Sonnet 4.6, 2026-05-08]
        # Let h(u) = -log(1 - θνu - σ²νu²/2).  κₙ = T/ν · h^(n)(0)  (n≥2).
        # Differentiating h at u=0  (p=θν, c=σ²ν  at u=0, denominator=1):
        #   h''(0)   = c + p²           = ν(σ² + νθ²)               → κ₂ = T(σ² + νθ²)           ✓
        #   h'''(0)  = 3cp + 2p³        = ν²θ(3σ² + 2νθ²)           → κ₃ = Tνθ(3σ² + 2νθ²)       ✓
        #   h''''(0): via Taylor -log(1-x) = x+x²/2+x³/3+…, x = θνu + σ²νu²/2:
        #     coeff of u⁴: σ⁴ν²/8 + θ²σ²ν³/2 + θ⁴ν⁴/4 = ν²/8·(σ⁴ + 4νθ²σ² + 2ν²θ⁴)
        #     → h''''(0) = 4!·ν²/8·(…) = 3ν²(σ⁴ + 4νθ²σ² + 2ν²θ⁴)
        #     → κ₄ = 3Tν(σ⁴ + 4νθ²σ² + 2ν²θ⁴)                                        ✓
        nu, sig2, th = self.nu, self.sigma**2, self.theta
        nth2 = nu * th**2
        omega = -self._mgf1_correction / nu   # = log(1 - theta*nu - sig2*nu/2) / nu
        c2 = texp * (sig2 + nth2)
        c3 = texp * nu * th * (3.0 * sig2 + 2.0 * nth2)
        c4 = 3.0 * texp * nu * (sig2**2 + 4.0 * nth2 * sig2 + 2.0 * nth2**2)
        c1 = texp * (omega + th)
        return float(c1), float(c2), float(c3), float(c4)


class NigABC(NigParams, OptABC):
    """
    Abstract base for Normal Inverse Gaussian (NIG) models — provides ``logp_mgf``
    and analytic ``logp_cum4``.

    Parameters, constraints, and the precomputed ``_mgf1_correction`` are
    defined in :class:`~pyfeng.params.NigParams`.

    References:
        - Barndorff-Nielsen OE (1997) Normal Inverse Gaussian Distributions and
          Stochastic Volatility Modelling. Scandinavian Journal of Statistics
          24:1–13. https://doi.org/10.1111/1467-9469.00045
        - Barndorff-Nielsen OE (1998) Processes of Normal Inverse Gaussian Type.
          Finance and Stochastics 2:41–68.
          https://doi.org/10.1007/s007800050032
    """

    def logp_mgf(self, uu, texp):
        """
        MGF of log price under the NIG model.

        The MGF of :math:`\\log(S_T/F_T)` at argument :math:`u` is

        .. math::

            \\exp\\!\\left(\\frac{T}{\\nu}\\left[
                \\left(\\sqrt{1 - 2\\theta\\nu - \\sigma^2\\nu} - 1\\right) u
                + 1 - \\sqrt{1 - 2\\theta\\nu u - \\sigma^2\\nu u^2}
            \\right]\\right).

        Args:
            uu: MGF argument (scalar or array).
            texp: time to expiry.

        Returns:
            MGF value(s) at ``uu``, same shape as ``uu``.
        """
        # [Verified: Claude Sonnet 4.6, 2026-05-08]
        # NIG log price: log(S_T/F_T) = ωT + θV_T + σW_{V_T},  V_T ~ IG(T, T²/ν).
        # For IG(T, T²/ν): E[exp(tV)] = exp(T/ν · (1 - √(1 - 2νt)))  (IG MGF).
        # Setting t = θu + σ²u²/2 gives the conditional log-price MGF:
        #   MGF(u) = exp(T/ν · [ων·u + 1 - √(1 - 2θνu - σ²νu²)])
        # _mgf1_correction = -ων = 1 - √(1 - 2θν - σ²ν),  so -_mgf1_correction = ων.
        # rv = ων·u + 1 - √(1 - 2θνu - σ²νu²)  ✓
        volvar = self.nu * self.sigma**2
        rv = -self._mgf1_correction * uu + 1 - np.sqrt(1 + (-2 * self.theta * self.nu - volvar * uu) * uu)
        np.exp(texp/self.nu*rv, out=rv)
        return rv

    def logp_cum4(self, texp):
        """
        Analytic cumulants of log(S_T/F) for the NIG model.

        From the CGF K(u) = (T/ν)[ωνu + 1 − √(1 − 2θνu − σ²νu²)]:

            c1 = T · (ω + θ)
            c2 = T · (σ² + νθ²)
            c3 = 3Tν · θ(σ² + νθ²)  =  3νθ · c2
            c4 = 3Tν · (σ² + νθ²)(σ² + 5νθ²)

        Returns:
            (c1, c2, c3, c4)
        """
        # [Verified: Claude Sonnet 4.6, 2026-05-08]
        # Let h(u) = 1 - √(1 - 2θνu - σ²νu²).  κₙ = T/ν · h^(n)(0)  (n≥2).
        # Differentiating h at u=0  (p=θν, c=σ²ν,  √f → 1):
        #   h''(0) = c + p² = ν(σ² + νθ²)  → κ₂ = T(σ² + νθ²)               ✓
        #   h'''(0) = 3p(c + p²) = 3ν²θ(σ² + νθ²)  → κ₃ = 3Tνθ(σ² + νθ²)   ✓
        #   h''''(0): Taylor √(1-x) = 1 - x/2 - x²/8 - x³/16 - 5x⁴/128 - …, x = 2θνu + σ²νu²:
        #     coeff of u⁴ in h(u): ν²/8·(σ⁴ + 6σ²νθ²·... → ν²/8·(σ²+νθ²)(σ²+5νθ²)
        #     → h''''(0) = 3ν²(σ²+νθ²)(σ²+5νθ²)  → κ₄ = 3Tν(σ²+νθ²)(σ²+5νθ²) ✓
        nu, sig2, th = self.nu, self.sigma**2, self.theta
        nth2 = nu * th**2
        omega = -self._mgf1_correction / nu
        c2 = texp * (sig2 + nth2)
        c3 = 3.0 * nu * th * c2
        c4 = 3.0 * nu * c2 * (sig2 + 5.0 * nth2)
        c1 = texp * (omega + th)
        return float(c1), float(c2), float(c3), float(c4)


# ──────────────────────────────────────────────────────────────────────────────
# Quadrature pricers — original (sigma, theta, nu) parametrization
# Consistent with the FFT and COS families.
# ──────────────────────────────────────────────────────────────────────────────

class VarGammaQuad(VarGammaABC):
    """
    Variance Gamma model priced by Gauss-Laguerre quadrature over the
    Gamma subordinator.

    Uses the original ``(sigma, theta, nu)`` parametrization, consistent
    with :class:`VarGammaFft` and :class:`VarGammaCos`.

    The option price is the weighted sum of BSM prices conditional on each
    quadrature point :math:`x_k` of the subordinator :math:`V_T`:

    .. math::

        C = \\sum_k w_k\\, C_\\text{BSM}(K,\\, F_k,\\, \\sigma\\sqrt{x_k/T},\\, T),

    where :math:`F_k = F_0 \\exp(\\omega T + (\\theta + \\sigma^2/2)\\,x_k)` is the
    conditional forward (martingale-normalized numerically).

    Parameters:
        n_quad: number of Gauss-Laguerre quadrature points (default 7).
    """

    n_quad = 7

    def quad(self, texp, var_rate):
        """Gauss-Laguerre quadrature for Gamma(shape=texp/var_rate, scale=var_rate) subordinator."""
        return DistGamma(shape=texp/var_rate, rate=1.0/var_rate).quad(self.n_quad)

    def price(self, strike, spot, texp, cp=1):
        # [Verified: Claude Sonnet 4.6, 2026-05-08]
        # Quadrature over subordinator V_T (Gamma for VG, IG for NIG).
        # Conditional on V_T = x_k:  log(S_T) = log(F) + ωT + θx_k + σW_{x_k}
        # → conditional BSM fwd:  F_k = F · exp(ωT + (θ + σ²/2)·x_k)
        #   (the σ²/2 is the Itô correction since E[exp(σW_{x_k})] = exp(σ²x_k/2))
        # → conditional BSM vol:  σ_k = σ√(x_k/T)  (matching variance σ²x_k = σ_k²T)  ✓
        # Normalization fwd_ratio /= ∑w·fwd_ratio enforces exact martingale numerically.
        fwd, df, _ = self._fwd_factor(spot, texp)

        var, w = self.quad(texp, self.nu)

        # Analytic omega: omega = -_mgf1_correction / nu  (same formula for VG and NIG).
        # Including omega*T in the exponent makes fwd_ratio have mean ~1 analytically;
        # the normalization line below corrects residual quadrature error.
        omega_T = -texp / self.nu * self._mgf1_correction
        fwd_ratio = np.exp(omega_T + (self.theta + 0.5 * self.sigma**2) * var)
        fwd_ratio /= np.dot(w, fwd_ratio)

        # Conditional BSM vol: sigma * sqrt(x_k / T)
        vol_bsm = self.sigma * np.sqrt(var / texp)

        strike_fwd = np.atleast_1d(strike / fwd)
        price = np.zeros_like(strike_fwd)
        for k in range(len(price)):
            price[k] = np.dot(w, fwd * Bsm.price_formula(
                strike_fwd[k], fwd_ratio, vol_bsm, texp, cp=cp))

        return df * price


class ExpNigQuad(NigABC):
    """
    NIG model priced by Choi-Du-Song (2021) quadrature over the
    Inverse Gaussian subordinator.

    Uses the original ``(sigma, theta, nu)`` parametrization, consistent
    with :class:`ExpNigFft` and :class:`NigCos`.

    Parameters:
        n_quad: number of quadrature points (default 7).

    References:
        Choi J, Du Y, Song Q (2021) Inverse Gaussian quadrature and finite
        normal-mixture approximation of the generalized hyperbolic distribution.
        Journal of Computational and Applied Mathematics 388:113302.
        https://doi.org/10.1016/j.cam.2020.113302
    """

    n_quad = 7

    def quad(self, texp, var_rate):
        """IG quadrature (Choi et al. 2021) for IG(mu=texp, lam=texp^2/var_rate) subordinator."""
        return DistInvGauss(mu=texp, lam=texp**2/var_rate).quad(self.n_quad)

    # price() is identical to VarGammaQuad — alias rather than duplicate.
    price = VarGammaQuad.price
