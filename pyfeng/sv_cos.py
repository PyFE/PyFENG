"""
European option pricing via the Fourier-Cosine (COS) method.

Core COS pricing formula (F&O Eq. 19), with x = log(S_T / F):

    V(x, t) ≈ e^{-rΔt} · F · Σ'_{k=0}^{N-1}
                  Re[ φ(u_k) · e^{-i u_k a} ] · V_k

where u_k = k π / (b-a), the prime (Σ') halves the k=0 term,
φ is the characteristic function of log(S_T/F), and V_k are
payoff cosine coefficients built from the chi/psi integrals
(F&O Eqs. 22-23).

Truncation interval [a, b] -- two methods:

  F&O (Eq. 49):   [c_1 ± L √(|c_2| + √|c_4|)]

  Junike & Pankrashkin (2022) §3 Chernoff bound:
      P(X ≥ c_1+w) ≤ inf_{s>0}  M(s)  · e^{-s(c_1+w)}
      P(X ≤ c_1-w) ≤ inf_{s>0}  M(-s) · e^{ s(c_1-w)}
  where M(s) = E[e^{sX}] is the real-argument MGF.
  The algorithm grows w until both tail bounds are below eps_junike.

Pricing formula -- two methods (controlled by ``pricing_formula``):

  'fang-oosterlee' (default): separate call/put payoff coefficients
      (F&O Eqs. 20-25); call coefficients degrade when z = log(K/F)
      approaches the truncation boundary b.

  'lefloch' (Le Floc'h 2020, Eq. 10): always compute put via:
      V_k^Put(z) = (2/(b-a)) · [(K/F)·ψ_k(a,z) - χ_k(a,z)],  z = log(K/F)
      Call = Put + df·(F-K)   (put-call parity; exact by construction)
      Boundary: z < a → P = 0;  z > b → P = df·(K-F)⁺
  This gives uniform accuracy across strikes because the put integral
  over [a, z] is well-conditioned for all z in [a, b].

References:
    Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions.
    SIAM Journal on Scientific Computing 31(2):826-848.
    https://doi.org/10.1137/080718061

    Junike G, Pankrashkin K (2022) Precise option pricing by the COS method --
    how to choose the truncation range. Applied Mathematics and Computation
    421:126935. https://doi.org/10.1016/j.amc.2022.126935

    Le Floc'h F (2020) More Robust Pricing of European Options Based on
    Fourier Cosine Series Expansions. arXiv:2005.13248.
    https://arxiv.org/abs/2005.13248
"""

import abc
import numpy as np
from . import opt_abc as opt
from .sv_fft import VarGammaFft, CgmyFft, HestonFft


__all__ = ['CosABC', 'BsmCos', 'VarGammaCos', 'CgmyCos', 'HestonCos']


class CosABC(opt.OptABC, abc.ABC):
    """
    Abstract base class for European vanilla pricing by the COS method.

    Subclasses implement ``mgf_logprice(uu, texp)``, the moment generating
    function of log(S_T/F), where F is the forward price supplied by
    ``OptABC._fwd_factor``.

    Pricing identity (F&O Eq. 19), evaluated in z = log(S_T/F):

        V ≈ df · F · (w_payoff @ A)

    where A_k = Re[φ(u_k) e^{-i u_k a}] (halved at k=0) are density
    coefficients, and w_payoff are the strike-dependent V_k coefficients
    from F&O Eqs. 20-25.

    Attributes:
        n_cos (int): Number of Fourier-cosine terms N (default 128).
        L (float): Truncation half-width multiplier (default 12.0).
        truncation_method (str): 'fang-oosterlee' (default) or 'junike'.
        eps_junike (float): Tail probability target for Junike truncation (default 1e-8).
        pricing_formula (str): 'fang-oosterlee' (default) or 'lefloch'.
            'lefloch' uses the put-first + PCP formula from Le Floc'h (2020)
            for more uniform accuracy across strikes.
    """

    n_cos: int = 128
    L: float = 12.0
    truncation_method: str = 'fang-oosterlee'
    eps_junike: float = 1e-8
    pricing_formula: str = 'fang-oosterlee'

    @abc.abstractmethod
    def mgf_logprice(self, uu, texp):
        """
        Moment generating function of log(S_T / F).

        Args:
            uu: scalar or array, real or complex.
            texp: time to expiry.

        Returns:
            MGF values with the same shape as ``uu``.
        """
        raise NotImplementedError

    def charfunc_logprice(self, u, texp):
        """Characteristic function φ(u) = MGF(i·u)."""
        return self.mgf_logprice(1j * u, texp)

    def _cumulants(self, texp):
        """
        First four cumulants of log(S_T/F) by finite differences on log MGF.

        Central-difference formulas (h = eps):
            c1 = [K(h) - K(-h)] / 2h
            c2 = [K(h) + K(-h) - 2K(0)] / h²
            c4 = [K(2h) - 4K(h) + 6K(0) - 4K(-h) + K(-2h)] / h⁴
        where K(u) = log MGF(u) is the cumulant generating function.

        Returns:
            (c1, c2, 0.0, c4)
        """
        eps = 1e-3
        # Wrap in array: VarGammaFft/CgmyFft use np.exp(..., out=rv) which
        # requires rv to be an ndarray, not a scalar.
        lm = lambda v: float(np.log(self.mgf_logprice(np.atleast_1d(np.float64(v)), texp)).real)
        lm0 = lm(0.0)
        lmp1, lmm1 = lm(eps), lm(-eps)
        lmp2, lmm2 = lm(2 * eps), lm(-2 * eps)
        c1 = (lmp1 - lmm1) / (2 * eps)
        c2 = (lmp1 + lmm1 - 2 * lm0) / eps**2
        c4 = (lmp2 - 4 * lmp1 + 6 * lm0 - 4 * lmm1 + lmm2) / eps**4
        return c1, c2, 0.0, c4

    def _junike_half_width(self, texp):
        """
        Chernoff-bound adaptive half-width (Junike & Pankrashkin 2022, §3).

        The algorithm finds the smallest w such that both tail bounds are
        below eps_junike:

            upper = inf_{s>0} M(s)  · e^{-s(c1+w)}   (right tail bound)
            lower = inf_{s>0} M(-s) · e^{ s(c1-w)}   (left  tail bound)

        Starting from the F&O Eq. 49 initial guess
            w = L · √(|c2| + √|c4|)
        L is grown by 1.25× each iteration (capped at 30) until
        max(0, upper + lower) ≤ eps_junike.

        Returns:
            w (float): half-width such that [c1-w, c1+w] captures the density.
        """
        c1, c2, _, c4 = self._cumulants(texp)
        L = 12.0
        w = 0.0
        for _ in range(15):
            w = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
            s_grid = np.logspace(-2, 2, 40)

            # Upper tail: inf_{s>0} exp(log M(s) - s*(c1+w))
            mgf_pos = self.mgf_logprice(s_grid, texp).real
            ok = np.isfinite(mgf_pos) & (mgf_pos > 0)
            if ok.any():
                upper = float(np.exp(np.min(np.log(mgf_pos[ok]) - s_grid[ok] * (c1 + w))))
            else:
                upper = 1.0

            # Lower tail: inf_{s>0} exp(log M(-s) + s*(c1-w))
            mgf_neg = self.mgf_logprice(-s_grid, texp).real
            ok = np.isfinite(mgf_neg) & (mgf_neg > 0)
            if ok.any():
                lower = float(np.exp(np.min(np.log(mgf_neg[ok]) + s_grid[ok] * (c1 - w))))
            else:
                lower = 1.0

            if max(0.0, upper + lower) <= self.eps_junike:
                break
            L = min(L * 1.25, 30.0)
        return w

    def _truncation_range(self, texp):
        """
        COS integration range [a, b] for log(S_T/F).

        F&O mode (Eq. 49):
            a = c1 - L · √(|c2| + √|c4|)
            b = c1 + L · √(|c2| + √|c4|)

        Junike mode (Junike & Pankrashkin 2022, §3):
            a = c1 - w,  b = c1 + w
        where w is the Chernoff-bound adaptive half-width from
        ``_junike_half_width``.
        """
        if self.truncation_method == 'fang-oosterlee':
            c1, c2, _, c4 = self._cumulants(texp)
            half = self.L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
            return c1 - half, c1 + half
        elif self.truncation_method == 'junike':
            c1, _, _, _ = self._cumulants(texp)
            half = self._junike_half_width(texp)
            return c1 - half, c1 + half
        else:
            raise ValueError(
                f"truncation_method must be 'fang-oosterlee' or 'junike', "
                f"got {self.truncation_method!r}"
            )

    def _integration_range(self, strike, spot, texp):
        """COS interval used by ``price``. Strike-independent by default."""
        return self._truncation_range(texp)

    @staticmethod
    def _chi(k, u, a, c, d):
        """
        F&O Eq. 22 -- integral of e^x · cos(u·(x-a)) from c to d.

            χ_k(c,d) = [cos(u(d-a))·eᵈ - cos(u(c-a))·eᶜ
                         + u·(sin(u(d-a))·eᵈ - sin(u(c-a))·eᶜ)] / (1 + u²)

        where u = k·π/(b-a).  Broadcasting: u shape (1,N), c/d shape (M,1).
        """
        exp_d, exp_c = np.exp(d), np.exp(c)
        cos_d = np.cos(u * (d - a))
        cos_c = np.cos(u * (c - a))
        sin_d = np.sin(u * (d - a))
        sin_c = np.sin(u * (c - a))
        num = cos_d * exp_d - cos_c * exp_c
        num += u * (sin_d * exp_d - sin_c * exp_c)
        return num / (1.0 + u**2)

    @staticmethod
    def _psi(k, u, a, c, d):
        """
        F&O Eq. 23 -- integral of cos(u·(x-a)) from c to d.

            ψ_k(c,d) = d - c                                  (k = 0)
                      = [sin(u(d-a)) - sin(u(c-a))] / u       (k ≥ 1)

        where u = k·π/(b-a).
        """
        safe_u = np.where(k == 0, 1.0, u)
        return np.where(
            k == 0,
            d - c,
            (np.sin(u * (d - a)) - np.sin(u * (c - a))) / safe_u,
        )

    def _cos_grid(self, a, b):
        """COS mode indices k and frequencies u_k = k·π/(b-a)."""
        k_arr = np.arange(int(self.n_cos))
        u_arr = k_arr * np.pi / (b - a)
        return k_arr, u_arr

    def _density_coefficients(self, u_arr, a, texp):
        """
        Density-side COS coefficients A_k (F&O Eqs. 8-9, 19).

            A_k = Re[ φ(u_k) · e^{-i u_k a} ]

        The k=0 term is halved for the prime summation (Σ').
        """
        coeff = self.charfunc_logprice(u_arr, texp) * np.exp(-1j * u_arr * a)
        coeff[0] *= 0.5
        return coeff.real

    def _vanilla_payoff_coefficients(self, strike, fwd, a, b, cp):
        """
        Vanilla call/put payoff coefficients V_k (F&O Eqs. 20-25).

        In F&O's y = log(S_T/K) the payoff boundary is y = 0.  In PyFENG's
        z = log(S_T/F) the boundary shifts to z* = log(K/F), and the scale
        factor becomes df·F (applied in ``price``).

        Call (cp > 0):
            V_k = (2/(b-a)) · [χ_k(z*, b) - (K/F)·ψ_k(z*, b)]

        Put (cp ≤ 0):
            V_k = (2/(b-a)) · [(K/F)·ψ_k(a, z*) - χ_k(a, z*)]
        """
        kk = np.atleast_1d(np.asarray(strike / fwd, dtype=float))
        cp_a = np.broadcast_to(
            np.atleast_1d(np.asarray(cp, dtype=float)), kk.shape
        ).copy()

        k_arr, u_arr = self._cos_grid(a, b)
        log_kk = np.clip(np.log(kk), a, b)[:, None]   # z* = log(K/F), clipped
        u = u_arr[None, :]
        k = k_arr[None, :]
        kk_c = kk[:, None]

        w_call = (2.0 / (b - a)) * (
            self._chi(k, u, a, log_kk, b)
            - kk_c * self._psi(k, u, a, log_kk, b)
        )
        w_put = (2.0 / (b - a)) * (
            kk_c * self._psi(k, u, a, a, log_kk)
            - self._chi(k, u, a, a, log_kk)
        )
        return np.where(cp_a[:, None] > 0, w_call, w_put)

    def _price_lefloch(self, strike, spot, texp, cp=1):
        """
        Le Floc'h (2020) improved put-first COS pricing (Eqs. 7-10).

        Always computes the put using forward-centered payoff coefficients:

            V_k^Put(z) = (2/(b-a)) · [(K/F)·ψ_k(a,z) - χ_k(a,z)]

        where z = log(K/F).  The call is derived via put-call parity:

            C = P + df·(F - K)

        This is exact by construction.

        Boundary conditions (Le Floc'h §4):
            z < a  →  P = 0                  (deep OTM put; density beyond b)
            z > b  →  P = df·(K - F)⁺        (beyond truncation: intrinsic)

        The improvement over F&O: for deep OTM calls (z close to b), the
        F&O call integral χ(z,b) - (K/F)·ψ(z,b) is nearly empty and
        numerically unstable.  Le Floc'h's put integral over [a, z] remains
        large and stable; the PCP correction distributes the payoff error
        uniformly across all strikes (Le Floc'h 2020, Fig. 2).

        Uses ``_truncation_range(texp)`` (global interval) rather than
        ``_integration_range`` (which may be per-strike for HestonCos F&O mode).
        """
        fwd, df, _ = self._fwd_factor(spot, texp)
        a, b = self._truncation_range(texp)
        k_arr, u_arr = self._cos_grid(a, b)
        cf_re = self._density_coefficients(u_arr, a, texp)

        kk_arr = np.atleast_1d(np.asarray(strike, dtype=float))
        fwd_f = float(np.asarray(fwd))
        df_f = float(np.asarray(df))
        log_kk = np.log(kk_arr / fwd_f)   # z = log(K/F) for each strike

        below_a = log_kk < a   # put = 0
        above_b = log_kk > b   # put = intrinsic

        z_col = np.clip(log_kk, a, b)[:, None]   # clipped z, shape (M, 1)
        kk_c = (kk_arr / fwd_f)[:, None]          # K/F, shape (M, 1)
        k = k_arr[None, :]                         # (1, N)
        u = u_arr[None, :]                         # (1, N)

        # Le Floc'h Eq. 7: V_k^Put = (2/(b-a)) * ((K/F)*psi(a,z) - chi(a,z))
        w_put = (2.0 / (b - a)) * (
            kk_c * self._psi(k, u, a, a, z_col)
            - self._chi(k, u, a, a, z_col)
        )
        put_cos = df_f * fwd_f * (w_put @ cf_re)   # (M,) COS put prices

        # Apply boundary conditions
        put_price = np.where(
            below_a,
            0.0,
            np.where(above_b, df_f * np.maximum(kk_arr - fwd_f, 0.0), put_cos),
        )

        # Call via PCP: C = P + df*(F - K)  (Le Floc'h Eq. 10 consequence)
        call_price = put_price + df_f * (fwd_f - kk_arr)

        cp_arr = np.broadcast_to(
            np.atleast_1d(np.asarray(cp, dtype=float)), kk_arr.shape
        )
        return np.where(cp_arr > 0, call_price, put_price)

    def price(self, strike, spot, texp, cp=1):
        """
        European call/put price via the COS method.

        F&O mode (``pricing_formula='fang-oosterlee'``, default):

            V ≈ df · F · (V_k @ A_k)   (F&O Eq. 19)

        where A_k = Re[φ(u_k) e^{-i u_k a}] and V_k are per-cp payoff
        coefficients (F&O Eqs. 20-25).

        Le Floc'h mode (``pricing_formula='lefloch'``):

            P = df · F · (V_k^Put @ A_k)   (Le Floc'h Eq. 10)
            C = P + df·(F - K)

        with boundary conditions and per-strike global truncation range.

        Raises:
            ValueError: if ``pricing_formula`` is not recognised.
        """
        scalar_out = np.isscalar(strike) and np.isscalar(cp)

        if self.pricing_formula == 'lefloch':
            result = np.atleast_1d(self._price_lefloch(strike, spot, texp, cp))
            if scalar_out:
                return float(result[0])
            return result.reshape(np.broadcast_shapes(np.shape(strike), np.shape(cp)))
        elif self.pricing_formula != 'fang-oosterlee':
            raise ValueError(
                f"pricing_formula must be 'fang-oosterlee' or 'lefloch', "
                f"got {self.pricing_formula!r}"
            )

        fwd, df, _ = self._fwd_factor(spot, texp)
        a, b = self._integration_range(strike, spot, texp)
        _, u_arr = self._cos_grid(a, b)
        cf_re = self._density_coefficients(u_arr, a, texp)
        w_payoff = self._vanilla_payoff_coefficients(strike, fwd, a, b, cp)

        price_arr = df * fwd * (w_payoff @ cf_re)

        if scalar_out:
            return float(price_arr[0])
        return price_arr.reshape(
            np.broadcast_shapes(np.shape(strike), np.shape(cp))
        )


class BsmCos(CosABC):
    """
    Black-Scholes-Merton European option pricing via the COS method.

    MGF of log(S_T/F) under BSM:
        M(u) = exp(-½ σ² T · u(1-u))

    Analytic cumulants (c4 = 0, so F&O Eq. 49 gives the exact range):
        c1 = -½ σ² T,   c2 = σ² T

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmCos(sigma=0.2, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
    """

    n_cos: int = 64

    def mgf_logprice(self, uu, texp):
        """BSM MGF: exp(-½ σ² T · u(1-u))."""
        return np.exp(-0.5 * self.sigma**2 * texp * uu * (1.0 - uu))

    def _cumulants(self, texp):
        """Analytic BSM cumulants of log(S_T/F): c1=-½σ²T, c2=σ²T, c3=c4=0."""
        s2t = self.sigma**2 * texp
        return -0.5 * s2t, s2t, 0.0, 0.0


class VarGammaCos(VarGammaFft, CosABC):
    """
    Variance Gamma (VG) European option pricing via the COS method.

    The VG log price is x = θ·G_T + σ·W_{G_T} + ω·T where G_T ~ Gamma(T/ν, ν)
    and ω = log(1 - θν - ½σ²ν)/ν is the martingale correction.

    Inherits the MGF from ``VarGammaFft``; overrides ``_cumulants`` with the
    analytic formulas from F&O 2008 Table 11 so the truncation range is exact.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.VarGammaCos(sigma=0.12, theta=-0.14, vov=0.2,
        ...                    intr=0.1, divr=0.0)
        >>> m.price(np.array([90.0, 100.0, 110.0]), 100.0, 1.0)
    """

    n_cos: int = 256

    # MRO routes price() through FftABC; rebind to the COS path.
    price = CosABC.price

    def _cumulants(self, texp):
        """
        Analytic VG cumulants of log(S_T/F) -- F&O 2008 Table 11.

            ω  = log(1 - θν - ½σ²ν) / ν
            c1 = T · (ω + θ)
            c2 = (σ² + ν θ²) · T
            c4 = 3T · (σ⁴ν + 2θ⁴ν³ + 4σ²θ²ν²)

        Returns:
            (c1, c2, 0.0, c4)
        """
        T = float(texp)
        nu, sig, th = self.vov, self.sigma, self.theta
        omega = np.log(1.0 - th * nu - 0.5 * sig**2 * nu) / nu
        c1 = T * (omega + th)
        c2 = (sig**2 + nu * th**2) * T
        c4 = 3.0 * T * (sig**4 * nu + 2.0 * th**4 * nu**3 + 4.0 * sig**2 * th**2 * nu**2)
        return float(c1), float(c2), 0.0, float(c4)


class CgmyCos(CgmyFft, CosABC):
    """
    CGMY infinite-activity Lévy European option pricing via the COS method.

    The CGMY process (Carr, Geman, Madan, Yor 2002) has per-unit-time CGF:
        κ(u) = C·Γ(-Y)·[(M-u)^Y - M^Y + (G+u)^Y - G^Y]

    Inherits the MGF from ``CgmyFft``; overrides the truncation range with
    the Y-dependent heuristic from F&O 2008 §5.4 because CGMY moments
    diverge as Y → 2, making the cumulant-based rule unreliable at high Y.

    Note:
        The default L=12 may need reducing to L=10 to match
        the source ``cos_pricing.CgmyModel`` exactly.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.CgmyCos(C=1.0, G=5.0, M=10.0, Y=0.5)
        >>> m.price(np.arange(80, 121, 10), 100.0, 1.0)
    """

    n_cos: int = 256

    # MRO routes price() through FftABC; rebind to the COS path.
    price = CosABC.price

    def _truncation_range(self, texp):
        """
        COS truncation range -- F&O 2008 §5.4 Y-dependent heuristic.

        F&O mode:
            [a, b] = [-L·Y, L·Y]         (generic Y)
            [a, b] = [-100, 20]           (Y ≈ 1.98, F&O special case)

        Junike mode (Junike & Pankrashkin 2022, §3):
            [a, b] = [c1 - w, c1 + w]
        where w is the Chernoff-bound adaptive half-width, using numerical
        cumulants from ``CosABC._cumulants``.
        """
        if self.truncation_method == 'junike':
            c1, _, _, _ = CosABC._cumulants(self, texp)
            half = self._junike_half_width(texp)
            return c1 - half, c1 + half
        if np.isclose(self.Y, 1.98):
            return -100.0, 20.0
        return -self.L * self.Y, self.L * self.Y


class HestonCos(HestonFft, CosABC):
    """
    Heston (1993) stochastic-volatility European option pricing via the COS method.

    Inherits the Lord & Kahl (2010) branch-cut-safe MGF from ``HestonFft``.

    F&O 2008 §5.2 Heston-tailored truncation: the integration interval is
    centered per-strike on x + c1 where x = log(F/K):

        [a_K, b_K] = [x + c1 - L·σ_h,  x + c1 + L·σ_h]
        σ_h = √(θ̄ + V_0 · ξ)    (F&O §5.2, long-run + initial variance scaled)

    where θ̄ is the long-run variance, V_0 the initial variance, and ξ the
    vol-of-vol.  Because width = 2·L·σ_h is strike-independent, the CF
    φ(u_k) is computed once and phase-shifted per strike:

        A_k(K) = Re[ φ(u_k) · e^{-i u_k a_K} ]

    The L multiplier adapts with maturity as max(10, 3·T + 2) when L is None.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.HestonCos(0.0175, vov=0.5751, mr=1.5768,
        ...                  theta=0.0398, rho=-0.5711)
        >>> m.price(np.array([90.0, 100.0, 110.0]), 100.0, 1.0)
    """

    n_cos: int = 160
    L = None

    def __init__(self, sigma, vov=0.01, rho=0.0, mr=0.01, theta=None,
                 intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: initial variance V_0 (must be > 0).
            vov: vol-of-vol ξ (must be > 0).
            rho: spot-variance correlation, in (-1, 1).
            mr: mean-reversion speed κ (must be > 0).
            theta: long-run variance θ̄ (must be > 0; defaults to sigma).
            intr: interest rate.
            divr: dividend yield.
            is_fwd: if True, treat spot as forward price.

        Raises:
            ValueError: on invalid parameter values.
        """
        if sigma <= 0.0:
            raise ValueError(f"sigma (initial variance) must be > 0, got {sigma}")
        if vov <= 0.0:
            raise ValueError(f"vov (vol-of-vol) must be > 0, got {vov}")
        if mr <= 0.0:
            raise ValueError(f"mr (mean-reversion) must be > 0, got {mr}")
        if theta is not None and theta <= 0.0:
            raise ValueError(f"theta (long-run variance) must be > 0, got {theta}")
        if not (-1.0 < rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {rho}")

        super().__init__(sigma=sigma, vov=vov, rho=rho, mr=mr, theta=theta,
                         intr=intr, divr=divr, is_fwd=is_fwd)

        # SvABC sets self.theta = sigma when theta is None; re-check after super.
        if self.theta <= 0.0:
            raise ValueError(f"theta resolved to <= 0 ({self.theta})")

    @staticmethod
    def _default_L(texp):
        """Tau-adaptive L = max(10, 3·T + 2) (F&O §5.2 recommendation)."""
        return max(10.0, 3.0 * float(texp) + 2.0)

    def _resolve_L(self, texp):
        """Return self.L if explicitly set, otherwise _default_L(texp)."""
        return self._default_L(texp) if self.L is None else float(self.L)

    def _sigma_h(self):
        """
        F&O 2008 §5.2 Heston truncation scale.

            σ_h = √(θ̄ + V_0 · ξ)

        where θ̄ = self.theta (long-run variance), V_0 = self.sigma
        (initial variance), ξ = self.vov (vol-of-vol).
        """
        return float(np.sqrt(self.theta + self.sigma * self.vov))

    def _c1_logF(self, texp):
        """
        First cumulant of log(S_T/F): c1 = -½ T · E[⟨V⟩_T / T].

        Computed analytically via ``avgvar_mv`` (mean of integrated variance).
        """
        T = float(texp)
        return -0.5 * T * self.avgvar_mv(T)[0]

    def _cumulants(self, texp, eps=1e-4):
        """
        Cumulants (c1, c2, 0, 0) of log(S_T/F).

        c1 is analytic via ``_c1_logF``; c2 is by finite differences on
        log MGF (F&O §5 sets c4 = 0 for Heston).
        """
        K = lambda uu: float(np.log(self.mgf_logprice(uu, texp)).real)
        c2 = (K(eps) + K(-eps) - 2.0 * K(0.0)) / (eps * eps)
        return float(self._c1_logF(texp)), float(c2), 0.0, 0.0

    def _truncation_range(self, texp):
        """
        Global COS truncation range [a, b] for log(S_T/F).

        Used by ``_price_lefloch`` and Junike mode.  L=None is resolved via
        ``_resolve_L`` so the adaptive formula max(10, 3·T+2) applies.

        F&O mode (Eq. 49) with adaptive L:
            half = L · √(|c2| + √|c4|)

        Junike mode:
            [a, b] = [c1 ± w]
        """
        if self.truncation_method == 'fang-oosterlee':
            c1, c2, _, c4 = self._cumulants(texp)
            half = self._resolve_L(texp) * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
            return float(c1 - half), float(c1 + half)
        elif self.truncation_method == 'junike':
            c1 = self._c1_logF(texp)
            half = self._junike_half_width(texp)
            return float(c1 - half), float(c1 + half)
        else:
            raise ValueError(
                f"truncation_method must be 'fang-oosterlee' or 'junike', "
                f"got {self.truncation_method!r}"
            )

    def _integration_range(self, strike, spot, texp):
        """
        COS interval -- per-strike (F&O §5.2) or global (Junike mode).

        F&O mode:
            [a_K, b_K] = [log(F/K) + c1 ± L·σ_h]

        Junike mode (Junike & Pankrashkin 2022, §3):
            [a, b] = [c1 ± w]   (single global interval for all strikes)
        """
        if self.truncation_method == 'junike':
            c1 = self._c1_logF(texp)
            half = self._junike_half_width(texp)
            return float(c1 - half), float(c1 + half)
        fwd, _, _ = self._fwd_factor(spot, texp)
        half = self._resolve_L(texp) * self._sigma_h()
        c1 = self._c1_logF(texp)
        x = np.log(np.asarray(fwd, dtype=float) / np.asarray(strike, dtype=float))
        return x + c1 - half, x + c1 + half

    def price(self, strike, spot, texp, cp=1):
        """
        European call/put price via the Heston COS method (F&O §5.2).

        Junike mode: scalar [a, b] -- delegates to ``CosABC.price``.

        F&O mode: per-strike [a_K, b_K] with constant width = 2·L·σ_h.
        Because u_k = k·π/width is strike-independent, φ(u_k) is computed
        once and phase-shifted per strike (F&O §5.2):

            A_k(K) = Re[ φ(u_k) · e^{-i u_k a_K} ]

        This avoids calling the CF once per strike.
        """
        if texp <= 0.0:
            raise ValueError(f"texp must be > 0, got {texp}")
        if int(self.n_cos) < 1:
            raise ValueError(f"n_cos must be >= 1, got {self.n_cos}")

        if self.truncation_method == 'junike' or self.pricing_formula == 'lefloch':
            return CosABC.price(self, strike, spot, texp, cp=cp)

        fwd, df, _ = self._fwd_factor(spot, texp)
        scalar_out = np.isscalar(strike) and np.isscalar(cp)

        a_arr, b_arr = self._integration_range(strike, spot, texp)
        half = self._resolve_L(texp) * self._sigma_h()
        width = 2.0 * half
        k_arr = np.arange(int(self.n_cos))
        u_arr = k_arr * np.pi / width           # (N,) -- constant across strikes

        phi   = self.charfunc_logprice(u_arr, texp)                    # (N,) once
        a_col = np.atleast_1d(np.asarray(a_arr, dtype=float))[:, None] # (M, 1)
        phase = np.exp(-1j * u_arr[None, :] * a_col)                   # (M, N)
        cf_mat = phi[None, :] * phase                                   # (M, N)
        cf_mat[:, 0] *= 0.5                                             # prime-sum
        cf_re = cf_mat.real                                             # (M, N)

        kk     = np.atleast_1d(np.asarray(strike, dtype=float)) / float(np.asarray(fwd))
        cp_arr = np.broadcast_to(np.atleast_1d(np.asarray(cp, dtype=float)), kk.shape)
        log_kk = np.clip(np.log(kk), a_arr, b_arr)[:, None]            # (M, 1)
        u      = u_arr[None, :]                                         # (1, N)
        k      = k_arr[None, :]                                         # (1, N)
        kk_c   = kk[:, None]                                            # (M, 1)
        b_col  = np.atleast_1d(np.asarray(b_arr, dtype=float))[:, None] # (M, 1)

        # F&O Eqs. 24-25 in z = log(S_T/F) coordinates
        W_call = (2.0 / width) * (
            CosABC._chi(k, u, a_col, log_kk, b_col)
            - kk_c * CosABC._psi(k, u, a_col, log_kk, b_col)
        )
        W_put = (2.0 / width) * (
            kk_c * CosABC._psi(k, u, a_col, a_col, log_kk)
            - CosABC._chi(k, u, a_col, a_col, log_kk)
        )
        W = np.where(cp_arr[:, None] > 0, W_call, W_put)

        price_arr = float(df) * float(np.asarray(fwd)) * (W * cf_re).sum(axis=1)

        if scalar_out:
            return float(price_arr[0])
        return price_arr.reshape(np.broadcast_shapes(np.shape(strike), np.shape(cp)))
