"""
Generic 2D Douglas ADI finite-difference mixin for stochastic-volatility models.

Works in normalized forward space (F₀ = 1, r = q = 0 internally).
Discounting and forward-scaling are handled by the outer ``price()`` method.

The 2D PDE for V(X, v, τ), τ = T − t has the form::

    dV/dτ = (F₀ + F₁ + F₂) V

    F₀: γ(X,v,τ)  · ∂²V/∂X∂v              (explicit; mixed derivative)
    F₁: aS(X,v,τ) · ∂²V/∂X² + bS(X,v,τ) · ∂V/∂X   (S-direction implicit)
    F₂: aV(X,v,τ) · ∂²V/∂v² + bV(X,v,τ) · ∂V/∂v   (v-direction implicit)

One Douglas step τₙ → τₙ₊₁ (θ = ½ gives Crank–Nicolson-like accuracy)::

    Y₀ = Vⁿ + dt (F₀ + F₁ + F₂) Vⁿ                  (explicit predictor)
    (I − θ dt F₁) Y₁ = Y₀ − θ dt F₁ Vⁿ              (S-implicit corrector)
    (I − θ dt F₂) Vⁿ⁺¹ = Y₁ − θ dt F₂ Vⁿ            (v-implicit corrector)

To use for a new SV model, subclass ``SvFinDiffMixin`` and implement
four abstract methods: ``_get_v0``, ``_grid_bounds``, ``_coefficients``,
``_apply_bc``.

References:
    von Sydow et al. (2019) BENCHOP-SLV, Int. J. Comput. Math.
    K. in 't Hout & S. Foulon (2010), ADI FD schemes for Heston.
"""

import abc
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Module-level utilities
# ─────────────────────────────────────────────────────────────────────────────

def _thomas_batch(sub, mid, sup, rhs):
    """
    Vectorized Thomas algorithm for batched tridiagonal systems.

    Coefficients ``sub``, ``mid``, ``sup`` are 1-D of length *n* (broadcast
    across every column of *rhs*) or 2-D of shape *(n, batch)*.  *rhs* has
    shape *(n, batch)*; the solution has the same shape.

    Equation at row i:  sub[i]·x[i−1] + mid[i]·x[i] + sup[i]·x[i+1] = rhs[i].
    sub[0] and sup[−1] are unused.
    """
    n = mid.shape[0]
    cp = np.empty_like(np.asarray(mid, dtype=float))
    dp = np.empty_like(rhs, dtype=float)

    cp[0] = sup[0] / mid[0]
    dp[0] = rhs[0] / mid[0]
    for i in range(1, n):
        denom = mid[i] - sub[i] * cp[i - 1]
        cp[i] = sup[i] / denom
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / denom

    x = np.empty_like(rhs, dtype=float)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _bilinear(V, S, v, S0, v0):
    """Bilinear interpolation of grid function V at the point (S0, v0)."""
    n_S, n_v = len(S), len(v)

    i = int(np.searchsorted(S, S0))
    i = max(1, min(i, n_S - 1))

    j = int(np.searchsorted(v, v0))
    j = max(1, min(j, n_v - 1))

    S_lo, S_hi = S[i - 1], S[i]
    v_lo, v_hi = v[j - 1], v[j]
    wS = float(np.clip((S0 - S_lo) / (S_hi - S_lo), 0.0, 1.0))
    wv = float(np.clip((v0 - v_lo) / (v_hi - v_lo), 0.0, 1.0))

    return (
        (1 - wS) * (1 - wv) * V[i - 1, j - 1]
        + wS     * (1 - wv) * V[i,     j - 1]
        + (1 - wS) * wv     * V[i - 1, j    ]
        + wS       * wv     * V[i,     j    ]
    )


def _standard_bc(V, kk, X, cp):
    """
    Standard European call/put boundary conditions in normalized space (r=q=0).

    *   S = 0 (X = X_min):  call → 0,   put → kk
    *   S = S_max (X = X_max): call → X_max − kk,  put → 0
    *   v = 0:  intrinsic value (no volatility → no time value)
    *   v = v_max: Neumann  dV/dv = 0  (V[·, L] = V[·, L−1])
    """
    M = len(X) - 1
    if cp > 0:   # call
        V[0,  :] = 0.0
        V[M,  :] = max(X[M] - kk, 0.0)
        V[:,  0] = np.maximum(X - kk, 0.0)
    else:        # put
        V[0,  :] = kk
        V[M,  :] = 0.0
        V[:,  0] = np.maximum(kk - X, 0.0)
    V[:, -1] = V[:, -2]   # Neumann at v_max
    return V


# ─────────────────────────────────────────────────────────────────────────────
# Generic ADI mixin
# ─────────────────────────────────────────────────────────────────────────────

class SvFinDiffMixin(abc.ABC):
    """
    Generic Douglas-scheme 2D ADI mixin for stochastic-volatility models.

    Subclass and implement four abstract methods to plug in any model::

        class MyModelFinDiff(SvFinDiffMixin, MyModelABC):
            def _get_v0(self, fwd, texp):   return ...
            def _grid_bounds(self, v0, kk, texp): return X_min, X_max, v_max
            def _coefficients(self, X, v, tau):   return aS, bS, aV, bV, gamma
            def _apply_bc(self, V, kk, X, cp):    return _standard_bc(V, kk, X, cp)

    The engine works in normalized forward space (X = F/F₀, so X₀ = 1, r=q=0).
    ``price()`` handles forward/discount scaling externally.
    """

    # Default numerical parameters — override via set_num_params()
    n_asset:   int   = 80
    n_vol:     int   = 40
    n_time:    int   = 80
    adi_theta: float = 0.5

    def set_num_params(self, n_asset=80, n_vol=40, n_time=80, adi_theta=0.5):
        """
        Configure the ADI grid (consistent with ``set_num_params`` in *_mc.py).

        Args:
            n_asset:   asset price grid intervals (M)
            n_vol:     vol/variance grid intervals (L)
            n_time:    time steps (N)
            adi_theta: Douglas weight; 0.5 → Crank–Nicolson accuracy
        """
        self.n_asset   = int(n_asset)
        self.n_vol     = int(n_vol)
        self.n_time    = int(n_time)
        self.adi_theta = float(adi_theta)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def _get_v0(self, fwd, texp):
        """
        Initial vol/variance in the normalized PDE.

        SABR: ``alpha = sigma / fwd^(1−beta)`` via ``SabrABC._variables()``.
        Heston: ``self.sigma`` (initial variance; scale-free).
        """

    @abc.abstractmethod
    def _grid_bounds(self, v0, kk, texp):
        """
        Normalized grid boundaries.

        Args:
            v0:   initial vol/variance (from ``_get_v0``)
            kk:   normalized strike K/F₀
            texp: time to expiry

        Returns:
            (X_min, X_max, v_max)
        """

    @abc.abstractmethod
    def _coefficients(self, X, v, tau):
        """
        PDE operator coefficients on the full (M+1) × (L+1) grid.

        Args:
            X:   asset grid, shape (M+1,)
            v:   vol/variance grid, shape (L+1,)
            tau: time-to-maturity; pass to callable parameters if time-dependent

        Returns:
            aS, bS, aV, bV, gamma — each broadcastable to (M+1, L+1).

            aS: (1/2) · diffusion²  in S-direction
            bS: drift               in S-direction  (0 if F is a martingale)
            aV: (1/2) · diffusion²  in v-direction
            bV: drift               in v-direction  (0 for SABR; κ(θ−v) for Heston)
            gamma: ρ · σ_S · σ_v   (mixed-derivative coefficient)
        """

    @abc.abstractmethod
    def _apply_bc(self, V, kk, X, cp):
        """
        Boundary conditions in normalized space.

        No discounting (r = q = 0 internally).  For standard European
        calls/puts return ``_standard_bc(V, kk, X, cp)``.
        """

    # ── Public pricing entry point ────────────────────────────────────────────

    def price(self, strike, spot, texp, cp=1):
        """
        Price European option(s) via Douglas ADI in normalized forward space.

        Loops over strikes (each requires a separate ADI solve because the
        terminal payoff and boundary conditions depend on K).

        Args:
            strike: scalar or ndarray of strike prices
            spot:   spot (or forward when ``is_fwd=True``) price
            texp:   time to expiry
            cp:     1 = call, −1 = put

        Returns:
            Option price(s); same shape as ``strike``.
        """
        fwd, df, _ = self._fwd_df_divf(spot, texp)
        v0  = self._get_v0(fwd, texp)
        kk  = np.asarray(strike, dtype=float) / fwd   # normalized strike(s)
        scalar = np.ndim(strike) == 0
        px  = np.array([self._solve_one(k, v0, texp, cp)
                        for k in np.atleast_1d(kk).ravel()])
        result = df * fwd * px
        return float(result[0]) if scalar else result.reshape(np.shape(strike))

    # ── Core ADI solver ───────────────────────────────────────────────────────

    def _pre_step(self, V, X, v, dX, dv, dt, tau, kk, cp):
        """
        Hook called before each Douglas step.

        Override in subclasses to apply model-specific pre-step adjustments,
        e.g. the degenerate PDE update at v = 0 for Heston.
        Default: no-op (returns V unchanged).
        """
        return V

    def _solve_one(self, kk, v0, texp, cp):
        """Full Douglas ADI solve in normalized space. Returns price / F₀."""
        M, L, N = self.n_asset, self.n_vol, self.n_time

        X_min, X_max, v_max = self._grid_bounds(v0, kk, texp)
        X  = np.linspace(X_min, X_max, M + 1)
        v  = np.linspace(0.0,   v_max,  L + 1)
        dX = (X_max - X_min) / M
        dv = v_max / L
        dt = texp / N

        # Terminal condition (τ = 0): intrinsic payoff broadcast over v-axis
        V = np.maximum(cp * (X[:, None] - kk), 0.0) * np.ones((M + 1, L + 1))
        # v = 0 initial condition is always intrinsic (τ = 0 payoff)
        V[:, 0] = np.maximum(cp * (X - kk), 0.0)
        V = self._apply_bc(V, kk, X, cp)

        for n in range(N):
            tau_old = n * dt
            tau_new = (n + 1) * dt
            V = self._pre_step(V, X, v, dX, dv, dt, tau_old, kk, cp)
            V = self._step(V, X, v, dX, dv, dt, tau_old, tau_new, kk, cp)

        return _bilinear(V, X, v, 1.0, v0)

    def _step(self, V, X, v, dX, dv, dt, tau_old, tau_new, kk, cp):
        """One Douglas step: advance V from τ_old to τ_new."""
        adt     = self.adi_theta * dt
        tau_mid = 0.5 * (tau_old + tau_new)

        aS, bS, aV, bV, gamma = self._coefficients(X, v, tau_mid)

        F0V = self._F0(V, gamma, dX, dv)
        F1V = self._F1(V, aS, bS, dX)
        F2V = self._F2(V, aV, bV, dv)

        Y0 = V + dt * (F0V + F1V + F2V)
        Y0 = self._apply_bc(Y0, kk, X, cp)

        Y1  = self._solve_S_implicit(Y0 - adt * F1V, adt, aS, bS, dX, X, kk, cp)
        Vn  = self._solve_v_implicit(Y1 - adt * F2V, adt, aV, bV, dv, X, kk, cp)
        return Vn

    # ── Spatial operators ─────────────────────────────────────────────────────

    @staticmethod
    def _F0(V, gamma, dX, dv):
        """Mixed cross-derivative operator (explicit)."""
        out = np.zeros_like(V)
        out[1:-1, 1:-1] = (
            gamma[1:-1, 1:-1]
            * (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2])
            / (4.0 * dX * dv)
        )
        return out

    @staticmethod
    def _F1(V, aS, bS, dX):
        """Asset-direction differential operator."""
        out = np.zeros_like(V)
        a = aS[1:-1, :] / dX**2      # (M-1, L+1) or broadcasts
        b = bS[1:-1, :] / (2.0 * dX)
        out[1:-1, :] = (
            a * (V[2:, :] - 2.0 * V[1:-1, :] + V[:-2, :])
            + b * (V[2:, :] - V[:-2, :])
        )
        return out

    @staticmethod
    def _F2(V, aV, bV, dv):
        """Vol-direction differential operator."""
        out = np.zeros_like(V)
        # aV, bV are (1, L+1) or (M+1, L+1); slice interior v-axis
        c = aV[..., 1:-1] / dv**2
        d = bV[..., 1:-1] / (2.0 * dv)
        out[:, 1:-1] = (
            c * (V[:, 2:] - 2.0 * V[:, 1:-1] + V[:, :-2])
            + d * (V[:, 2:] - V[:, :-2])
        )
        return out

    # ── Implicit sweeps (Thomas algorithm) ───────────────────────────────────

    def _solve_S_implicit(self, rhs, adt, aS, bS, dX, X, kk, cp):
        """Solve (I − adt·F₁)Y = rhs for interior S-points, all interior v."""
        M, L = self.n_asset, self.n_vol
        Y = self._apply_bc(rhs.copy(), kk, X, cp)

        # Coefficient arrays for interior (i=1..M-1, j=1..L-1)
        a = aS[1:M, 1:L] / dX**2      # (M-1, L-1)
        b = bS[1:M, 1:L] / (2.0 * dX) # (M-1, L-1)

        sub = -adt * (a - b)
        mid =  1.0 + adt * 2.0 * a
        sup = -adt * (a + b)

        rhs_int = rhs[1:M, 1:L].copy()
        # Fold known boundary S-values into right-hand side
        rhs_int[0,  :] -= sub[0,  :] * Y[0,  1:L]
        rhs_int[-1, :] -= sup[-1, :] * Y[M,  1:L]

        Y[1:M, 1:L] = _thomas_batch(sub, mid, sup, rhs_int)
        return self._apply_bc(Y, kk, X, cp)

    def _solve_v_implicit(self, rhs, adt, aV, bV, dv, X, kk, cp):
        """Solve (I − adt·F₂)Y = rhs for interior v-points, all S."""
        M, L = self.n_asset, self.n_vol
        Y = self._apply_bc(rhs.copy(), kk, X, cp)

        # Extract 1-D interior v-coefficients  (aV, bV may be (1,L+1))
        av = np.asarray(aV).ravel() if aV.shape[0] == 1 else aV[0]
        bv = np.asarray(bV).ravel() if bV.shape[0] == 1 else bV[0]
        c = av[1:L] / dv**2    # (L-1,)
        d = bv[1:L] / (2.0 * dv)

        sub = -adt * (c - d)
        mid =  1.0 + adt * 2.0 * c
        sup = -adt * (c + d)

        # Neumann BC at v_max: V[:, L] = V[:, L-1]  →  fold last sup into diagonal
        mid_eff = mid.copy()
        mid_eff[-1] += sup[-1]

        # Transpose so v-axis is first (Thomas iterates along v)
        rhs_int = rhs[:, 1:L].T.copy()       # (L-1, M+1)
        rhs_int[0, :] -= sub[0] * Y[:, 0]    # Dirichlet at v = 0

        sol = _thomas_batch(sub, mid_eff, sup, rhs_int)  # (L-1, M+1)
        Y[:, 1:L] = sol.T
        Y[:, L]   = Y[:, L - 1]              # enforce Neumann mirror
        return self._apply_bc(Y, kk, X, cp)


# ─────────────────────────────────────────────────────────────────────────────
# SABR mixin
# ─────────────────────────────────────────────────────────────────────────────

class SabrFinDiffMixin(SvFinDiffMixin):
    """
    SABR PDE coefficients for the generic Douglas ADI engine.

    PyFENG parameter mapping:
        ``sigma`` → σ₀ (initial vol level)
        ``beta``  → β  (CEV elasticity)
        ``vov``   → ν  (vol-of-vol)
        ``rho``   → ρ  (correlation)

    In the normalized forward space (X = F/F₀) the effective initial vol
    is  α = σ₀ / F₀^(1−β),  computed by ``SabrABC._variables(fwd, texp)``.
    """

    def _get_v0(self, fwd, texp):
        alpha, *_ = self._variables(fwd, texp)   # SabrABC helper
        return float(alpha)

    def _grid_bounds(self, v0, kk, texp):
        sq    = np.sqrt(max(texp, 0.0))
        X_ref = max(1.0, float(kk))
        X_max = max(4.0 * X_ref,
                    X_ref + 5.0 * v0 * (X_ref ** self.beta) * sq)
        v_max = max(1.0, 5.0 * v0,
                    v0 * (1.0 + 4.0 * self.vov * sq))
        X_min = 0.0 if self.beta > 0.0 else min(0.0, 1.0 - 5.0 * v0 * sq)
        return float(X_min), float(X_max), float(v_max)

    def _coefficients(self, X, v, tau):
        Xc = X[:, None]   # (M+1, 1)
        vr = v[None, :]   # (1,  L+1)
        aS    = 0.5 * vr**2 * Xc**(2.0 * self.beta)
        bS    = np.zeros_like(aS)                  # F is a martingale (r = q = 0)
        aV    = 0.5 * self.vov**2 * vr**2          # (1, L+1) — depends on v only
        bV    = np.zeros_like(aV)                  # SABR vol is a martingale
        gamma = self.rho * self.vov * vr**2 * Xc**self.beta
        return aS, bS, aV, bV, gamma

    def _apply_bc(self, V, kk, X, cp):
        return _standard_bc(V, kk, X, cp)


# ─────────────────────────────────────────────────────────────────────────────
# Heston mixin
# ─────────────────────────────────────────────────────────────────────────────

class HestonFinDiffMixin(SvFinDiffMixin):
    """
    Heston PDE coefficients for the generic Douglas ADI engine.

    PyFENG parameter mapping:
        ``sigma`` → v₀ (initial variance)
        ``vov``   → ξ  (CIR diffusion coefficient, vol-of-vol)
        ``mr``    → κ  (mean-reversion speed)
        ``theta`` → θ  (long-run variance)
        ``rho``   → ρ  (correlation)

    The variance process is scale-free so no forward normalization is needed.

    Boundary at v = 0:
        When the Feller condition 2κθ ≥ ξ² is violated the variance can reach
        zero and the PDE degenerates to  ∂V/∂τ = κθ ∂V/∂v.  This is integrated
        explicitly via ``_pre_step`` at every time step rather than freezing the
        option value at the intrinsic payoff (which would under-price options
        with positive expected variance recovery).
    """

    def _get_v0(self, fwd, texp):
        return float(self.sigma)   # initial variance; no fwd dependency

    def _grid_bounds(self, v0, kk, texp):
        X_max = max(4.0 * max(1.0, float(kk)),
                    1.0 + 5.0 * np.sqrt(v0 * max(texp, 0.0)))
        # CIR stationary std dev: ξ · √(θ / (2κ))
        cir_std = self.vov * np.sqrt(
            max(self.theta, 1e-8) / (2.0 * max(self.mr, 1e-8))
        )
        v_max = max(5.0 * float(v0), self.theta + 5.0 * cir_std)
        return 0.0, float(X_max), float(v_max)

    def _coefficients(self, X, v, tau):
        Xc = X[:, None]   # (M+1, 1)
        vr = v[None, :]   # (1,  L+1)
        aS    = 0.5 * vr * Xc**2
        bS    = np.zeros_like(aS)                  # r = q = 0 (normalized)
        aV    = 0.5 * self.vov**2 * vr             # (1, L+1)
        bV    = self.mr * (self.theta - vr)        # mean-reverting CIR drift
        gamma = self.rho * self.vov * vr * Xc
        return aS, bS, aV, bV, gamma

    def _apply_bc(self, V, kk, X, cp):
        """
        Heston boundary conditions.

        S = 0:     call → 0,       put → kk   (absorbing)
        S = S_max: call → X_max−kk, put → 0  (linear extrapolation)
        v = v_max: Neumann  dV/dv = 0         (far-field)
        v = 0:     NOT reset here — handled by ``_pre_step`` via the
                   degenerate PDE  ∂V/∂τ = κθ ∂V/∂v.
        """
        M = len(X) - 1
        if cp > 0:
            V[0,  :] = 0.0
            V[M,  :] = max(X[M] - kk, 0.0)
        else:
            V[0,  :] = kk
            V[M,  :] = 0.0
        V[:, -1] = V[:, -2]   # Neumann at v_max
        return V

    def _pre_step(self, V, X, v, dX, dv, dt, tau, kk, cp):
        """
        Explicit degenerate PDE step at v = 0.

        Integrates  ∂V/∂τ = κθ (V[:,1] − V[:,0]) / dv  for one time step
        using forward Euler before the Douglas ADI sweep, then re-applies the
        S-direction and v_max boundary conditions.
        """
        V[:, 0] += dt * self.mr * self.theta * (V[:, 1] - V[:, 0]) / dv
        return self._apply_bc(V, kk, X, cp)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete classes
# ─────────────────────────────────────────────────────────────────────────────

from .sabr import SabrABC      # noqa: E402
from .heston import HestonABC  # noqa: E402


class SabrFinDiff(SabrFinDiffMixin, SabrABC):
    """European SABR option pricing via 2D Douglas ADI finite differences."""
    _benchmark_file = 'sabr_benchmark.xlsx'


class HestonFinDiff(HestonFinDiffMixin, HestonABC):
    """European Heston option pricing via 2D Douglas ADI finite differences."""
    _benchmark_file = 'heston_benchmark.xlsx'
