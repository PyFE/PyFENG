"""
Generic 2D Douglas ADI finite-difference mixin for stochastic-volatility models.

Works in normalized forward space (F₀ = 1, r = q = 0 internally).
Discounting and forward-scaling are handled by the outer ``price()`` method.

The 2D PDE for V(X, v, τ), τ = T − t has the form::

    dV/dτ = (F₀ + F₁ + F₂) V

    F₀: γ(X,v)  · ∂²V/∂X∂v              (explicit; mixed derivative)
    F₁: aS(X,v) · ∂²V/∂X² + bS(X,v) · ∂V/∂X   (S-direction implicit)
    F₂: aV(X,v) · ∂²V/∂v² + bV(X,v) · ∂V/∂v   (v-direction implicit)

One Douglas step τₙ → τₙ₊₁ (θ = ½ gives Crank–Nicolson-like accuracy)::

    Y₀ = Vⁿ + dt (F₀ + F₁ + F₂) Vⁿ                  (explicit predictor)
    (I − θ dt F₁) Y₁ = Y₀ − θ dt F₁ Vⁿ              (S-implicit corrector)
    (I − θ dt F₂) Vⁿ⁺¹ = Y₁ − θ dt F₂ Vⁿ            (v-implicit corrector)

To use for a new SV model, subclass ``SvFinDiffABC`` and implement
four abstract methods: ``_get_v0``, ``_grid_bounds``, ``_coefficients``,
``_apply_bdd_cond``.

References:
    von Sydow et al. (2019) BENCHOP-SLV, Int. J. Comput. Math.
    K. in 't Hout & S. Foulon (2010), ADI FD schemes for Heston.

Credits:
    Adapted from the MATH5030 Group 16 project (``mafn-finite-difference``),
    which provided the original Douglas ADI implementation for SABR and
    extended Heston models.
    Repository: https://github.com/py2025/finite-difference
"""

import abc
import numpy as np
from dataclasses import dataclass, field


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
# Generic ADI mixin ABC
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SvFinDiffABC(abc.ABC):
    """
    Generic Douglas-scheme 2D ADI mixin for stochastic-volatility models.

    Subclass and implement four abstract methods to plug in any model::

        class MyModelFinDiff(SvFinDiffABC, MyModelABC):
            def _get_v0(self, fwd, texp):         return ...
            def _grid_bounds(self, v0, kk, texp): return X_min, X_max, v_max
            def _coefficients(self, X, v):        return aS, bS, aV, bV, gamma
            def _apply_bdd_cond(self, V, kk, X, cp):    return _standard_bc(V, kk, X, cp)

    The engine works in normalized forward space (X = F/F₀, so X₀ = 1, r=q=0).
    ``price()`` handles forward/discount scaling externally.
    """

    # Numerical parameters as kw_only dataclass fields
    # n_grid = (N_time, M_price, L_vol); matches BENCHOP-SLV convention M = 2L, N ≈ L
    n_grid:    tuple = field(default=(80, 80, 40), kw_only=True, metadata={'kind': 'numerical'})
    adi_theta: float = field(default=0.5,          kw_only=True, metadata={'kind': 'numerical'})

    def __post_init__(self):
        self.n_grid    = tuple(int(x) for x in self.n_grid)
        self.adi_theta = float(self.adi_theta)

    def configure(self, n_grid=None, adi_theta=None):
        """
        Configure the ADI grid.

        Args:
            n_grid:    (N_time, M_price, L_vol) — number of time steps and
                       spatial grid intervals in the price and vol directions.
                       BENCHOP-SLV convention: M = 2L, N ≈ L.
            adi_theta: Douglas weight; 0.5 → Crank–Nicolson accuracy
        """
        if n_grid is not None:
            self.n_grid = n_grid
        if adi_theta is not None:
            self.adi_theta = adi_theta
        self.__post_init__()
        return self


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
    def _coefficients(self, X, v):
        """
        PDE operator coefficients on the full (M+1) × (L+1) grid.

        Parameters are assumed constant (time-independent).

        Args:
            X: asset grid, shape (M+1,)
            v: vol/variance grid, shape (L+1,)

        Returns:
            aS, bS, aV, bV, gamma — each broadcastable to (M+1, L+1).

            aS: (1/2) · diffusion²  in S-direction
            bS: drift               in S-direction  (0 if F is a martingale)
            aV: (1/2) · diffusion²  in v-direction
            bV: drift               in v-direction  (0 for SABR; κ(θ−v) for Heston)
            gamma: ρ · σ_S · σ_v   (mixed-derivative coefficient)
        """

    @abc.abstractmethod
    def _apply_bdd_cond(self, V, kk, X, cp):
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

    def _pre_step(self, V, X, v, dX, dv, dt, kk, cp):
        """
        Hook called before each Douglas step.

        Override in subclasses to apply model-specific pre-step adjustments,
        e.g. the degenerate PDE update at v = 0 for Heston.
        Default: no-op (returns V unchanged).
        """
        return V

    def _solve_one(self, kk, v0, texp, cp):
        """Full Douglas ADI solve in normalized space. Returns price / F₀."""
        N, M, L = self.n_grid

        X_min, X_max, v_max = self._grid_bounds(v0, kk, texp)
        X  = np.linspace(X_min, X_max, M + 1)
        v  = np.linspace(0.0,   v_max,  L + 1)
        dX = (X_max - X_min) / M
        dv = v_max / L
        dt = texp / N

        # Terminal condition (τ = 0): intrinsic payoff broadcast over v-axis
        V = np.maximum(cp * (X[:, None] - kk), 0.0) * np.ones((M + 1, L + 1))
        V[:, 0] = np.maximum(cp * (X - kk), 0.0)
        V = self._apply_bdd_cond(V, kk, X, cp)

        for _ in range(N):
            V = self._pre_step(V, X, v, dX, dv, dt, kk, cp)
            V = self._step(V, X, v, dX, dv, dt, kk, cp)

        return _bilinear(V, X, v, 1.0, v0)

    def _step(self, V, X, v, dX, dv, dt, kk, cp):
        """One Douglas step: advance V by one time increment dt."""
        adt = self.adi_theta * dt

        aS, bS, aV, bV, gamma = self._coefficients(X, v)

        F0V = self._F0(V, gamma, dX, dv)
        F1V = self._F1(V, aS, bS, dX)
        F2V = self._F2(V, aV, bV, dv)

        Y0 = V + dt * (F0V + F1V + F2V)
        Y0 = self._apply_bdd_cond(Y0, kk, X, cp)

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
        a = aS[1:-1, :] / dX**2
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
        _, M, L = self.n_grid
        Y = self._apply_bdd_cond(rhs.copy(), kk, X, cp)

        a = aS[1:M, 1:L] / dX**2
        b = bS[1:M, 1:L] / (2.0 * dX)

        sub = -adt * (a - b)
        mid =  1.0 + adt * 2.0 * a
        sup = -adt * (a + b)

        rhs_int = rhs[1:M, 1:L].copy()
        rhs_int[0,  :] -= sub[0,  :] * Y[0,  1:L]
        rhs_int[-1, :] -= sup[-1, :] * Y[M,  1:L]

        Y[1:M, 1:L] = _thomas_batch(sub, mid, sup, rhs_int)
        return self._apply_bdd_cond(Y, kk, X, cp)

    def _solve_v_implicit(self, rhs, adt, aV, bV, dv, X, kk, cp):
        """Solve (I − adt·F₂)Y = rhs for interior v-points, all S."""
        _, M, L = self.n_grid
        Y = self._apply_bdd_cond(rhs.copy(), kk, X, cp)

        av = np.asarray(aV).ravel() if aV.shape[0] == 1 else aV[0]
        bv = np.asarray(bV).ravel() if bV.shape[0] == 1 else bV[0]
        c = av[1:L] / dv**2
        d = bv[1:L] / (2.0 * dv)

        sub = -adt * (c - d)
        mid =  1.0 + adt * 2.0 * c
        sup = -adt * (c + d)

        # Neumann BC at v_max: fold last sup into diagonal
        mid_eff = mid.copy()
        mid_eff[-1] += sup[-1]

        rhs_int = rhs[:, 1:L].T.copy()       # (L-1, M+1)
        rhs_int[0, :] -= sub[0] * Y[:, 0]    # Dirichlet at v = 0

        sol = _thomas_batch(sub, mid_eff, sup, rhs_int)  # (L-1, M+1)
        Y[:, 1:L] = sol.T
        Y[:, L]   = Y[:, L - 1]              # enforce Neumann mirror
        return self._apply_bdd_cond(Y, kk, X, cp)


# ─────────────────────────────────────────────────────────────────────────────
# Imports — deferred to avoid circular references
# ─────────────────────────────────────────────────────────────────────────────

from .sabr import SabrABC                    # noqa: E402
from .heston import HestonABC, HestonCevABC  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# SABR concrete pricer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SabrFinDiff(SvFinDiffABC, SabrABC):
    """
    European SABR option pricing via 2D Douglas ADI finite differences.

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

    def _coefficients(self, X, v):
        Xc = X[:, None]   # (M+1, 1)
        vr = v[None, :]   # (1,  L+1)
        aS    = 0.5 * vr**2 * Xc**(2.0 * self.beta)
        bS    = np.zeros_like(aS)                  # F is a martingale (r = q = 0)
        aV    = 0.5 * self.vov**2 * vr**2          # (1, L+1) — depends on v only
        bV    = np.zeros_like(aV)                  # SABR vol is a martingale
        gamma = self.rho * self.vov * vr**2 * Xc**self.beta
        return aS, bS, aV, bV, gamma

    def _apply_bdd_cond(self, V, kk, X, cp):
        return _standard_bc(V, kk, X, cp)


# ─────────────────────────────────────────────────────────────────────────────
# Heston concrete pricer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HestonFinDiff(SvFinDiffABC, HestonABC):
    """
    European Heston option pricing via 2D Douglas ADI finite differences.

    PyFENG parameter mapping:
        ``sigma`` → v₀ (initial variance)
        ``vov``   → ξ  (CIR diffusion coefficient, vol-of-vol)
        ``mr``    → κ  (mean-reversion speed)
        ``theta`` → θ  (long-run variance)
        ``rho``   → ρ  (correlation)

    Boundary at v = 0:
        When the Feller condition 2κθ ≥ ξ² is violated the variance can reach
        zero and the PDE degenerates to  ∂V/∂τ = κθ ∂V/∂v.  This is integrated
        explicitly via ``_pre_step`` at every time step rather than freezing the
        option value at the intrinsic payoff.
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

    def _coefficients(self, X, v):
        Xc = X[:, None]   # (M+1, 1)
        vr = v[None, :]   # (1,  L+1)
        aS    = 0.5 * vr * Xc**2
        bS    = np.zeros_like(aS)                  # r = q = 0 (normalized)
        aV    = 0.5 * self.vov**2 * vr             # (1, L+1)
        bV    = self.mr * (self.theta - vr)        # mean-reverting CIR drift
        gamma = self.rho * self.vov * vr * Xc
        return aS, bS, aV, bV, gamma

    def _apply_bdd_cond(self, V, kk, X, cp):
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

    def _pre_step(self, V, X, v, dX, dv, dt, kk, cp):
        """
        v = 0 boundary update before each Douglas step.

        **Feller violated** (2κθ < ξ²): v = 0 is attainable.  The PDE
        degenerates to  ∂V/∂τ = κθ ∂V/∂v, integrated by forward Euler.
        CFL stability requires  κθ · dt/dv ≤ 1.

        **Feller satisfied** (2κθ ≥ ξ²): v = 0 is unreachable.  The v = 0
        column is pinned to the absorbing Dirichlet value (intrinsic payoff),
        which is both physically correct and unconditionally stable.
        """
        if 2.0 * self.mr * self.theta < self.vov**2:   # Feller violated
            V[:, 0] += dt * self.mr * self.theta * (V[:, 1] - V[:, 0]) / dv
        else:                                           # Feller satisfied
            V[:, 0] = np.maximum(cp * (X - kk), 0.0)
        return self._apply_bdd_cond(V, kk, X, cp)


# ─────────────────────────────────────────────────────────────────────────────
# CEV-Heston concrete pricer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HestonCevFinDiff(HestonCevABC, HestonFinDiff):
    """
    European CEV-Heston option pricing via 2D Douglas ADI finite differences.

    Generalises ``HestonFinDiff`` with a CEV elasticity ``beta``::

        dS = sqrt(v) · S^beta · dW_S,   beta = 1 → standard Heston

    Inherits grid bounds, boundary conditions, and the degenerate-PDE
    ``_pre_step`` from ``HestonFinDiff``; overrides only ``_coefficients``.

    PDE coefficients that differ from standard Heston:
        aS    = ½ · v · X^(2β)   (Heston: ½·v·X²)
        gamma = ρ · ξ · v · X^β  (Heston: ρ·ξ·v·X)
    """

    def _coefficients(self, X, v):
        Xc = X[:, None]   # (M+1, 1)
        vr = v[None, :]   # (1,  L+1)
        aS    = 0.5 * vr * Xc**(2.0 * self.beta)
        bS    = np.zeros_like(aS)
        aV    = 0.5 * self.vov**2 * vr             # unchanged from Heston
        bV    = self.mr * (self.theta - vr)        # unchanged from Heston
        gamma = self.rho * self.vov * vr * Xc**self.beta
        return aS, bS, aV, bV, gamma
