"""
Benchmark tests for sv_fin_diff.py — 2D Douglas ADI finite-difference mixin.

Parameters and reference values are loaded from the PyFENG benchmark Excel files
via Class.init_benchmark(no):

    SabrFinDiff.init_benchmark(20)  → BENCHOP-SLV Set I
    SabrFinDiff.init_benchmark(21)  → BENCHOP-SLV Set II
    HestonFinDiff.init_benchmark(10) → BENCHOP-SLV Heston Set I

Reference: von Sydow et al. (2018) BENCHOP-SLV, Int. J. Comput. Math.
https://doi.org/10.1080/00207160.2018.1544368
"""

import numpy as np
import pytest

import pyfeng as pf


# ─────────────────────────────────────────────────────────────────────────────
# SABR benchmark sets
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("set_no", [20, 21])
def test_sabr_adi_benchop(set_no):
    """BENCHOP-SLV SABR sets 20 (Set I) and 21 (Set II).

    Set 20: S0=0.5, σ₀=0.5, β=0.5, ρ=0, ν=0.4, T=2
    Set 21: S0=0.07, σ₀=0.4, β=0.5, ρ=−0.6, ν=0.8, T=10
    Tolerance: abs=3e-3 (consistent with the BENCHOP-SLV paper error budget).
    """
    m, df, info = pf.SabrFinDiff.init_benchmark(set_no)
    m.set_num_params(n_grid=(80, 80, 40))

    args   = info["args_pricing"]
    prices = m.price(**args)
    refs   = info["val"]

    for K, p, r in zip(args["strike"], prices, refs):
        assert p == pytest.approx(r, abs=3e-3), (
            f"SABR set {set_no} K={K:.6f}: got {p:.6f}, expected {r:.6f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Heston benchmark set
# ─────────────────────────────────────────────────────────────────────────────

def test_heston_adi_benchop():
    """BENCHOP-SLV Heston Set I (set 10).

    S0=100, K∈{133.33, 100, 80}, T=1
    κ=2.58, θ=0.043, ξ=1, ρ=−0.36, V₀=0.114

    Note: these parameters violate the Feller condition (2κθ=0.22 < ξ²=1),
    making convergence with uniform grids slower than for well-conditioned
    cases.  The coarse (80×40×80) grid achieves ~1–2 % relative error; tighter
    tolerances require Richardson extrapolation or finer grids.
    """
    m, df, info = pf.HestonFinDiff.init_benchmark(10)
    m.set_num_params(n_grid=(80, 80, 40))

    args   = info["args_pricing"]
    prices = m.price(**args)
    refs   = info["val"]

    for K, p, r in zip(args["strike"], prices, refs):
        assert p == pytest.approx(r, abs=0.15), (
            f"Heston set 10 K={K:.4f}: got {p:.6f}, expected {r:.6f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Put-call parity  (normalized space: C − P = S₀ − K  when r = 0)
# ─────────────────────────────────────────────────────────────────────────────

def test_sabr_put_call_parity():
    """SABR put-call parity: C − P = S₀ − K (r = 0)."""
    m, _, info = pf.SabrFinDiff.init_benchmark(20)
    m.set_num_params(n_grid=(80, 80, 40))

    args = info["args_pricing"]
    S0 = args["spot"]; K = float(args["strike"][1])  # ATM strike
    call = m.price(K, S0, args["texp"], cp=1)
    put  = m.price(K, S0, args["texp"], cp=-1)
    assert call - put == pytest.approx(S0 - K, abs=1e-3), (
        f"SABR PCP: C−P={call-put:.6f}, S₀−K={S0-K:.6f}"
    )


def test_heston_put_call_parity():
    """Heston put-call parity at ATM: C − P = S₀ − K (r = 0)."""
    m, _, info = pf.HestonFinDiff.init_benchmark(10)
    m.set_num_params(n_grid=(80, 80, 40))

    args = info["args_pricing"]
    S0 = float(args["spot"]); K = float(args["strike"][1])  # ATM strike (K=100)
    call = m.price(K, S0, args["texp"], cp=1)
    put  = m.price(K, S0, args["texp"], cp=-1)
    assert call - put == pytest.approx(S0 - K, abs=1e-2), (
        f"Heston PCP: C−P={call-put:.6f}, S₀−K={S0-K:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────

def test_sabr_array_strikes():
    """price() should accept an ndarray of strikes and return the same shape."""
    m, _, info = pf.SabrFinDiff.init_benchmark(20)
    m.set_num_params(n_grid=(80, 80, 40))

    args   = info["args_pricing"]
    prices = m.price(**args)
    assert prices.shape == args["strike"].shape
    assert np.all(prices > 0)


def test_degenerate_to_cev():
    """With vov→0, both SabrFinDiff and HestonCevFinDiff reduce to pure CEV.

    SABR (vov→0):      dF = sigma · F^beta · dW  →  CEV with sigma_cev = m.sigma
    HestonCev (vov→0): dS = sqrt(v) · S^beta · dW, v frozen at sigma
                                                  →  CEV with sigma_cev = sqrt(m.sigma)

    Strong mr pins the Heston variance tightly to sigma=theta; Feller is satisfied
    (2κθ >> ξ²) so the v=0 boundary uses the stable Dirichlet condition.
    Tolerance abs=1e-3: residual error is O(vov²) plus finite-grid discretisation.
    """
    beta = 0.5
    S0, texp = 1.0, 1.0
    strikes = np.array([0.8, 1.0, 1.2])

    # SABR with vov→0: effective CEV vol = m_sabr.sigma
    m_sabr = pf.SabrFinDiff(sigma=0.3, vov=1e-4, rho=0.0, beta=beta)
    m_sabr.set_num_params(n_grid=(80, 80, 40))
    p_sabr = m_sabr.price(strikes, S0, texp, cp=1)

    # HestonCev with vov→0, strong mr: variance frozen at sigma=theta,
    # effective CEV vol = sqrt(m_hcev.sigma)
    m_hcev = pf.HestonCevFinDiff(sigma=0.09, vov=1e-4, mr=100.0, theta=0.09, rho=0.0, beta=beta)
    m_hcev.set_num_params(n_grid=(80, 80, 40))
    p_hcev = m_hcev.price(strikes, S0, texp, cp=1)

    # Both converge to the same CEV: sigma_cev = m_sabr.sigma = sqrt(m_hcev.sigma)
    p_ref = pf.Cev(sigma=np.sqrt(m_hcev.sigma), beta=beta).price(strikes, S0, texp, cp=1)

    for K, pr, ps, ph in zip(strikes, p_ref, p_sabr, p_hcev):
        assert ps == pytest.approx(pr, abs=1e-3), (
            f"SABR→CEV  K={K}: got {ps:.6f}, expected {pr:.6f}"
        )
        assert ph == pytest.approx(pr, abs=1e-3), (
            f"HestonCev→CEV  K={K}: got {ph:.6f}, expected {pr:.6f}"
        )


def test_heston_positive_price():
    """Heston ATM call is positive and finite."""
    m, _, info = pf.HestonFinDiff.init_benchmark(10)
    m.set_num_params(n_grid=(80, 80, 40))

    args  = info["args_pricing"]
    price = m.price(float(args["strike"][1]), float(args["spot"]), args["texp"], cp=1)
    assert np.isfinite(price)
    assert price > 0
