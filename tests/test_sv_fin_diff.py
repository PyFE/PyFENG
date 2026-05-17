"""
Benchmark tests for sv_fin_diff.py — 2D Douglas ADI finite-difference mixin.

Reference: von Sydow et al. (2019) BENCHOP-SLV, Int. J. Comput. Math.
           Sets I and II correspond to the paper's SABR test problems.
           Heston set uses parameters from the paper's §2.2 Heston benchmark.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyfeng as pf


# ─────────────────────────────────────────────────────────────────────────────
# SABR Set I  (BENCHOP-SLV §2.1, parameter set I)
#   S0 = 0.5, σ₀ = α = 0.5, β = 0.5, ρ = 0, ν = 0.4, T = 2, r = 0
# ─────────────────────────────────────────────────────────────────────────────

SABR_SET_I = dict(sigma=0.5, beta=0.5, rho=0.0, vov=0.4)
SABR_SET_I_SPOT = 0.5
SABR_SET_I_TEXP = 2.0
SABR_SET_I_ROWS = [
    (0.434062, 0.221383196830866),
    (0.500000, 0.193836689413803),
    (0.575955, 0.166240814653231),
]

# ─────────────────────────────────────────────────────────────────────────────
# SABR Set II  (BENCHOP-SLV §2.1, parameter set II)
#   S0 = 0.07, σ₀ = α = 0.4, β = 0.5, ρ = −0.6, ν = 0.8, T = 10, r = 0
# ─────────────────────────────────────────────────────────────────────────────

SABR_SET_II = dict(sigma=0.4, beta=0.5, rho=-0.6, vov=0.8)
SABR_SET_II_SPOT = 0.07
SABR_SET_II_TEXP = 10.0
SABR_SET_II_ROWS = [
    (0.051023, 0.052450313614407),
    (0.070000, 0.046658753491306),
    (0.096036, 0.039291470612989),
]

# ─────────────────────────────────────────────────────────────────────────────
# Heston Set  (BENCHOP-SLV §2.2)
#   K = 100, T = 1, κ = 2.58, θ = 0.043, ξ = 1, ρ = −0.36, V₀ = 0.114
# ─────────────────────────────────────────────────────────────────────────────

HESTON_PARAMS = dict(sigma=0.114, vov=1.0, mr=2.58, theta=0.043, rho=-0.36)
HESTON_STRIKE = 100.0
HESTON_TEXP = 1.0
HESTON_ROWS = [
    (75.0,  0.908502728459621),
    (100.0, 9.046650119220966),
    (125.0, 28.514786399298796),
]


# ─────────────────────────────────────────────────────────────────────────────
# SABR tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strike,ref", SABR_SET_I_ROWS)
def test_sabr_set_i(strike, ref):
    """BENCHOP-SLV parameter set I: β=0.5, ρ=0, T=2."""
    m = pf.SabrFinDiff(**SABR_SET_I)
    m.set_num_params(n_asset=80, n_vol=40, n_time=80)
    price = m.price(strike, SABR_SET_I_SPOT, SABR_SET_I_TEXP, cp=1)
    assert price == pytest.approx(ref, abs=3e-3), (
        f"Set I K={strike:.6f}: got {price:.6f}, expected {ref:.6f}"
    )


@pytest.mark.parametrize("strike,ref", SABR_SET_II_ROWS)
def test_sabr_set_ii(strike, ref):
    """BENCHOP-SLV parameter set II: β=0.5, ρ=−0.6, T=10."""
    m = pf.SabrFinDiff(**SABR_SET_II)
    m.set_num_params(n_asset=80, n_vol=40, n_time=80)
    price = m.price(strike, SABR_SET_II_SPOT, SABR_SET_II_TEXP, cp=1)
    assert price == pytest.approx(ref, abs=3e-3), (
        f"Set II K={strike:.6f}: got {price:.6f}, expected {ref:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Heston tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spot,ref", HESTON_ROWS)
def test_heston_benchop(spot, ref):
    """BENCHOP-SLV Heston: κ=2.58, θ=0.043, ξ=1, ρ=−0.36, V₀=0.114, K=100, T=1.

    Note: these parameters violate the Feller condition (2κθ = 0.22 < ξ² = 1),
    making convergence with uniform grids slower than for well-conditioned cases.
    The solver matches the reference to within ~1–2 % for a coarse (80×40×80) grid.
    Tighter tolerances require Richardson extrapolation or finer grids.
    """
    m = pf.HestonFinDiff(**HESTON_PARAMS)
    m.set_num_params(n_asset=80, n_vol=40, n_time=80)
    price = m.price(HESTON_STRIKE, spot, HESTON_TEXP, cp=1)
    assert price == pytest.approx(ref, abs=0.15), (
        f"Heston S0={spot}: got {price:.6f}, expected {ref:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Put-call parity (normalized space: C − P ≈ F·e^{-rT} − K·e^{-rT} = S0 − K when r=0)
# ─────────────────────────────────────────────────────────────────────────────

def test_sabr_put_call_parity():
    """Put-call parity: C − P = S0 − K (r=0)."""
    m = pf.SabrFinDiff(**SABR_SET_I)
    m.set_num_params(n_asset=80, n_vol=40, n_time=80)
    S0, K, T = SABR_SET_I_SPOT, 0.500000, SABR_SET_I_TEXP
    call = m.price(K, S0, T, cp=1)
    put  = m.price(K, S0, T, cp=-1)
    assert call - put == pytest.approx(S0 - K, abs=1e-3), (
        f"PCP violation: C-P={call-put:.6f}, S0-K={S0-K:.6f}"
    )


def test_heston_put_call_parity():
    """Put-call parity for Heston at-the-money (r=0)."""
    m = pf.HestonFinDiff(**HESTON_PARAMS)
    m.set_num_params(n_asset=80, n_vol=40, n_time=80)
    S0, K, T = 100.0, 100.0, HESTON_TEXP
    call = m.price(K, S0, T, cp=1)
    put  = m.price(K, S0, T, cp=-1)
    assert call - put == pytest.approx(S0 - K, abs=1e-2), (
        f"PCP violation: C-P={call-put:.6f}, S0-K={S0-K:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests: array input and positive prices
# ─────────────────────────────────────────────────────────────────────────────

def test_sabr_array_strikes():
    """price() should accept an ndarray of strikes and return same shape."""
    m = pf.SabrFinDiff(**SABR_SET_I)
    m.set_num_params(n_asset=80, n_vol=40, n_time=80)
    strikes = np.array([k for k, _ in SABR_SET_I_ROWS])
    prices = m.price(strikes, SABR_SET_I_SPOT, SABR_SET_I_TEXP, cp=1)
    assert prices.shape == strikes.shape
    assert np.all(prices > 0)


def test_heston_positive_price():
    """Spot check: Heston ATM call is positive and finite."""
    m = pf.HestonFinDiff(**HESTON_PARAMS)
    m.set_num_params(n_asset=80, n_vol=40, n_time=80)
    price = m.price(100.0, 100.0, HESTON_TEXP, cp=1)
    assert np.isfinite(price)
    assert price > 0
