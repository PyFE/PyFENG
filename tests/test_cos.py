"""
Tests for COS-method pricing engines: BsmCos, VarGammaCos, HestonCos.

Cross-validates each COS pricer against the corresponding FFT/analytic
pricer in PyFENG and checks structural properties (put-call parity,
vectorisation, multiple maturities).

Run:
    cd PyFENG
    python -m pytest tests/test_cos.py -v
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


# Paper benchmark parameters — F&O (2008) Table 2
HESTON = dict(sigma=0.0175, vov=0.5751, mr=1.5768, rho=-0.5711, theta=0.0398)
# ATM call: T=1 → 5.785155435, T=10 → 22.318945791

# Variance Gamma parameters satisfying 1 - theta*nu - 0.5*sigma^2*nu > 0
VG = dict(sigma=0.12, vov=0.174, theta=-0.14)

STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
SPOT = 100.0


class TestBsmCos(unittest.TestCase):

    def setUp(self):
        self.cos = pf.BsmCos(sigma=0.2, intr=0.05, divr=0.1)
        self.bsm = pf.Bsm(sigma=0.2, intr=0.05, divr=0.1)

    def test_call_vs_analytic(self):
        texp = 1.2
        cos_p = self.cos.price(STRIKES, SPOT, texp, cp=1)
        bsm_p = self.bsm.price(STRIKES, SPOT, texp, cp=1)
        np.testing.assert_allclose(cos_p, bsm_p, atol=1e-6,
                                   err_msg="BsmCos calls deviate from analytic Bsm")

    def test_put_vs_analytic(self):
        texp = 1.2
        cos_p = self.cos.price(STRIKES, SPOT, texp, cp=-1)
        bsm_p = self.bsm.price(STRIKES, SPOT, texp, cp=-1)
        np.testing.assert_allclose(cos_p, bsm_p, atol=1e-6,
                                   err_msg="BsmCos puts deviate from analytic Bsm")

    def test_put_call_parity(self):
        texp = 1.2
        fwd = SPOT * np.exp((0.05 - 0.1) * texp)
        df = np.exp(-0.05 * texp)
        c = self.cos.price(STRIKES, SPOT, texp, cp=1)
        p = self.cos.price(STRIKES, SPOT, texp, cp=-1)
        np.testing.assert_allclose(c - p, df * (fwd - STRIKES), atol=1e-8,
                                   err_msg="BsmCos put-call parity violated")

    def test_scalar_output(self):
        result = self.cos.price(100.0, SPOT, 1.0, cp=1)
        self.assertIsInstance(result, float)


class TestVarGammaCos(unittest.TestCase):

    def setUp(self):
        self.cos = pf.VarGammaCos(**VG)
        self.fft = pf.VarGammaFft(**VG)

    def test_calls_vs_fft(self):
        texp = 1.0
        cos_p = self.cos.price(STRIKES, SPOT, texp, cp=1)
        fft_p = self.fft.price(STRIKES, SPOT, texp, cp=1)
        np.testing.assert_allclose(cos_p, fft_p, atol=1e-5,
                                   err_msg="VarGammaCos calls deviate from VarGammaFft")

    def test_puts_vs_fft(self):
        texp = 1.0
        cos_p = self.cos.price(STRIKES, SPOT, texp, cp=-1)
        fft_p = self.fft.price(STRIKES, SPOT, texp, cp=-1)
        np.testing.assert_allclose(cos_p, fft_p, atol=1e-5,
                                   err_msg="VarGammaCos puts deviate from VarGammaFft")

    def test_put_call_parity(self):
        texp = 1.0
        df = np.exp(0.0)   # intr=divr=0 by default
        c = self.cos.price(STRIKES, SPOT, texp, cp=1)
        p = self.cos.price(STRIKES, SPOT, texp, cp=-1)
        np.testing.assert_allclose(c - p, df * (SPOT - STRIKES), atol=1e-5,
                                   err_msg="VarGammaCos put-call parity violated")


class TestHestonCos(unittest.TestCase):

    def setUp(self):
        self.cos = pf.HestonCos(**HESTON)
        self.fft = pf.HestonFft(**HESTON)

    # ── agreement with HestonFft ──────────────────────────────────────────

    def test_calls_vs_fft_T1(self):
        cos_p = self.cos.price(STRIKES, SPOT, 1.0, cp=1)
        fft_p = self.fft.price(STRIKES, SPOT, 1.0, cp=1)
        np.testing.assert_allclose(cos_p, fft_p, atol=1e-4,
                                   err_msg="HestonCos calls T=1 deviate from HestonFft")

    def test_calls_vs_fft_T10(self):
        cos_p = self.cos.price(STRIKES, SPOT, 10.0, cp=1)
        fft_p = self.fft.price(STRIKES, SPOT, 10.0, cp=1)
        np.testing.assert_allclose(cos_p, fft_p, atol=1e-4,
                                   err_msg="HestonCos calls T=10 deviate from HestonFft")

    def test_paper_benchmark_T1(self):
        """F&O Table 2: ATM call at T=1 ≈ 5.785155435."""
        ref = self.fft.price(100.0, SPOT, 1.0, cp=1)
        cos_p = self.cos.price(100.0, SPOT, 1.0, cp=1)
        np.testing.assert_allclose(cos_p, ref, atol=1e-4,
                                   err_msg="HestonCos T=1 ATM price off from HestonFft reference")

    def test_paper_benchmark_T10(self):
        """F&O Table 2: ATM call at T=10 ≈ 22.318945791."""
        ref = self.fft.price(100.0, SPOT, 10.0, cp=1)
        cos_p = self.cos.price(100.0, SPOT, 10.0, cp=1)
        np.testing.assert_allclose(cos_p, ref, atol=1e-4,
                                   err_msg="HestonCos T=10 ATM price off from HestonFft reference")

    # ── structural correctness ────────────────────────────────────────────

    def test_put_call_parity(self):
        texp = 1.0
        df = np.exp(0.0)
        c = self.cos.price(STRIKES, SPOT, texp, cp=1)
        p = self.cos.price(STRIKES, SPOT, texp, cp=-1)
        np.testing.assert_allclose(c - p, df * (SPOT - STRIKES), atol=1e-4,
                                   err_msg="HestonCos put-call parity violated")

    def test_vectorized_vs_scalar(self):
        texp = 1.0
        vec = self.cos.price(STRIKES, SPOT, texp, cp=1)
        scalar = np.array([self.cos.price(float(k), SPOT, texp, cp=1) for k in STRIKES])
        np.testing.assert_allclose(vec, scalar, atol=1e-12,
                                   err_msg="HestonCos vectorised != scalar loop")

    def test_multiple_maturities(self):
        for texp in [1.0, 2.0, 5.0, 10.0]:
            cos_p = self.cos.price(STRIKES, SPOT, texp, cp=1)
            fft_p = self.fft.price(STRIKES, SPOT, texp, cp=1)
            np.testing.assert_allclose(
                cos_p, fft_p, atol=1e-4,
                err_msg=f"HestonCos vs HestonFft mismatch at T={texp}"
            )

    def test_puts_vs_fft(self):
        cos_p = self.cos.price(STRIKES, SPOT, 1.0, cp=-1)
        fft_p = self.fft.price(STRIKES, SPOT, 1.0, cp=-1)
        np.testing.assert_allclose(cos_p, fft_p, atol=1e-4,
                                   err_msg="HestonCos puts deviate from HestonFft")

    def test_scalar_output(self):
        result = self.cos.price(100.0, SPOT, 1.0, cp=1)
        self.assertIsInstance(result, float)

    def test_mixed_cp(self):
        """Mixed call/put array in one price() call."""
        cp = np.array([1, -1, 1, -1, 1])
        cos_p = self.cos.price(STRIKES, SPOT, 1.0, cp=cp)
        ref = np.array([
            self.cos.price(float(k), SPOT, 1.0, cp=int(c))
            for k, c in zip(STRIKES, cp)
        ])
        np.testing.assert_allclose(cos_p, ref, atol=1e-12,
                                   err_msg="HestonCos mixed cp array mismatch")


if __name__ == "__main__":
    unittest.main()
