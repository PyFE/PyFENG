import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


# ── shared test parameters ────────────────────────────────────────────────────
_PARAMS_VG = [
    dict(sigma=0.12, nu=0.2,  theta=-0.14),
    dict(sigma=0.20, nu=0.5,  theta=-0.10),
    dict(sigma=0.15, nu=0.1,  theta=0.05),
]
_PARAMS_NIG = [
    dict(sigma=0.12, nu=0.2,  theta=-0.14),
    dict(sigma=0.20, nu=0.5,  theta=-0.10),
    dict(sigma=0.15, nu=0.1,  theta=0.05),
]
_SV_PARAMS = [          # (sigma_bs, rho, vov) triples — no restriction issues here
    (0.30, -0.20, 0.50),
    (0.25,  0.10, 0.30),
    (0.20,  0.00, 0.40),
    (0.15, -0.50, 0.20),
]


class TestSubordBmLogp(unittest.TestCase):
    """
    Tests for log-price cumulants of subordinated Brownian motion models
    (Variance Gamma and NIG).

    Compares analytic logp_cum4 with numerical logp_cum4_numeric
    (Choudhury-Lucantoni on the CGF) over randomized texp.
    """

    TEXP_BASE = 1.0

    def test_vargamma_logp_cum4(self):
        """
        Analytic VarGamma cumulants (logp_cum4) match numerical ones
        (logp_cum4_numeric) over randomized texp.
        """
        rng = np.random.default_rng(seed=42)
        for params in _PARAMS_VG:
            m = pf.VarGammaCos(**params)
            for texp in self.TEXP_BASE * rng.uniform(0.25, 2, 5):
                c1_a, c2_a, c3_a, c4_a = m.logp_cum4(texp)
                c1_n, c2_n, c3_n, c4_n = m.logp_cum4_numeric(texp)

                np.testing.assert_allclose(c1_n, c1_a, rtol=1e-4, atol=1e-10)
                np.testing.assert_allclose(c2_n, c2_a, rtol=1e-4, atol=1e-10)
                np.testing.assert_allclose(c3_n, c3_a, rtol=1e-4, atol=1e-10)
                np.testing.assert_allclose(c4_n, c4_a, rtol=5e-4, atol=1e-10)

    def test_nig_logp_cum4(self):
        """
        Analytic NIG cumulants (logp_cum4) match numerical ones
        (logp_cum4_numeric) over randomized texp.
        """
        rng = np.random.default_rng(seed=42)
        for params in _PARAMS_NIG:
            m = pf.NigCos(**params)
            for texp in self.TEXP_BASE * rng.uniform(0.25, 2, 5):
                c1_a, c2_a, c3_a, c4_a = m.logp_cum4(texp)
                c1_n, c2_n, c3_n, c4_n = m.logp_cum4_numeric(texp)

                np.testing.assert_allclose(c1_n, c1_a, rtol=1e-4, atol=1e-10)
                np.testing.assert_allclose(c2_n, c2_a, rtol=1e-4, atol=1e-10)
                np.testing.assert_allclose(c3_n, c3_a, rtol=1e-4, atol=1e-10)
                np.testing.assert_allclose(c4_n, c4_a, rtol=5e-4, atol=1e-10)


class TestSubordBmSvParam(unittest.TestCase):
    """
    Tests for the (sigma, theta, nu) <-> (sigma_bs, rho, vov) conversion
    shared by VarGammaCos/VarGammaQuad and NigCos/ExpNigQuad.
    """

    def test_sv_params_roundtrip_from_sv(self):
        """from_sv_param followed by sv_params_dict() recovers the original SV parameters."""
        for sigma_bs, rho, vov in _SV_PARAMS:
            for cls in (pf.VarGammaCos, pf.NigCos):
                m = cls.from_sv_param(sigma_bs, rho, vov)
                d = m.sv_params_dict()
                np.testing.assert_allclose(d['sigma_bs'], sigma_bs, rtol=1e-12,
                                           err_msg=f"{cls.__name__}: sigma_bs roundtrip failed")
                np.testing.assert_allclose(d['rho'],      rho,      rtol=1e-12, atol=1e-14,
                                           err_msg=f"{cls.__name__}: rho roundtrip failed")
                np.testing.assert_allclose(d['vov'],      vov,      rtol=1e-12,
                                           err_msg=f"{cls.__name__}: vov roundtrip failed")

    def test_from_sv_param_constructor(self):
        """from_sv_param classmethod constructs an instance with correct orig params."""
        sv_params = [
            (0.30, -0.20, 0.50),
            (0.25,  0.10, 0.30),
            (0.20,  0.00, 0.40),
        ]
        for sigma_bs, rho, vov in sv_params:
            for cls in (pf.VarGammaCos, pf.NigCos):
                m = cls.from_sv_param(sigma_bs, rho, vov)
                sigma_e, theta_e, nu_e = cls.to_orig_param(sigma_bs, rho, vov)
                np.testing.assert_allclose(m.sigma, sigma_e, rtol=1e-14)
                np.testing.assert_allclose(m.theta, theta_e, rtol=1e-14)
                np.testing.assert_allclose(m.nu,    nu_e,    rtol=1e-14)
                assert isinstance(m, cls)



class TestSubordBmPriceConsistency(unittest.TestCase):
    """
    Cross-method price consistency: Quadrature vs FFT vs COS.

    FFT is the reference (dense grid, very accurate).
    Quad should be tight vs FFT; COS gets a wider tolerance.
    """

    SPOT  = 1.0
    TEXP  = 1.0
    # Strikes spanning OTM put through OTM call
    STRIKES = np.array([0.80, 0.90, 1.00, 1.10, 1.20])

    def _check(self, p_quad, p_ref, p_cos, label):
        np.testing.assert_allclose(
            p_quad, p_ref, rtol=5e-4,
            err_msg=f"{label}: Quad vs FFT mismatch",
        )
        np.testing.assert_allclose(
            p_cos, p_ref, rtol=5e-3,
            err_msg=f"{label}: COS vs FFT mismatch",
        )

    def test_vargamma_price_consistency(self):
        """VarGammaQuad prices agree with VarGammaFft and VarGammaCos."""
        for params in _PARAMS_VG:
            label = str(params)
            m_quad = pf.VarGammaQuad(**params)
            m_quad.n_quad = 15
            p_quad = m_quad.price(self.STRIKES, self.SPOT, self.TEXP)
            p_fft  = pf.VarGammaFft(**params).price(self.STRIKES, self.SPOT, self.TEXP)
            p_cos  = pf.VarGammaCos(**params).price(self.STRIKES, self.SPOT, self.TEXP)
            self._check(p_quad, p_fft, p_cos, f"VG {label}")

    def test_nig_price_consistency(self):
        """ExpNigQuad prices agree with ExpNigFft and NigCos."""
        for params in _PARAMS_NIG:
            label = str(params)
            m_quad = pf.ExpNigQuad(**params)
            m_quad.n_quad = 15
            p_quad = m_quad.price(self.STRIKES, self.SPOT, self.TEXP)
            p_fft  = pf.ExpNigFft(**params).price(self.STRIKES, self.SPOT, self.TEXP)
            p_cos  = pf.NigCos(**params).price(self.STRIKES, self.SPOT, self.TEXP)
            self._check(p_quad, p_fft, p_cos, f"NIG {label}")


class TestSubordBmBenchmark(unittest.TestCase):
    """
    Benchmark prices from Fang & Oosterlee (2008), Table 7.

    VG model, call option:
        S0=100, K=90, r=0.1, q=0, sigma=0.12, theta=-0.14, nu=0.2
        T=0.1 : 10.993703187...
        T=1.0 : 19.099354724...
    """

    PARAMS  = dict(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1, divr=0.0)
    SPOT    = 100.0
    STRIKE  = 90.0
    REF     = {0.1: 10.993703187, 1.0: 19.099354724}

    def test_vargamma_benchmark(self):
        """VarGammaFft, VarGammaCos, and VarGammaQuad match Fang&Oosterlee Table 7."""
        for texp, ref in self.REF.items():
            p_fft = pf.VarGammaFft(**self.PARAMS).price(self.STRIKE, self.SPOT, texp, cp=1)
            p_cos = pf.VarGammaCos(**self.PARAMS).price(self.STRIKE, self.SPOT, texp, cp=1)
            m_quad = pf.VarGammaQuad(**self.PARAMS)
            m_quad.n_quad = 15
            p_quad = m_quad.price(self.STRIKE, self.SPOT, texp, cp=1)

            np.testing.assert_allclose(p_fft,  ref, rtol=1e-4,
                                       err_msg=f"FFT  T={texp}: vs Fang&Oosterlee ref")
            np.testing.assert_allclose(p_cos,  ref, rtol=1e-4,
                                       err_msg=f"COS  T={texp}: vs Fang&Oosterlee ref")
            np.testing.assert_allclose(p_quad, ref, rtol=1e-5,
                                       err_msg=f"Quad T={texp}: vs Fang&Oosterlee ref")


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
