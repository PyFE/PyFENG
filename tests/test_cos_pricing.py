"""Tests for the COS option pricing classes in pyfeng/sv_cos.py."""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pyfeng as pf


# Heston paper parameters used across multiple test cases.
HESTON_SIGMA = 0.0175
HESTON_VOV   = 0.5751
HESTON_MR    = 1.5768
HESTON_THETA = 0.0398
HESTON_RHO   = -0.5711


def _make_heston():
    return pf.HestonCos(
        HESTON_SIGMA, vov=HESTON_VOV, mr=HESTON_MR,
        theta=HESTON_THETA, rho=HESTON_RHO,
    )


class TestBsmCos(unittest.TestCase):

    def test_analytic_matches_bsm(self):
        sigma, intr, divr, texp = 0.2, 0.05, 0.1, 1.2
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        m_analytic = pf.Bsm(sigma, intr=intr, divr=divr)
        m_cos = pf.BsmCos(sigma, intr=intr, divr=divr)
        ref = m_analytic.price(strikes, 100.0, texp)
        cos = m_cos.price(strikes, 100.0, texp)
        np.testing.assert_allclose(cos, ref, atol=1e-6)

    def test_convergence(self):
        m = pf.Bsm(0.2, intr=0.05)
        ref = m.price(100.0, 100.0, 1.0)
        # Check monotone decrease for N=16,32,64; N=64 already reaches machine eps.
        prev_err = None
        for n in [16, 32, 64]:
            mc = pf.BsmCos(0.2, intr=0.05)
            mc.n_cos = n
            err = abs(mc.price(100.0, 100.0, 1.0) - ref)
            if prev_err is not None:
                self.assertLess(err, prev_err, f"N={n}: error did not decrease")
            prev_err = err
        self.assertLess(err, 1e-8)

    def test_put_call_parity(self):
        sigma, intr, divr, texp = 0.2, 0.05, 0.1, 1.0
        strikes = np.array([80.0, 100.0, 120.0])
        m = pf.BsmCos(sigma, intr=intr, divr=divr)
        fwd = 100.0 * np.exp((intr - divr) * texp)
        df = np.exp(-intr * texp)
        calls = m.price(strikes, 100.0, texp, cp=1)
        puts = m.price(strikes, 100.0, texp, cp=-1)
        pcp = abs(calls - puts - df * (fwd - strikes))
        np.testing.assert_allclose(pcp, 0.0, atol=1e-8)

    def test_scalar_returns_float(self):
        m = pf.BsmCos(0.2)
        result = m.price(100.0, 100.0, 1.0)
        self.assertIsInstance(result, float)

    def test_vector_shape(self):
        m = pf.BsmCos(0.2)
        strikes = np.array([90.0, 100.0, 110.0])
        result = m.price(strikes, 100.0, 1.0)
        self.assertEqual(result.shape, (3,))

    def test_both_truncation_methods_agree(self):
        m_fo = pf.BsmCos(0.2, intr=0.05)
        m_fo.truncation_method = 'fang-oosterlee'
        m_jn = pf.BsmCos(0.2, intr=0.05)
        m_jn.truncation_method = 'junike'
        strikes = np.array([80.0, 100.0, 120.0])
        np.testing.assert_allclose(
            m_fo.price(strikes, 100.0, 1.0),
            m_jn.price(strikes, 100.0, 1.0),
            atol=1e-8,
        )

    def test_edge_no_nan(self):
        m = pf.BsmCos(0.2, intr=0.05)
        for k in [50.0, 150.0]:
            for t in [0.01, 10.0]:
                p = m.price(k, 100.0, t)
                self.assertTrue(np.isfinite(p), f"NaN/Inf at K={k}, T={t}")
                self.assertGreaterEqual(p, 0.0)


class TestHestonCos(unittest.TestCase):

    def test_paper_benchmark_t1(self):
        m = _make_heston()
        price = m.price(100.0, 100.0, 1.0)
        np.testing.assert_allclose(price, 5.785155435, atol=1e-5)

    def test_cross_validate_heston_fft(self):
        m_cos = _make_heston()
        m_fft = pf.HestonFft(
            HESTON_SIGMA, vov=HESTON_VOV, mr=HESTON_MR,
            theta=HESTON_THETA, rho=HESTON_RHO,
        )
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        np.testing.assert_allclose(
            m_cos.price(strikes, 100.0, 1.0),
            m_fft.price(strikes, 100.0, 1.0),
            atol=1e-5,
        )

    def test_multiple_maturities(self):
        m_cos = _make_heston()
        m_fft = pf.HestonFft(
            HESTON_SIGMA, vov=HESTON_VOV, mr=HESTON_MR,
            theta=HESTON_THETA, rho=HESTON_RHO,
        )
        # T=0.1 needs larger tolerance: short maturities require more COS terms.
        tol = {0.1: 2e-3, 0.5: 1e-4, 1.0: 1e-4, 5.0: 1e-4}
        for texp in [0.1, 0.5, 1.0, 5.0]:
            p_cos = m_cos.price(100.0, 100.0, texp)
            p_fft = m_fft.price(100.0, 100.0, texp)
            np.testing.assert_allclose(p_cos, p_fft, atol=tol[texp],
                                       err_msg=f"T={texp}")

    def test_adaptive_l_formula(self):
        m = _make_heston()
        self.assertAlmostEqual(m._resolve_L(1.0), 10.0)
        self.assertAlmostEqual(m._resolve_L(10.0), 32.0)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            pf.HestonCos(-0.01, vov=0.5, mr=1.0, theta=0.04, rho=0.0)
        with self.assertRaises(ValueError):
            pf.HestonCos(0.04, vov=-0.5, mr=1.0, theta=0.04, rho=0.0)
        with self.assertRaises(ValueError):
            pf.HestonCos(0.04, vov=0.5, mr=-1.0, theta=0.04, rho=0.0)
        with self.assertRaises(ValueError):
            pf.HestonCos(0.04, vov=0.5, mr=1.0, theta=-0.04, rho=0.0)
        with self.assertRaises(ValueError):
            pf.HestonCos(0.04, vov=0.5, mr=1.0, theta=0.04, rho=1.0)
        with self.assertRaises(ValueError):
            pf.HestonCos(0.04, vov=0.5, mr=1.0, theta=0.04, rho=-1.0)

    def test_texp_zero_raises(self):
        m = _make_heston()
        with self.assertRaises(ValueError):
            m.price(100.0, 100.0, 0.0)

    def test_put_call_parity(self):
        m = _make_heston()
        strikes = np.array([90.0, 100.0, 110.0])
        fwd = 100.0
        calls = m.price(strikes, fwd, 1.0, cp=1)
        puts = m.price(strikes, fwd, 1.0, cp=-1)
        pcp = abs(calls - puts - (fwd - strikes))  # intr=0 so df=1
        np.testing.assert_allclose(pcp, 0.0, atol=1e-4)

    def test_vector_scalar_consistency(self):
        m = _make_heston()
        strikes = np.array([90.0, 100.0, 110.0])
        vec = m.price(strikes, 100.0, 1.0)
        for i, k in enumerate(strikes):
            scalar = m.price(k, 100.0, 1.0)
            self.assertLess(abs(vec[i] - scalar), 1e-12,
                            f"Mismatch at K={k}")

    def test_fo_truncation_per_strike(self):
        m = _make_heston()
        m.truncation_method = 'fang-oosterlee'
        a, b = m._integration_range(np.array([80.0, 100.0, 120.0]), 100.0, 1.0)
        self.assertEqual(np.asarray(a).shape, (3,))
        self.assertEqual(np.asarray(b).shape, (3,))

    def test_junike_truncation_global(self):
        m = _make_heston()
        m.truncation_method = 'junike'
        a, b = m._integration_range(np.array([80.0, 100.0, 120.0]), 100.0, 1.0)
        self.assertEqual(np.ndim(a), 0)
        self.assertEqual(np.ndim(b), 0)

    def test_mgf_martingale(self):
        m = _make_heston()
        for texp in [0.5, 1.0, 5.0]:
            val = m.mgf_logprice(np.float64(1.0), texp)
            np.testing.assert_allclose(float(np.real(val)), 1.0, atol=1e-12,
                                       err_msg=f"T={texp}")


class TestVarGammaCos(unittest.TestCase):

    def _make_vg(self):
        return pf.VarGammaCos(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1)

    def test_cumulants_analytic_vs_fd(self):
        m = self._make_vg()
        texp = 1.0
        c1_a, c2_a, _, _ = m._cumulants(texp)
        # Finite-difference cross-check; wrap in array to satisfy VarGammaFft's
        # mgf_logprice which uses np.exp(..., out=rv) and requires an ndarray.
        eps = 1e-4
        lm = lambda v: np.log(m.mgf_logprice(np.atleast_1d(np.float64(v)), texp)).real.item()
        c1_fd = (lm(eps) - lm(-eps)) / (2 * eps)
        c2_fd = (lm(eps) + lm(-eps) - 2 * lm(0.0)) / eps**2
        np.testing.assert_allclose(c1_a, c1_fd, atol=1e-5)
        np.testing.assert_allclose(c2_a, c2_fd, atol=1e-5)

    def test_cross_vargamma_fft(self):
        m_cos = self._make_vg()
        m_fft = pf.VarGammaFft(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1)
        strikes = np.array([90.0, 100.0, 110.0])
        np.testing.assert_allclose(
            m_cos.price(strikes, 100.0, 1.0),
            m_fft.price(strikes, 100.0, 1.0),
            atol=5e-4,
        )

    def test_put_call_parity(self):
        m = self._make_vg()
        strikes = np.array([90.0, 100.0, 110.0])
        intr, texp = 0.1, 1.0
        fwd = 100.0 * np.exp(intr * texp)
        df = np.exp(-intr * texp)
        calls = m.price(strikes, 100.0, texp, cp=1)
        puts = m.price(strikes, 100.0, texp, cp=-1)
        np.testing.assert_allclose(calls - puts, df * (fwd - strikes), atol=1e-4)

    def test_both_truncation_methods(self):
        m_fo = self._make_vg()
        m_jn = self._make_vg()
        m_jn.truncation_method = 'junike'
        strikes = np.array([90.0, 100.0, 110.0])
        np.testing.assert_allclose(
            m_fo.price(strikes, 100.0, 1.0),
            m_jn.price(strikes, 100.0, 1.0),
            atol=5e-4,
        )

    def test_non_negative_calls(self):
        m = self._make_vg()
        strikes = np.array([80.0, 100.0, 120.0])
        prices = m.price(strikes, 100.0, 1.0)
        self.assertTrue(np.all(prices >= 0.0))


class TestCgmyCos(unittest.TestCase):

    def _make_cgmy(self, Y=0.5):
        return pf.CgmyCos(C=1.0, G=5.0, M=5.0, Y=Y)

    def test_put_call_parity(self):
        m = self._make_cgmy()
        strikes = np.array([90.0, 100.0, 110.0])
        calls = m.price(strikes, 100.0, 1.0, cp=1)
        puts = m.price(strikes, 100.0, 1.0, cp=-1)
        pcp = abs(calls - puts - (100.0 - strikes))  # intr=0
        np.testing.assert_allclose(pcp, 0.0, atol=1e-3)

    def test_y_values_no_blowup(self):
        # Avoid Y in {0, 1, 2, ...}: Gamma(-Y) has poles there.
        for Y in [0.5, 0.7, 1.5]:
            m = pf.CgmyCos(C=1.0, G=5.0, M=5.0, Y=Y)
            p = m.price(100.0, 100.0, 1.0)
            self.assertTrue(np.isfinite(p), f"Y={Y}: not finite")
            self.assertGreater(p, 0.0, f"Y={Y}: not positive")

    def test_cross_cgmy_fft(self):
        m_cos = self._make_cgmy(Y=0.5)
        m_fft = pf.CgmyFft(C=1.0, G=5.0, M=5.0, Y=0.5)
        strikes = np.array([90.0, 100.0, 110.0])
        np.testing.assert_allclose(
            m_cos.price(strikes, 100.0, 1.0),
            m_fft.price(strikes, 100.0, 1.0),
            atol=5e-3,
        )

    def test_y_near_2_fo_completes(self):
        m = pf.CgmyCos(C=1.0, G=5.0, M=5.0, Y=1.98)
        p = m.price(100.0, 100.0, 1.0)
        self.assertTrue(np.isfinite(p))

    def test_y_near_2_junike_stable(self):
        # For Y near 2, moments diverge so cumulant-based Junike range is huge;
        # verify it completes (no exception/NaN/Inf) rather than checking accuracy.
        m = pf.CgmyCos(C=1.0, G=5.0, M=5.0, Y=1.98)
        m.truncation_method = 'junike'
        p = m.price(100.0, 100.0, 1.0)
        self.assertTrue(np.isfinite(p))


class TestTruncationSwitching(unittest.TestCase):

    def test_unknown_method_raises(self):
        m = pf.BsmCos(0.2)
        m.truncation_method = 'bogus'
        with self.assertRaises(ValueError):
            m.price(100.0, 100.0, 1.0)

    def test_bsm_both_methods_agree(self):
        m_fo = pf.BsmCos(0.2, intr=0.05)
        m_jn = pf.BsmCos(0.2, intr=0.05)
        m_jn.truncation_method = 'junike'
        ref = pf.Bsm(0.2, intr=0.05).price(100.0, 100.0, 1.0)
        np.testing.assert_allclose(m_fo.price(100.0, 100.0, 1.0), ref, atol=1e-8)
        np.testing.assert_allclose(m_jn.price(100.0, 100.0, 1.0), ref, atol=1e-8)

    def test_heston_both_methods_agree(self):
        m_fo = _make_heston()
        m_jn = _make_heston()
        m_jn.truncation_method = 'junike'
        # Junike uses a wider global interval; more COS terms are needed to
        # maintain the same frequency resolution.
        m_jn.n_cos = 512
        strikes = np.array([90.0, 100.0, 110.0])
        np.testing.assert_allclose(
            m_fo.price(strikes, 100.0, 1.0),
            m_jn.price(strikes, 100.0, 1.0),
            atol=1e-4,
        )


class TestLeFloch(unittest.TestCase):
    """Le Floc'h (2020) put-first improved COS formula."""

    def test_lefloch_agrees_with_fo_bsm(self):
        """Standard-strike BSM: Le Floc'h and F&O agree to 1e-6."""
        m_fo = pf.BsmCos(sigma=0.2, intr=0.05)
        m_lf = pf.BsmCos(sigma=0.2, intr=0.05)
        m_lf.pricing_formula = 'lefloch'
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        np.testing.assert_allclose(
            m_fo.price(strikes, 100.0, 1.0),
            m_lf.price(strikes, 100.0, 1.0),
            atol=1e-6,
        )

    def test_lefloch_matches_analytic_bsm(self):
        """Le Floc'h should match the exact BSM formula."""
        sigma, intr = 0.2, 0.05
        m_bsm = pf.Bsm(sigma, intr=intr)
        m_lf = pf.BsmCos(sigma, intr=intr)
        m_lf.pricing_formula = 'lefloch'
        strikes = np.array([80.0, 100.0, 120.0])
        np.testing.assert_allclose(
            m_lf.price(strikes, 100.0, 1.0),
            m_bsm.price(strikes, 100.0, 1.0),
            atol=1e-6,
        )

    def test_lefloch_put_call_parity_exact(self):
        """PCP is exact by construction: C - P = df*(F-K) to machine epsilon."""
        m = pf.BsmCos(sigma=0.2, intr=0.05, divr=0.02)
        m.pricing_formula = 'lefloch'
        strikes = np.array([80.0, 100.0, 120.0])
        fwd = 100.0 * np.exp((0.05 - 0.02) * 1.0)
        df = np.exp(-0.05 * 1.0)
        calls = m.price(strikes, 100.0, 1.0, cp=1)
        puts = m.price(strikes, 100.0, 1.0, cp=-1)
        # PCP holds to floating-point precision: call and put share the same put
        # computation; call = put + df*(F-K) algebraically.
        np.testing.assert_allclose(calls - puts, df * (fwd - strikes), atol=1e-15)

    def test_lefloch_boundary_below_a(self):
        """z = log(K/F) < a: put should be 0 (density entirely above K)."""
        m = pf.BsmCos(sigma=0.2)
        m.pricing_formula = 'lefloch'
        # K=1, F=100 → z = log(0.01) ≈ -4.6, well below a ≈ -2.4
        put = m.price(1.0, 100.0, 1.0, cp=-1)
        self.assertAlmostEqual(put, 0.0, places=10)

    def test_lefloch_boundary_above_b(self):
        """z = log(K/F) > b: put = intrinsic df*(K-F)."""
        m = pf.BsmCos(sigma=0.2)
        m.pricing_formula = 'lefloch'
        # K=99999, F=100 → z = log(999.99) ≈ 6.9 >> b ≈ 2.4
        K, F = 99999.0, 100.0
        put = m.price(K, F, 1.0, cp=-1)
        self.assertAlmostEqual(put, K - F, places=4)

    def test_lefloch_wide_strike_range_non_negative(self):
        """
        Le Floc'h applies explicit boundary conditions for z = log(K/F) outside
        [a, b]: puts and calls are always non-negative and calls agree with the
        BSM analytic formula for strikes within the density support.

        Note on accuracy: PyFENG's F&O range uses sqrt(|c2| + sqrt(|c4|)) which
        is wider than the paper's sqrt(|c2|) range.  So for typical parameters
        z << b and both methods are accurate.  Le Floc'h's structural advantage
        shows in correct boundary handling and the exact PCP identity.
        """
        sigma, intr, texp = 0.3, 0.05, 1.0
        ref = pf.Bsm(sigma, intr=intr)
        m_lf = pf.BsmCos(sigma, intr=intr)
        m_lf.pricing_formula = 'lefloch'

        # Wide range: from deep ITM to deep OTM (including beyond-b strikes)
        strikes = np.array([40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0])
        lf_calls = m_lf.price(strikes, 100.0, texp, cp=1)
        lf_puts  = m_lf.price(strikes, 100.0, texp, cp=-1)

        # Non-negativity guaranteed by construction
        self.assertTrue(np.all(lf_calls >= -1e-10), "Le Floc'h calls non-negative")
        self.assertTrue(np.all(lf_puts  >= -1e-10), "Le Floc'h puts non-negative")

        # For strikes well within [a, b] (rough threshold: K in [40, 200] here),
        # Le Floc'h should match the BSM analytic formula closely.
        ref_calls = ref.price(strikes, 100.0, texp, cp=1)
        # Generous tolerance: the widest-range strikes may use boundary conditions
        np.testing.assert_allclose(lf_calls, ref_calls, atol=1e-5,
                                   err_msg="Le Floc'h calls should match BSM analytic")

    def test_lefloch_heston_agrees_standard_strikes(self):
        """Le Floc'h and F&O agree for standard ATM Heston strikes (T=1).

        F&O mode uses per-strike truncation centered on each strike; Le Floc'h
        uses the global cumulant-based range.  For ATM/near-ATM strikes both
        intervals cover the density well, so prices should match.
        """
        m_fo = _make_heston()
        m_lf = _make_heston()
        m_lf.pricing_formula = 'lefloch'
        # Increase n_cos for Le Floc'h: global range is wider than per-strike
        # range, so more COS terms are needed for the same frequency resolution.
        m_lf.n_cos = 512
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        np.testing.assert_allclose(
            m_fo.price(strikes, 100.0, 1.0),
            m_lf.price(strikes, 100.0, 1.0),
            atol=1e-3,
        )

    def test_lefloch_scalar_output(self):
        """Scalar inputs → float output."""
        m = pf.BsmCos(0.2)
        m.pricing_formula = 'lefloch'
        result = m.price(100.0, 100.0, 1.0)
        self.assertIsInstance(result, float)

    def test_lefloch_vector_shape(self):
        """Array inputs → same-shape array output."""
        m = pf.BsmCos(0.2)
        m.pricing_formula = 'lefloch'
        strikes = np.array([90.0, 100.0, 110.0])
        result = m.price(strikes, 100.0, 1.0)
        self.assertEqual(result.shape, (3,))

    def test_lefloch_unknown_formula_raises(self):
        """Unknown pricing_formula raises ValueError."""
        m = pf.BsmCos(0.2)
        m.pricing_formula = 'bogus-formula'
        with self.assertRaises(ValueError):
            m.price(100.0, 100.0, 1.0)

    def test_lefloch_vg_agrees_fo(self):
        """VarGammaCos: Le Floc'h and F&O agree for standard strikes."""
        m_fo = pf.VarGammaCos(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1)
        m_lf = pf.VarGammaCos(sigma=0.12, theta=-0.14, nu=0.2, intr=0.1)
        m_lf.pricing_formula = 'lefloch'
        strikes = np.array([90.0, 100.0, 110.0])
        np.testing.assert_allclose(
            m_fo.price(strikes, 100.0, 1.0),
            m_lf.price(strikes, 100.0, 1.0),
            atol=5e-4,
        )


class TestCrossModel(unittest.TestCase):

    def test_bsm_cos_vs_bsm_analytic(self):
        sigma, intr = 0.3, 0.05
        strikes = np.arange(80.0, 121.0, 10.0)
        ref = pf.Bsm(sigma, intr=intr).price(strikes, 100.0, 1.0)
        cos = pf.BsmCos(sigma, intr=intr).price(strikes, 100.0, 1.0)
        np.testing.assert_allclose(cos, ref, atol=1e-6)

    def test_heston_smile(self):
        m_cos = _make_heston()
        m_fft = pf.HestonFft(
            HESTON_SIGMA, vov=HESTON_VOV, mr=HESTON_MR,
            theta=HESTON_THETA, rho=HESTON_RHO,
        )
        strikes = np.arange(80.0, 121.0, 5.0)
        np.testing.assert_allclose(
            m_cos.price(strikes, 100.0, 1.0),
            m_fft.price(strikes, 100.0, 1.0),
            atol=1e-4,
        )

    def test_vg_near_bsm_limit(self):
        """VG with tiny vov (nu → 0) should approach BSM."""
        sigma = 0.2
        m_vg = pf.VarGammaCos(sigma=sigma, theta=0.0, nu=0.001)
        m_bsm = pf.Bsm(sigma)
        strikes = np.array([90.0, 100.0, 110.0])
        np.testing.assert_allclose(
            m_vg.price(strikes, 100.0, 1.0),
            m_bsm.price(strikes, 100.0, 1.0),
            atol=1e-2,
        )


if __name__ == '__main__':
    unittest.main()
