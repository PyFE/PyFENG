import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestRiskParitySCA(unittest.TestCase):
    def setUp(self):
        self.cov = np.array([
            [ 94.868, 33.750, 12.325, -1.178, 8.778 ],
            [ 33.750, 445.642, 98.955, -7.901, 84.954 ],
            [ 12.325, 98.955, 117.265, 0.503, 45.184 ],
            [ -1.178, -7.901, 0.503, 5.460, 1.057 ],
            [ 8.778, 84.954, 45.184, 1.057, 34.126 ]
        ]) / 10000

    def test_public_api(self):
        self.assertTrue(hasattr(pf, "RiskParitySCA"))

    def test_fit_sca_satisfies_simplex_and_box_constraints(self):
        m = pf.RiskParitySCA(cov_m=self.cov, w_max=0.65)
        m.max_iter = 800
        w = m.fit_sca(tol=1e-8)
        self.assertTrue(np.isclose(w.sum(), 1.0, atol=1e-8))
        self.assertTrue(np.all(w >= -1e-10))
        self.assertTrue(np.all(w <= 0.65 + 1e-8))

    def test_tol_and_max_iter_are_class_data(self):
        self.assertEqual(pf.RiskParitySCA.tol, 1e-6)
        self.assertEqual(pf.RiskParitySCA.max_iter, 200)
        m = pf.RiskParitySCA(cov_m=self.cov, w_max=0.65)
        self.assertEqual(m.tol, 1e-6)
        self.assertEqual(m.max_iter, 200)

    def test_sigma_and_cor_m_initialization(self):
        sigma = np.sqrt(np.diag(self.cov))
        cor_m = self.cov / np.outer(sigma, sigma)
        w = pf.RiskParitySCA(sigma=sigma, cor_m=cor_m, w_max=0.65).fit_sca()
        self.assertTrue(np.isclose(w.sum(), 1.0, atol=1e-8))
        self.assertTrue(np.all(w <= 0.65 + 1e-8))

    def test_infeasible_w_max_raises(self):
        with self.assertRaises(ValueError):
            pf.RiskParitySCA(cov_m=np.eye(4), w_max=0.2)

    def test_result_diagnostics_are_stored(self):
        m = pf.RiskParitySCA(cov_m=self.cov, w_max=0.65)
        m.max_iter = 800
        m.fit_sca(tol=1e-8)
        self.assertIn("n_iter", m._result)
        self.assertIn("err", m._result)
        self.assertIn("objective", m._result)
        self.assertIn("gap", m._result)
        self.assertLess(m._result["err"], 1e-8)

    def test_non_binding_cap_matches_risk_parity_ccd(self):
        w_ccd = pf.RiskParity(cov_m=self.cov).fit_ccd(tol=1e-12)
        m = pf.RiskParitySCA(cov_m=self.cov, w_max=1.0)
        m.max_iter = 1000
        w_sca = m.fit_sca(tol=1e-10)
        np.testing.assert_allclose(w_sca, w_ccd, atol=1e-5)

    def test_tight_cap_objective(self):
        # paper's 5-asset example, w_max=0.4 forces redistribution from unconstrained w[3]=0.613
        m = pf.RiskParitySCA(cov_m=self.cov, w_max=0.4)
        w = m.fit_sca()
        self.assertLess(m._result['objective'], 1.55e-7)


if __name__ == "__main__":
    unittest.main()
