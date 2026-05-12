import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestSCASolver(unittest.TestCase):
    def setUp(self):
        self.cov = np.array(
            [
                [0.04, 0.01, 0.00],
                [0.01, 0.09, 0.02],
                [0.00, 0.02, 0.16],
            ]
        )

    def test_public_api(self):
        self.assertTrue(hasattr(pf, "SCASolver"))

    def test_fit_sca_satisfies_simplex_and_box_constraints(self):
        m = pf.SCASolver(cov_m=self.cov, w_max=0.5, tol=1e-8, max_iter=800)
        w = m.fit_sca()
        self.assertTrue(np.isclose(w.sum(), 1.0, atol=1e-8))
        self.assertTrue(np.all(w >= -1e-10))
        self.assertTrue(np.all(w <= 0.5 + 1e-8))

    def test_cov_alias_matches_cov_m(self):
        w_cov = pf.SCASolver(cov=self.cov, w_max=0.5).fit_sca()
        w_cov_m = pf.SCASolver(cov_m=self.cov, w_max=0.5).fit_sca()
        np.testing.assert_allclose(w_cov, w_cov_m, atol=1e-12)

    def test_sigma_and_cor_m_initialization(self):
        sigma = np.sqrt(np.diag(self.cov))
        cor_m = self.cov / np.outer(sigma, sigma)
        w = pf.SCASolver(sigma=sigma, cor_m=cor_m, w_max=0.5).fit_sca()
        self.assertTrue(np.isclose(w.sum(), 1.0, atol=1e-8))
        self.assertTrue(np.all(w <= 0.5 + 1e-8))

    def test_infeasible_w_max_raises(self):
        with self.assertRaises(ValueError):
            pf.SCASolver(cov_m=np.eye(4), w_max=0.2)

    def test_result_diagnostics_are_stored(self):
        m = pf.SCASolver(cov_m=self.cov, w_max=0.5, tol=1e-8, max_iter=800)
        m.fit_sca()
        self.assertIn("n_iter", m._result)
        self.assertIn("err", m._result)
        self.assertIn("objective", m._result)
        self.assertIn("gap", m._result)
        self.assertLess(m._result["err"], 1e-8)

    def test_non_binding_cap_matches_risk_parity_ccd(self):
        w_ccd = pf.RiskParity(cov_m=self.cov).fit_ccd(tol=1e-12)
        w_sca = pf.SCASolver(cov_m=self.cov, w_max=1.0, tol=1e-10, max_iter=1000).fit_sca()
        np.testing.assert_allclose(w_sca, w_ccd, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
