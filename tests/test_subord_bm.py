import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestSubordBmLogp(unittest.TestCase):
    """
    Tests for log-price cumulants of subordinated Brownian motion models
    (Variance Gamma and NIG).

    Compares analytic logp_cum4 with numerical logp_cum4_numeric
    (Choudhury-Lucantoni on the CGF) over randomized texp.
    """

    # (sigma, nu, theta)
    PARAMS_VG = [
        dict(sigma=0.12, nu=0.2,  theta=-0.14),
        dict(sigma=0.20, nu=0.5,  theta=-0.10),
        dict(sigma=0.15, nu=0.1,  theta=0.05),
    ]
    PARAMS_NIG = [
        dict(sigma=0.12, nu=0.2,  theta=-0.14),
        dict(sigma=0.20, nu=0.5,  theta=-0.10),
        dict(sigma=0.15, nu=0.1,  theta=0.05),
    ]
    TEXP_BASE = 1.0

    def test_vargamma_logp_cum4(self):
        """
        Analytic VarGamma cumulants (logp_cum4) match numerical ones
        (logp_cum4_numeric) over randomized texp.
        """
        rng = np.random.default_rng(seed=42)
        for params in self.PARAMS_VG:
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
        for params in self.PARAMS_NIG:
            m = pf.NigCos(**params)
            for texp in self.TEXP_BASE * rng.uniform(0.25, 2, 5):
                c1_a, c2_a, c3_a, c4_a = m.logp_cum4(texp)
                c1_n, c2_n, c3_n, c4_n = m.logp_cum4_numeric(texp)

                np.testing.assert_allclose(c1_n, c1_a, rtol=1e-4, atol=1e-10)
                np.testing.assert_allclose(c2_n, c2_a, rtol=1e-4, atol=1e-10)
                np.testing.assert_allclose(c3_n, c3_a, rtol=1e-4, atol=1e-10)
                np.testing.assert_allclose(c4_n, c4_a, rtol=5e-4, atol=1e-10)


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
