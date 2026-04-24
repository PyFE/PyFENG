import unittest
import warnings
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.getcwd())
from pyfeng.mgf2mom import Mgf2Mom


class TestMgf2Mom(unittest.TestCase):
    """
    Tests based on numerical examples in:
    Choudhury GL, Lucantoni DM (1996) Numerical Computation of the Moments of a
    Probability Distribution from its Transform. Operations Research 44:368-381.
    Algorithm 3 is expected to match exact moments to 9 or more significant places.
    """

    def test_exponential_large_mean(self):
        """
        Table I: Exponential distribution with mean = 10.
        M(z) = (1 - 10z)^{-1}, exact moments: mu_n = n! * 10^n.
        """
        def mgf(z):
            return 1.0 / (1.0 - 10.0 * z)

        m = Mgf2Mom(mgf)
        mu = m.moments(6)

        exact = np.array([math.factorial(n) * 10**n for n in range(1, 7)], dtype=float)
        np.testing.assert_allclose(mu, exact, rtol=1e-6,
                                   err_msg="Exponential(mean=10): moments mismatch")

    def test_exponential_small_mean(self):
        """
        Table II: Exponential distribution with mean = 0.1.
        M(z) = (1 - 0.1z)^{-1}, exact moments: mu_n = n! * 0.1^n.
        """
        def mgf(z):
            return 1.0 / (1.0 - 0.1 * z)

        m = Mgf2Mom(mgf)
        mu = m.moments(9)

        exact = np.array([math.factorial(n) * 0.1**n for n in range(1, 10)], dtype=float)
        np.testing.assert_allclose(mu, exact, rtol=1e-6,
                                   err_msg="Exponential(mean=0.1): moments mismatch")

    def test_uniform(self):
        """
        Table III: Uniform distribution on [0, 1].
        M(z) = (exp(z) - 1) / z, exact moments: mu_n = 1 / (n + 1).
        """
        def mgf(z):
            return np.expm1(z) / z

        m = Mgf2Mom(mgf)
        mu = m.moments(10)

        exact = np.array([1.0 / (n + 1) for n in range(1, 11)], dtype=float)
        np.testing.assert_allclose(mu, exact, rtol=1e-6,
                                   err_msg="Uniform[0,1]: moments mismatch")

    def test_erlang(self):
        """
        Table IV: Erlang distribution with mean = 1, SCV = 1/40 (Erlang-40, rate 40).
        M(z) = (1 - z/40)^{-40}, exact moments: mu_n = prod_{i=1}^{n} (1 + (i-1)/40).
        """
        def mgf(z):
            return (1.0 - z / 40.0) ** (-40)

        m = Mgf2Mom(mgf)
        mu = m.moments(6)

        exact = np.array(
            [np.prod([1.0 + (i - 1) / 40.0 for i in range(1, n + 1)]) for n in range(1, 7)],
            dtype=float
        )
        np.testing.assert_allclose(mu, exact, rtol=1e-6,
                                   err_msg="Erlang(mean=1, SCV=1/40): moments mismatch")

    def test_n_less_than_2_warning(self):
        """
        moments(n) with n < 2 should emit a UserWarning and return 2 moments.
        """
        def mgf(z):
            return 1.0 / (1.0 - z)

        m = Mgf2Mom(mgf)
        with self.assertWarns(UserWarning):
            mu = m.moments(1)
        self.assertEqual(len(mu), 2)


if __name__ == "__main__":
    unittest.main()
