import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestCirCum4Numeric(unittest.TestCase):
    """
    Tests for CirModel.cum4_numeric.

    cum4_numeric returns the first four cumulants of V_T given V_0 = v0 by
    numerically differentiating the noncentral chi-squared MGF of V_T.
    The first two cumulants (mean and variance) must match mv exactly.
    """

    # (sigma, mr, theta, texp, v0)
    PARAMS = [
        (0.3, 2.0, 0.04, 1.0,  0.04),
        (0.5, 1.0, 0.09, 0.5,  0.06),
        (0.2, 4.0, 0.02, 2.0,  0.03),
        (0.4, 0.5, 0.06, 0.25, 0.08),
        (0.6, 3.0, 0.05, 0.1,  0.05),   # short texp (Taylor regime)
    ]

    def test_cum4_vs_mv(self):
        """
        κ₁ and κ₂ from cum4_numeric must match mv (mean and variance of V_T).
        """
        for sigma, mr, theta, texp, v0 in self.PARAMS:
            cir = pf.CirModel(sigma, mr, theta)
            k1, k2, _, _ = cir.cum4_numeric(texp, v0)
            mean, var = cir.mv(texp, v0)

            np.testing.assert_allclose(
                k1, mean, rtol=1e-5,
                err_msg=f"κ₁ mismatch: sigma={sigma}, mr={mr}, theta={theta}, texp={texp}, v0={v0}"
            )
            np.testing.assert_allclose(
                k2, var, rtol=1e-5,
                err_msg=f"κ₂ mismatch: sigma={sigma}, mr={mr}, theta={theta}, texp={texp}, v0={v0}"
            )


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
