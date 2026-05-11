import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
from pyfeng.disthelper import DistGamma, DistInvGauss, DistGig, DistGh, DistNig


class TestDistHelperScipyStats(unittest.TestCase):
    """
    Compare mvsk() against scipy_stats().stats('mvsk') for each distribution.

    mvsk() convention:
        DistGamma / DistInvGauss / DistGig  →  (mean, var_scaled, skewness, excess_kurtosis)
        DistGh / DistNig                    →  (mean, variance,   skewness, excess_kurtosis)

    scipy stats('mvsk') always returns (mean, variance, skewness, excess_kurtosis).
    """

    rtol = 1e-5

    # ------------------------------------------------------------------ Gamma
    GAMMA_PARAMS = [
        dict(shape=3.0, rate=2.0),
        dict(shape=0.5, rate=1.0),
        dict(shape=10.0, rate=0.5),
        dict(shape=1.0, rate=3.0),   # exponential
    ]

    def test_gamma(self):
        for kw in self.GAMMA_PARAMS:
            d = DistGamma(**kw)
            mean, var_scaled, skew, exkurt = d.mvsk()
            var = var_scaled * mean**2
            sm, sv, ss, sk = d.scipy_stats().stats('mvsk')
            np.testing.assert_allclose(mean,   sm, rtol=self.rtol, err_msg=f"Gamma mean {kw}")
            np.testing.assert_allclose(var,    sv, rtol=self.rtol, err_msg=f"Gamma var {kw}")
            np.testing.assert_allclose(skew,   ss, rtol=self.rtol, err_msg=f"Gamma skew {kw}")
            np.testing.assert_allclose(exkurt, sk, rtol=self.rtol, err_msg=f"Gamma exkurt {kw}")

    # --------------------------------------------------------------- InvGauss
    IG_PARAMS = [
        dict(mu=1.5, lam=2.0),
        dict(mu=0.5, lam=0.5),
        dict(mu=3.0, lam=5.0),
        dict(mu=1.0, lam=1.0),
    ]

    def test_invgauss(self):
        for kw in self.IG_PARAMS:
            d = DistInvGauss(**kw)
            mean, var_scaled, skew, exkurt = d.mvsk()
            var = var_scaled * mean**2
            sm, sv, ss, sk = d.scipy_stats().stats('mvsk')
            np.testing.assert_allclose(mean,   sm, rtol=self.rtol, err_msg=f"IG mean {kw}")
            np.testing.assert_allclose(var,    sv, rtol=self.rtol, err_msg=f"IG var {kw}")
            np.testing.assert_allclose(skew,   ss, rtol=self.rtol, err_msg=f"IG skew {kw}")
            np.testing.assert_allclose(exkurt, sk, rtol=self.rtol, err_msg=f"IG exkurt {kw}")

    # -------------------------------------------------------------------- GIG
    GIG_PARAMS = [
        dict(gamma=1.2, delta=0.8, p= 0.3),
        dict(gamma=0.5, delta=1.5, p=-0.5),   # IG special case
        dict(gamma=2.0, delta=1.0, p= 1.0),
        dict(gamma=1.0, delta=1.0, p=-1.5),
    ]

    def test_gig(self):
        for kw in self.GIG_PARAMS:
            d = DistGig(**kw)
            mean, var_scaled, skew, exkurt = d.mvsk()
            var = var_scaled * mean**2
            sm, sv, ss, sk = d.scipy_stats().stats('mvsk')
            np.testing.assert_allclose(mean,   sm, rtol=self.rtol, err_msg=f"GIG mean {kw}")
            np.testing.assert_allclose(var,    sv, rtol=self.rtol, err_msg=f"GIG var {kw}")
            np.testing.assert_allclose(skew,   ss, rtol=self.rtol, err_msg=f"GIG skew {kw}")
            np.testing.assert_allclose(exkurt, sk, rtol=self.rtol, err_msg=f"GIG exkurt {kw}")

    # --------------------------------------------------------------------- GH
    GH_PARAMS = [
        dict(mu=0.0,  beta=0.0,  gamma=1.0, delta=1.0, p=-0.5),  # symmetric NIG
        dict(mu=0.1,  beta=0.3,  gamma=1.5, delta=0.5, p=-0.5),
        dict(mu=0.0,  beta=0.2,  gamma=2.0, delta=1.0, p= 1.0),  # hyperbolic
        dict(mu=-0.1, beta=-0.1, gamma=1.0, delta=0.8, p= 0.5),
    ]

    def test_gh(self):
        for kw in self.GH_PARAMS:
            d = DistGh(**kw)
            mean, var, skew, exkurt = d.mvsk()
            sm, sv, ss, sk = d.scipy_stats().stats('mvsk')
            np.testing.assert_allclose(mean,   sm, rtol=self.rtol, err_msg=f"GH mean {kw}")
            np.testing.assert_allclose(var,    sv, rtol=self.rtol, err_msg=f"GH var {kw}")
            np.testing.assert_allclose(skew,   ss, rtol=1e-3,      err_msg=f"GH skew {kw}")
            np.testing.assert_allclose(exkurt, sk, rtol=1e-3,      err_msg=f"GH exkurt {kw}")

    def test_gh_quad_vs_analytic(self):
        """mvsk_quad and mvsk (analytic) must agree closely for GH."""
        for kw in self.GH_PARAMS:
            d = DistGh(**kw)
            analytic = d.mvsk()
            quad     = d.mvsk_quad()
            for i, label in enumerate(("mean", "var", "skew", "exkurt")):
                np.testing.assert_allclose(
                    analytic[i], quad[i], rtol=1e-3,
                    err_msg=f"GH {label} analytic vs quad {kw}"
                )

    # -------------------------------------------------------------------- NIG
    NIG_PARAMS = [
        dict(mu=0.0,  beta=0.0,  gamma=1.0, delta=1.0),
        dict(mu=0.2,  beta=0.4,  gamma=2.0, delta=0.5),
        dict(mu=-0.1, beta=-0.2, gamma=0.8, delta=1.2),
    ]

    def test_nig(self):
        for kw in self.NIG_PARAMS:
            d = DistNig(**kw)
            mean, var, skew, exkurt = d.mvsk()
            sm, sv, ss, sk = d.scipy_stats().stats('mvsk')
            np.testing.assert_allclose(mean,   sm, rtol=self.rtol, err_msg=f"NIG mean {kw}")
            np.testing.assert_allclose(var,    sv, rtol=self.rtol, err_msg=f"NIG var {kw}")
            np.testing.assert_allclose(skew,   ss, rtol=self.rtol, err_msg=f"NIG skew {kw}")
            np.testing.assert_allclose(exkurt, sk, rtol=self.rtol, err_msg=f"NIG exkurt {kw}")

    def test_nig_quad_vs_analytic(self):
        """mvsk_quad (inherited) and mvsk (analytic override) must agree closely for NIG."""
        for kw in self.NIG_PARAMS:
            d = DistNig(**kw)
            analytic = d.mvsk()
            quad     = d.mvsk_quad()
            for i, label in enumerate(("mean", "var", "skew", "exkurt")):
                np.testing.assert_allclose(
                    analytic[i], quad[i], rtol=1e-3,
                    err_msg=f"NIG {label} analytic vs quad {kw}"
                )


if __name__ == "__main__":
    unittest.main()
