import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestBsmCos(unittest.TestCase):
    """
    Tests for Black-Scholes COS pricer (BsmCos).

    BsmCos uses exact analytic cumulants so its truncation range is tight
    and accuracy vs the closed-form Bsm should be near machine precision.
    """

    strikes = np.arange(80., 121., 10.)   # [80, 90, 100, 110, 120]
    spot    = 100.
    sigma, intr, divr = 0.2, 0.05, 0.1

    def test_vs_analytic(self):
        """BsmCos matches closed-form Bsm to within 1e-10 for calls and puts."""
        cos = pf.BsmCos(self.sigma, intr=self.intr, divr=self.divr)
        ref = pf.Bsm(self.sigma,    intr=self.intr, divr=self.divr)
        for texp in [0.5, 1.2, 3.0]:
            for cp in [1, -1]:
                np.testing.assert_allclose(
                    cos.price(self.strikes, self.spot, texp, cp=cp),
                    ref.price(self.strikes, self.spot, texp, cp=cp),
                    atol=1e-10,
                    err_msg=f"texp={texp}, cp={cp}",
                )

    def test_reference_values(self):
        """Pin prices to the values from the BsmCos docstring example."""
        m      = pf.BsmCos(0.2, intr=0.05, divr=0.1)
        result = m.price(np.arange(80., 121., 10.), 100., 1.2)
        expect = np.array([15.71361973, 9.69250803, 5.52948546, 2.94558338, 1.48139131])
        np.testing.assert_allclose(result, expect, atol=1e-7)

    def test_put_call_parity(self):
        """C - P = df * (F - K) to within 1e-12."""
        m = pf.BsmCos(self.sigma, intr=self.intr, divr=self.divr)
        for texp in [0.5, 1.2, 3.0]:
            C = m.price(self.strikes, self.spot, texp, cp=1)
            P = m.price(self.strikes, self.spot, texp, cp=-1)
            fwd, df, _ = m._fwd_factor(self.spot, texp)
            np.testing.assert_allclose(
                C - P, df * (fwd - self.strikes), atol=1e-12,
                err_msg=f"texp={texp}",
            )

    def test_scalar_in_scalar_out(self):
        """Scalar strike and scalar cp must return a Python float."""
        m = pf.BsmCos(self.sigma, intr=self.intr, divr=self.divr)
        p = m.price(100., self.spot, 1.0, cp=1)
        self.assertIsInstance(p, float)

    def test_array_output_shape(self):
        """Output shape matches the input strikes array."""
        m = pf.BsmCos(self.sigma, intr=self.intr, divr=self.divr)
        p = m.price(self.strikes, self.spot, 1.0)
        self.assertEqual(p.shape, self.strikes.shape)

    def test_is_fwd(self):
        """is_fwd=True treats spot as the forward; prices must be identical."""
        m_spot = pf.BsmCos(self.sigma, intr=self.intr, divr=self.divr, is_fwd=False)
        m_fwd  = pf.BsmCos(self.sigma, intr=self.intr,                  is_fwd=True)
        fwd_val = m_spot.forward(self.spot, 1.2)
        np.testing.assert_allclose(
            m_spot.price(self.strikes, self.spot,   1.2),
            m_fwd.price( self.strikes, fwd_val,     1.2),
            atol=1e-10,
        )

    def test_default_n_cos_and_L(self):
        """Class-level defaults for n_cos and L are set correctly."""
        m = pf.BsmCos(self.sigma)
        self.assertEqual(m.n_cos, 128)
        self.assertEqual(m.L, 12.0)


class TestHestonCos(unittest.TestCase):
    """
    Tests for Heston COS pricer (HestonCos).

    HestonCos uses the same Lord-Kahl (2010) MGF as HestonFft and analytic
    cumulants from Fang & Oosterlee (2008) Appendix A. Cross-validation
    against HestonFft is the primary accuracy check (atol = 5e-4).
    """

    strikes        = np.array([80., 90., 100., 110., 120.])
    spot           = 100.
    sigma, vov, mr, rho = 0.04, 0.5, 1.5, -0.7   # primary benchmark params

    def _cos(self, **kw):
        return pf.HestonCos(self.sigma, vov=self.vov, mr=self.mr, rho=self.rho, **kw)

    def _fft(self, **kw):
        return pf.HestonFft(self.sigma, vov=self.vov, mr=self.mr, rho=self.rho, **kw)

    def test_vs_fft(self):
        """HestonCos agrees with HestonFft to within 5e-4 across maturities."""
        cos, fft = self._cos(), self._fft()
        for texp in [0.5, 1.0, 5.0, 10.0]:
            for cp in [1, -1]:
                np.testing.assert_allclose(
                    cos.price(self.strikes, self.spot, texp, cp=cp),
                    fft.price(self.strikes, self.spot, texp, cp=cp),
                    atol=5e-4,
                    err_msg=f"texp={texp}, cp={cp}",
                )

    def test_vs_fft_multiple_param_sets(self):
        """Cross-validate for varied Heston parameters (different rho, vov, mr)."""
        param_sets = [
            dict(sigma=0.09, vov=0.3, mr=4.0, rho=0.0),   # Feller satisfied, no corr
            dict(sigma=0.04, vov=0.5, mr=2.0, rho=0.5),   # positive correlation
            dict(sigma=0.06, vov=0.4, mr=1.0, rho=-0.5),  # moderate parameters
        ]
        for params in param_sets:
            cos = pf.HestonCos(**params)
            fft = pf.HestonFft(**params)
            np.testing.assert_allclose(
                cos.price(self.strikes, self.spot, 1.0),
                fft.price(self.strikes, self.spot, 1.0),
                atol=5e-4,
                err_msg=f"params={params}",
            )

    def test_put_call_parity(self):
        """C - P = df*(F-K) within 5e-4 (bounded by truncation-range residual)."""
        cos = self._cos()
        for texp in [0.5, 1.0, 3.0]:
            C = cos.price(self.strikes, self.spot, texp, cp=1)
            P = cos.price(self.strikes, self.spot, texp, cp=-1)
            fwd, df, _ = cos._fwd_factor(self.spot, texp)
            np.testing.assert_allclose(
                C - P, df * (fwd - self.strikes), atol=5e-4,
                err_msg=f"texp={texp}",
            )

    def test_convergence_in_n(self):
        """Pricing error vs HestonFft strictly decreases as n_cos increases."""
        ref = self._fft().price(self.strikes, self.spot, 1.0)

        m_64, m_256 = self._cos(), self._cos()
        m_64.n_cos, m_256.n_cos = 64, 256

        err_64  = np.max(np.abs(m_64.price( self.strikes, self.spot, 1.0) - ref))
        err_256 = np.max(np.abs(m_256.price(self.strikes, self.spot, 1.0) - ref))
        self.assertLess(err_256, err_64)

    def test_scalar_in_scalar_out(self):
        """Scalar strike + scalar cp returns a Python float."""
        p = self._cos().price(100., self.spot, 1.0, cp=1)
        self.assertIsInstance(p, float)

    def test_array_output_shape(self):
        """Output shape matches the strikes array."""
        p = self._cos().price(self.strikes, self.spot, 1.0)
        self.assertEqual(p.shape, self.strikes.shape)

    def test_nonnegative_prices(self):
        """Call and put prices are non-negative for all strikes."""
        cos = self._cos()
        for cp in [1, -1]:
            p = cos.price(self.strikes, self.spot, 1.0, cp=cp)
            np.testing.assert_(np.all(p >= 0), f"Negative prices for cp={cp}")

    def test_theta_defaults_to_sigma(self):
        """When theta is omitted, HestonCos inherits SvABC default theta=sigma."""
        m = self._cos()
        self.assertEqual(m.theta, m.sigma)

    def test_n_cos_instance_override(self):
        """n_cos can be overridden on an instance; setting it changes results."""
        m = self._cos()
        self.assertEqual(m.n_cos, 128)          # class-level default
        p_128 = m.price(self.strikes, self.spot, 1.0)
        m.n_cos = 32
        p_32 = m.price(self.strikes, self.spot, 1.0)
        # n_cos=32 is too coarse — result should differ from n_cos=128
        self.assertFalse(np.allclose(p_32, p_128, atol=1e-8))


class TestVarGammaCos(unittest.TestCase):
    """
    Tests for Variance Gamma COS pricer (VarGammaCos).

    Uses the same SvABC parameter convention as VarGammaFft:
        sigma = Brownian vol, vov = Gamma clock variance rate (nu), theta = drift.
    Cross-validation against VarGammaFft is the primary accuracy check.
    """

    strikes = np.array([80., 90., 100., 110., 120.])
    spot    = 100.
    # F&O (2008) Section 5.4 parameters
    sigma, vov, theta, intr = 0.12, 0.2, -0.14, 0.1

    def _cos(self, **kw):
        return pf.VarGammaCos(self.sigma, vov=self.vov, theta=self.theta,
                               intr=self.intr, **kw)

    def _fft(self, **kw):
        return pf.VarGammaFft(self.sigma, vov=self.vov, theta=self.theta,
                               intr=self.intr, **kw)

    def test_vs_fft(self):
        """VarGammaCos agrees with VarGammaFft to within 5e-4."""
        cos, fft = self._cos(), self._fft()
        for texp in [0.5, 1.0, 2.0]:
            np.testing.assert_allclose(
                cos.price(self.strikes, self.spot, texp),
                fft.price(self.strikes, self.spot, texp),
                atol=5e-4,
                err_msg=f"texp={texp}",
            )

    def test_put_call_parity(self):
        """C - P = df*(F-K) to within 1e-5."""
        cos = self._cos()
        for texp in [0.5, 1.0, 2.0]:
            C = cos.price(self.strikes, self.spot, texp, cp=1)
            P = cos.price(self.strikes, self.spot, texp, cp=-1)
            fwd, df, _ = cos._fwd_factor(self.spot, texp)
            np.testing.assert_allclose(
                C - P, df * (fwd - self.strikes), atol=1e-5,
                err_msg=f"texp={texp}",
            )

    def test_convergence_in_n(self):
        """Pricing error vs VarGammaFft strictly decreases as n_cos increases."""
        ref = self._fft().price(self.strikes, self.spot, 1.0)

        m_64, m_256 = self._cos(), self._cos()
        m_64.n_cos, m_256.n_cos = 64, 256

        err_64  = np.max(np.abs(m_64.price( self.strikes, self.spot, 1.0) - ref))
        err_256 = np.max(np.abs(m_256.price(self.strikes, self.spot, 1.0) - ref))
        self.assertLess(err_256, err_64)

    def test_scalar_in_scalar_out(self):
        """Scalar strike + scalar cp returns a Python float."""
        p = self._cos().price(100., self.spot, 1.0, cp=1)
        self.assertIsInstance(p, float)

    def test_array_output_shape(self):
        """Output shape matches the strikes array."""
        p = self._cos().price(self.strikes, self.spot, 1.0)
        self.assertEqual(p.shape, self.strikes.shape)

    def test_nonnegative_prices(self):
        """Call and put prices are non-negative for all strikes."""
        cos = self._cos()
        for cp in [1, -1]:
            p = cos.price(self.strikes, self.spot, 1.0, cp=cp)
            np.testing.assert_(np.all(p >= 0), f"Negative prices for cp={cp}")

    def test_call_decreasing_in_strike(self):
        """Call prices must decrease monotonically with strike (no-arbitrage)."""
        cos = self._cos()
        C = cos.price(self.strikes, self.spot, 1.0, cp=1)
        np.testing.assert_(np.all(np.diff(C) < 0),
                            "Call prices are not monotonically decreasing in strike")

    def test_put_increasing_in_strike(self):
        """Put prices must increase monotonically with strike (no-arbitrage)."""
        cos = self._cos()
        P = cos.price(self.strikes, self.spot, 1.0, cp=-1)
        np.testing.assert_(np.all(np.diff(P) > 0),
                            "Put prices are not monotonically increasing in strike")


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
