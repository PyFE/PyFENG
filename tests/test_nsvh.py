import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf
from pyfeng.nsvh import NsvhABC


class NsvhTest(NsvhABC):
    """Concrete subclass of NsvhABC for testing price_vsk only."""
    def price(self, strike, spot, texp, cp=1):
        return None


class TestNsvhPriceVsk(unittest.TestCase):

    # Parameter sets: (sigma, vov, rho, texp)
    PARAMS = [
        (0.2, 0.3, 0.0, 1.0),
        (0.2, 0.3, -0.5, 1.0),
        (0.2, 0.3, 0.5, 1.0),
        (0.1, 0.5, -0.3, 0.5),
        (0.3, 0.8, 0.4, 2.0),
        (0.15, 0.2, 0.0, 5.0),
        (0.4, 1.0, -0.7, 0.25),
        (0.05, 0.6, 0.6, 3.0),
        #(0.2, 0.3,  1.0, 1.0),  # rho=+1: lognormal limit
        #(0.2, 0.3, -1.0, 1.0),  # rho=-1: lognormal limit
    ]

    def test_fit_roundtrip(self):
        """fit(price_vsk(...)) recovers the original parameters (interior rho only)."""
        for sigma, vov, rho, texp in self.PARAMS_INTERIOR:
            m = pf.Nsvh1(sigma=sigma, vov=vov, rho=rho)
            mvs = m.price_vsk(texp=texp)
            m2 = pf.Nsvh1.from_vsk(mvs, texp=texp)
            np.testing.assert_allclose(m2.sigma, sigma, rtol=1e-10,
                err_msg=f"sigma mismatch: {sigma}, {vov}, {rho}, {texp}")
            np.testing.assert_allclose(m2.vov, vov, rtol=1e-10,
                err_msg=f"vov mismatch: {sigma}, {vov}, {rho}, {texp}")
            np.testing.assert_allclose(m2.rho, rho, rtol=1e-10,
                err_msg=f"rho mismatch: {sigma}, {vov}, {rho}, {texp}")

    def test_price_vsk_lam0_vs_sabrnorm(self):
        """
        NsvhABC(lam=0).price_vsk == SabrNormVolApprox.price_vsk
        """
        for sigma, vov, rho, texp in self.PARAMS:
            m_nsvh = NsvhTest(sigma=sigma, vov=vov, rho=rho, lam=0)
            m_sabr = pf.SabrNormVolApprox(sigma=sigma, vov=vov, rho=rho)

            v_nsvh = m_nsvh.price_vsk(texp=texp)
            v_sabr = m_sabr.price_vsk(texp=texp)

            np.testing.assert_allclose(
                v_nsvh, v_sabr,
                rtol=1e-10,
                err_msg=f"lam=0 mismatch: sigma={sigma}, vov={vov}, rho={rho}, texp={texp}"
            )

    def test_price_vsk_lam1_vs_nsvh1(self):
        """
        NsvhABC(lam=1).price_vsk == Nsvh1.price_vsk
        """
        for sigma, vov, rho, texp in self.PARAMS:
            m_nsvh = NsvhTest(sigma=sigma, vov=vov, rho=rho, lam=1)
            m_nsvh1 = pf.Nsvh1(sigma=sigma, vov=vov, rho=rho)

            v_nsvh = m_nsvh.price_vsk(texp=texp)
            v_nsvh1 = m_nsvh1.price_vsk(texp=texp)

            np.testing.assert_allclose(
                v_nsvh, v_nsvh1,
                rtol=1e-10,
                err_msg=f"lam=1 mismatch: sigma={sigma}, vov={vov}, rho={rho}, texp={texp}"
            )


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
