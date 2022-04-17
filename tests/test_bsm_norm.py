import unittest
import copy
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestBsmMethods(unittest.TestCase):
    def test_bsm_price(self):
        bsm = pf.Bsm(sigma=0.2, intr=0.05, divr=0.1)
        result = bsm.price(strike=np.arange(80, 126, 5), spot=100, texp=1.2)
        expect_result = np.array(
            [
                15.71361973,
                12.46799006,
                9.69250803,
                7.38869609,
                5.52948546,
                4.06773495,
                2.94558338,
                2.10255008,
                1.48139131,
                1.03159130,
            ]
        )
        np.testing.assert_almost_equal(result, expect_result)

    def test_norm_price(self):
        bsm = pf.Norm(sigma=20, intr=0.05, divr=0.1)
        result = bsm.price(strike=np.arange(80, 126, 5), spot=100, texp=1.2)
        expect_result = np.array(
            [
                16.572334463,
                13.264066668,
                10.347114013,
                7.849408092,
                5.778270261,
                4.119308420,
                2.838573670,
                1.887447196,
                1.209104773,
                0.745157542,
            ]
        )
        np.testing.assert_almost_equal(result, expect_result)

    def test_bsm_iv_greeks(self):
        for k in range(100):
            spot = np.random.uniform(80, 120)
            strike = np.random.uniform(80, 120)
            sigma = np.random.uniform(0.1, 10)
            texp = np.random.uniform(0.1, 10)
            intr = np.random.uniform(0, 0.3)
            divr = np.random.uniform(0, 0.3)
            cp = 1 if np.random.rand() > 0.5 else -1
            is_fwd = np.random.rand() > 0.5

            # print( spot, strike, vol, texp, intr, divr, cp)
            m_bsm = pf.Bsm(sigma, intr=intr, divr=divr, is_fwd=is_fwd)
            price = (m_bsm.price(strike, spot, texp, cp),)

            # get implied vol
            iv = m_bsm.impvol(price, strike, spot, texp=texp, cp=cp)

            # now price option with the obtained implied vol
            m_bsm2 = copy.copy(m_bsm)
            m_bsm2.sigma = iv
            price_imp = m_bsm2.price(strike, spot, texp, cp)

            # compare the two prices
            self.assertAlmostEqual(price, price_imp, delta=200 * m_bsm.IMPVOL_TOL)

            delta1 = m_bsm.delta(strike=strike, spot=spot, texp=texp, cp=cp)
            delta2 = m_bsm.delta_numeric(strike=strike, spot=spot, texp=texp, cp=cp)
            self.assertAlmostEqual(delta1, delta2, delta=1e-4)

            gamma1 = m_bsm.delta(strike=strike, spot=spot, texp=texp, cp=cp)
            gamma2 = m_bsm.delta_numeric(strike=strike, spot=spot, texp=texp, cp=cp)
            self.assertAlmostEqual(gamma1, gamma2, delta=1e-4)

            vega1 = m_bsm.vega(strike=strike, spot=spot, texp=texp, cp=cp)
            vega2 = m_bsm.vega_numeric(strike=strike, spot=spot, texp=texp, cp=cp)
            self.assertAlmostEqual(vega1, vega2, delta=1e-3)

    def test_norm_iv_greeks(self):
        for k in range(100):
            spot = np.random.uniform(80, 120)
            strike = np.random.uniform(80, 120)
            sigma = np.random.uniform(1, 100)
            texp = np.random.uniform(0.1, 10)
            intr = np.random.uniform(0, 0.3)
            divr = np.random.uniform(0, 0.3)
            cp = 1 if np.random.rand() > 0.5 else -1
            is_fwd = np.random.rand() > 0.5

            # print( spot, strike, vol, texp, intr, divr, cp)
            m_norm = pf.Norm(sigma, intr=intr, divr=divr, is_fwd=is_fwd)
            price = m_norm.price(strike, spot, texp, cp)

            # get implied vol
            iv = m_norm.impvol(price, strike, spot, texp=texp, cp=cp)

            # now price option with the obtained implied vol
            m_norm2 = copy.copy(m_norm)
            m_norm2.sigma = iv
            price_imp = m_norm2.price(strike, spot, texp, cp)

            # compare the two prices
            self.assertAlmostEqual(price, price_imp, delta=200 * m_norm.IMPVOL_TOL)

            delta1 = m_norm.delta(strike=strike, spot=spot, texp=texp, cp=cp)
            delta2 = m_norm.delta_numeric(strike=strike, spot=spot, texp=texp, cp=cp)
            self.assertAlmostEqual(delta1, delta2, delta=1e-4)

            gamma1 = m_norm.delta(strike=strike, spot=spot, texp=texp, cp=cp)
            gamma2 = m_norm.delta_numeric(strike=strike, spot=spot, texp=texp, cp=cp)
            self.assertAlmostEqual(gamma1, gamma2, delta=1e-4)

            vega1 = m_norm.vega(strike=strike, spot=spot, texp=texp, cp=cp)
            vega2 = m_norm.vega_numeric(strike=strike, spot=spot, texp=texp, cp=cp)
            self.assertAlmostEqual(vega1, vega2, delta=1e-3)

    def test_BsmDisp(self):
        strike = np.arange(80, 126, 5)
        dbs = pf.BsmDisp(sigma=0.2, beta=1, pivot=125, intr=0.05, divr=0.1)

        # DBS = BSM if beta=1
        bsm = pf.Bsm(sigma=dbs.sigma_disp, intr=0.05, divr=0.1)
        r1 = bsm.price(strike, 100, 2.5, cp=-1)
        r2 = dbs.price(strike, 100, 2.5, cp=-1)
        np.testing.assert_almost_equal(r1, r2)

        # DBS = Norm if beta=0
        dbs.beta = 0.0001
        dbs.is_fwd = True
        norm = pf.Norm(
            sigma=dbs.sigma_disp * dbs.pivot, intr=0.05, divr=0.1, is_fwd=True
        )
        r1 = norm.price(strike, 100, 2.5, cp=-1)
        r2 = dbs.price(strike, 100, 2.5, cp=-1)
        np.testing.assert_almost_equal(r1 / r2, 1, decimal=4)
        dbs.is_fwd = False

        # Approximate BSM vol
        dbs.beta = 0.2
        v1 = dbs.vol_smile(strike, 100, 2.5, model="bsm")
        v2 = dbs.vol_smile(strike, 100, 2.5, model="bsm-approx")
        np.testing.assert_almost_equal(v1 / v2, 1, decimal=4)

        p1 = dbs.price(strike, 100, 2.5)
        p2 = pf.Bsm(v1, intr=0.05, divr=0.1).price(strike, 100, 2.5)
        np.testing.assert_almost_equal(p1, p2)

        # Approximate Bachelier vol
        dbs.beta = 0.8
        v1 = dbs.vol_smile(strike, 100, 2.5, model="norm")
        v2 = dbs.vol_smile(strike, 100, 2.5, model="norm-approx")
        np.testing.assert_almost_equal(v1 / v2, 1, decimal=4)

        p1 = dbs.price(strike, 100, 2.5)
        p2 = pf.Norm(v1, intr=0.05, divr=0.1).price(strike, 100, 2.5)
        np.testing.assert_almost_equal(p1, p2)


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
