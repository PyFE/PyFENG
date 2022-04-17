import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestMultiAsset(unittest.TestCase):
    def test_BsmSpreadKirk(self):
        m = pf.BsmSpreadKirk((0.2, 0.3), cor=-0.5)
        result = m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        result2 = np.array([22.1563225, 17.1844182, 12.9897421, 9.6414167, 6.9994207])
        np.testing.assert_almost_equal(result, result2)

    def test_BsmSpreadBjerksund2014(self):
        m = pf.BsmSpreadBjerksund2014((0.2, 0.3), cor=-0.5)
        result = m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        result2 = np.array([22.1317202, 17.1830425, 12.9897421, 9.5443194, 6.8061260])
        np.testing.assert_almost_equal(result, result2)

    def test_NormSpread(self):
        m = pf.NormSpread((20, 30), cor=-0.5, intr=0.05)
        result = m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        result2 = np.array([17.9567619, 13.7464682, 10.2666994, 7.4709872, 5.2905716])
        np.testing.assert_almost_equal(result, result2)

    def test_BsmBasketLevy1992(self):
        """
        Test case in Krekel, M., de Kock, J., Korn, R., & Man, T.-K. (2004)
        Table 2 (Varying K), 3 (Varying fwd) and 1 (Varying correlation)
        """
        texp = 5
        rho = 0.5
        o4 = np.ones(4)
        sigma = o4 * 0.4
        fwd = o4 * 100
        p_grid = np.arange(50, 151, 10)

        # Table 2
        m = pf.BsmBasketLevy1992(sigma, rho)
        result = np.round(m.price(p_grid, fwd, texp), 2)
        result2 = np.array(
            [54.34, 47.52, 41.57, 36.4, 31.92, 28.05, 24.7, 21.8, 19.28, 17.1, 15.19]
        )
        np.testing.assert_almost_equal(result, result2)

        # Table 3
        m = pf.BsmBasketLevy1992(sigma, rho)
        result = np.round(m.price(100, p_grid[:, None] * o4, texp), 2)
        result2 = np.array(
            [4.34, 7.52, 11.57, 16.4, 21.92, 28.05, 34.7, 41.8, 49.28, 57.1, 65.19]
        )
        np.testing.assert_almost_equal(result, result2)

        # Table 1
        result2 = np.array([22.06, 25.17, 28.05, 30.75, 32.04, 33.92])
        result = np.zeros_like(result2)
        rhos = [0.1, 0.3, 0.5, 0.7, 0.8, 0.95]
        for k in range(len(rhos)):
            m = pf.BsmBasketLevy1992(sigma, rhos[k])
            result[k] = np.round(m.price(100, fwd, texp), 2)
        np.testing.assert_almost_equal(result, result2)

    def test_BsmBasketMilevsky1998(self):
        """
        Test case in Krekel, M., de Kock, J., Korn, R., & Man, T.-K. (2004)
        Table 2 (Varying K), 3 (Varying fwd) and 1 (Varying correlation)
        """
        texp = 5
        rho = 0.5
        o4 = np.ones(4)
        sigma = o4 * 0.4
        fwd = o4 * 100
        p_grid = np.arange(50, 151, 10)

        # Table 2
        m = pf.BsmBasketMilevsky1998(sigma, rho)
        result = np.round(m.price(p_grid, fwd, texp), 2)
        # Replaced 38.01 (Krekel et al., 2004) with 38.03
        result2 = np.array(
            [51.93, 44.41, 38.03, 32.68, 28.22, 24.5, 21.39, 18.77, 16.57, 14.7, 13.1]
        )
        np.testing.assert_almost_equal(result, result2)

        # Table 3
        m = pf.BsmBasketMilevsky1998(sigma, rho)
        result = np.round(m.price(100, p_grid[:, None] * o4, texp), 2)
        result2 = np.array(
            [3.93, 6.56, 9.95, 14.1, 18.97, 24.5, 30.63, 37.32, 44.49, 52.08, 60.05]
        )
        np.testing.assert_almost_equal(result, result2)

        # Table 1
        result2 = np.array([20.25, 22.54, 24.5, 26.18, 26.93, 27.97])
        result = np.zeros_like(result2)
        rhos = [0.1, 0.3, 0.5, 0.7, 0.8, 0.95]
        for k in range(len(rhos)):
            m = pf.BsmBasketMilevsky1998(sigma, rhos[k])
            result[k] = np.round(m.price(100, fwd, texp), 2)
        np.testing.assert_almost_equal(result, result2)

    def test_BsmRainbow2(self):
        o2 = np.ones(2)
        m = pf.BsmMax2(0.2 * o2, cor=0, divr=0.1, intr=0.05)
        result2 = np.array([6.655098004, 11.195681033, 16.92856557])
        result = np.zeros_like(result2)
        fwds = [90, 100, 110]
        for k in range(len(fwds)):
            result[k] = m.price(100, fwds[k] * o2, 3)

    def test_BsmNormNdMc(self):
        spot = np.ones(4) * 100
        sigma = np.ones(4) * 0.4
        texp = 5
        # Basket Option with equal weight
        payoff = lambda x: np.fmax(np.mean(x, axis=1) - strike, 0)  # Basket option
        strikes = np.arange(80, 121, 10)

        # Test BsmNd
        m = pf.BsmNdMc(sigma, cor=0.5, rn_seed=1234)
        m.simulate(tobs=[texp], n_path=20000)
        p = []
        for strike in strikes:
            p.append(m.price_european(spot, texp, payoff))
        p = np.array(p)
        p2 = np.array([36.31612946, 31.80861014, 27.91269315, 24.55319506, 21.62677625])
        np.testing.assert_almost_equal(p, p2)

        # Test NormNd
        m = pf.NormNdMc(sigma * spot, cor=0.5, rn_seed=1234)
        m.simulate(tobs=[texp], n_path=20000)
        p = []
        for strike in strikes:
            p.append(m.price_european(spot, texp, payoff))
        p = np.array(p)
        p2 = np.array([39.42304794, 33.60383167, 28.32667559, 23.60383167, 19.42304794])
        np.testing.assert_almost_equal(p, p2)

    def test_BsmBasket1Bm(self):
        ### Check the BsmBasket1Bm price should be same as that of the Bsm price if sigma components are same.
        for k in range(100):
            n = np.random.randint(1, 8)
            spot = np.random.uniform(80, 120, size=n)
            strike = np.random.uniform(80, 120, size=10)
            sigma = np.random.uniform(0.01, 1) * np.ones(n)
            texp = np.random.uniform(0.1, 10)
            intr = np.random.uniform(0, 0.1)
            divr = np.random.uniform(0, 0.1)
            weight = np.random.rand(n)
            weight /= np.sum(weight)

            cp = np.where(np.random.rand(10) > 0.5, 1, -1)
            is_fwd = np.random.rand() > 0.5

            m = pf.BsmBasket1Bm(
                sigma, weight=weight, intr=intr, divr=divr, is_fwd=is_fwd
            )
            p = m.price(strike, spot, texp, cp)

            m2 = pf.Bsm(sigma[0], intr=intr, divr=divr, is_fwd=is_fwd)
            p2 = m2.price(strike, np.sum(spot * weight), texp, cp)
            np.testing.assert_almost_equal(p, p2)


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
