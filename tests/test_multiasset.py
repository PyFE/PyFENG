import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestMultiAsset(unittest.TestCase):
    def test_BsmSpreadKirk(self):
        m = pf.BsmSpreadKirk((0.2, 0.3), rho=-0.5)
        result = m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        result2 = np.array([22.1563225, 17.1844182, 12.9897421, 9.6414167, 6.9994207])
        np.testing.assert_almost_equal(result, result2)

    def test_BsmSpreadBjerksund2014(self):
        m = pf.BsmSpreadBjerksund2014((0.2, 0.3), rho=-0.5)
        result = m.price(np.arange(-2, 3) * 10, [100, 120], 1.3)
        result2 = np.array([22.1317202, 17.1830425, 12.9897421, 9.5443194, 6.8061260])
        np.testing.assert_almost_equal(result, result2)

    def test_NormSpread(self):
        m = pf.NormBasket.init_spread((20, 30), rho=-0.5, intr=0.05)
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
        m = pf.BsmMax2(0.2 * o2, rho=0, divr=0.1, intr=0.05)
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
        m = pf.BsmNdMc(sigma, rho=0.5, rn_seed=1234)
        m.simulate(tobs=[texp], n_path=20000)
        p = []
        for strike in strikes:
            p.append(m.price_european(spot, texp, payoff))
        p = np.array(p)
        p2 = np.array([36.31612946, 31.80861014, 27.91269315, 24.55319506, 21.62677625])
        np.testing.assert_almost_equal(p, p2)

        # Test NormNd
        m = pf.NormNdMc(sigma * spot, rho=0.5, rn_seed=1234)
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


class TestBsmBasketChoi2018(unittest.TestCase):
    """
    Tests for BsmBasketChoi2018 against numerical results in:
    Choi J (2018) Sum of all Black-Scholes-Merton models: An efficient pricing method for
    spread, basket, and Asian options. J Futures Markets 38:627-644.

    All prices are present values (forward value discounted by e^{-rT}).
    """

    def test_spread_S1(self):
        """
        Table 4: set S1, spread option with varying K.
        N=2, T=1, S=(100, 96), w=(1,-1), sigma=(20%, 10%), rho=50%, q=5%, r=10%.
        Fast prices with M_2=4 and control variate: errors ~1e-8 vs converged prices.
        """
        m = pf.BsmBasketChoi2018(
            (0.20, 0.10), rho=0.50,
            weight=np.array([1.0, -1.0]),
            intr=0.10, divr=0.05,
        )
        m.set_num_params(n_quad=[4])
        strike = np.arange(0.0, 4.1, 0.4)
        result = m.price(strike, np.array([100.0, 96.0]), texp=1.0)
        result2 = np.array([
            8.5132252, 8.3124607, 8.1149938, 7.9208198, 7.7299325,
            7.5423239, 7.3579843, 7.1769024, 6.9990651, 6.8244581, 6.6530651,
        ])
        np.testing.assert_almost_equal(result, result2, decimal=6)

    def test_spread_S2_rho(self):
        """
        Table 5: set S2, spread option with varying rho (K=100, lam=9).
        N=2, T=1, S=(200, 100), w=(1,-1), sigma=(15%, 30%), q=r=0.
        Fast prices with lam=9: errors ~5e-8 vs converged prices.
        """
        spot = np.array([200.0, 100.0])
        rhos   = [0.90, 0.70, 0.50, 0.30, 0.10, -0.10, -0.30, -0.50, -0.70, -0.90]
        result2 = np.array([
            5.4792720, 9.3209439, 11.9804918, 14.1425869, 16.0102190,
            17.6770249, 19.1954201, 20.5982705, 21.9077989, 23.1398674,
        ])
        result = np.zeros(len(rhos))
        for i, rho in enumerate(rhos):
            m = pf.BsmBasketChoi2018((0.15, 0.30), rho=rho, weight=np.array([1.0, -1.0]))
            m.set_num_params(lam=9)
            result[i] = m.price(100.0, spot, texp=1.0)
        np.testing.assert_almost_equal(result, result2, decimal=6)

    def test_basket_B1_strike(self):
        """
        Table 6: set B1, basket option with varying K (lam=9, M=125).
        N=4, T=5, S=100, w=1/4, sigma=40%, rho=50%, r=q=0.
        Fast prices with lam=9 differ from converged prices by at most 1.5e-4.
        """
        m = pf.BsmBasketChoi2018(0.4 * np.ones(4), rho=0.5)
        m.set_num_params(lam=9)
        strike = np.arange(50, 151, 10, dtype=float)
        result = m.price(strike, 100.0 * np.ones(4), texp=5.0)
        result2 = np.array([
            54.3101761, 47.4811265, 41.5225192, 36.3517843, 31.8768032,
            28.0073695, 24.6605295, 21.7625789, 19.2493294, 17.0655420, 15.1640103,
        ])
        np.testing.assert_almost_equal(result, result2, decimal=3)

    def test_basket_B1_rho(self):
        """
        Table 7: set B1, basket option with varying rho (K=100, lam=9).
        N=4, T=5, S=100, w=1/4, sigma=40%, r=q=0.
        For rho<=0.80: fast prices accurate to 3 decimal places (M>=27).
        For rho=0.95: only M=8 nodes; fast price error ~3e-3 (per paper FP Err=-3.1e-3).
        """
        spot = 100.0 * np.ones(4)

        # rho in [-0.10, 0.80]: M >= 27, accurate to 3 decimal places
        rhos   = [-0.10, 0.10, 0.30, 0.50, 0.80]
        result2 = np.array([17.7569163, 21.6920965, 25.0292992, 28.0073695, 32.0412265])
        result = np.zeros(len(rhos))
        for i, rho in enumerate(rhos):
            m = pf.BsmBasketChoi2018(0.4 * np.ones(4), rho=rho)
            m.set_num_params(lam=9)
            result[i] = m.price(100.0, spot, texp=5.0)
        np.testing.assert_almost_equal(result, result2, decimal=3)

        # rho=0.95: M=8, coarser — test to 2 decimal places only
        m = pf.BsmBasketChoi2018(0.4 * np.ones(4), rho=0.95)
        m.set_num_params(lam=9)
        np.testing.assert_almost_equal(m.price(100.0, spot, texp=5.0), 33.9186874, decimal=2)


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
