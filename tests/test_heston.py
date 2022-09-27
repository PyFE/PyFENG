import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf
import pyfeng.ex as pfex


class TestHestonMc(unittest.TestCase):
    """
    Tests for Heston MC models
    """
    def test_conv_mv(self):
        """
        Check the infinite sum for the mean and variance
        Returns:

        """
        for no in range(1, 5):
            m = pf.HestonMcGlassermanKim2011(sigma=0.01, mr=1, vov=1)
            kk = 10000  # number of exact terms
            for dt in [1, 2, 4, 8, 16]:
                mean, var = m.x1star_avgvar_mv(dt, 0)
                mean_1, var_1 = m.x1star_avgvar_mv(dt, kk)
                mean_2, var_2 = m.x1star_avgvar_mv_asymp(dt, kk)

                np.testing.assert_(mean_1 > 0)
                np.testing.assert_(var_1 > 0)
                np.testing.assert_allclose(mean_1/mean, mean_2/mean, atol=1e-7)
                np.testing.assert_allclose(var_1/var, var_2/var, atol=1e-7)

                mean, var = m.x2star_avgvar_mv(dt, 0)
                mean_1, var_1 = m.x2star_avgvar_mv(dt, kk)
                mean_2, var_2 = m.x2star_avgvar_mv_asymp(dt, kk)

                np.testing.assert_(mean_1 > 0)
                np.testing.assert_(var_1 > 0)
                np.testing.assert_allclose(mean_1/mean, mean_2/mean, atol=1e-7)
                np.testing.assert_allclose(var_1/var, var_2/var, atol=1e-7)

    def test_avgvar_mv(self):
        for no in [1, 2, 3]:
            m, p, rv = pf.HestonMcGlassermanKim2011.init_benchmark(no)
            ratio = np.random.uniform(0.25, 4, 10)
            m1, v1 = m.cond_avgvar_mv_numeric(rv['args_pricing']['texp'], m.sigma, m.sigma * ratio)
            m2, v2 = m.cond_avgvar_mv(rv['args_pricing']['texp'], m.sigma, m.sigma * ratio)
            np.testing.assert_allclose(m1, m2, rtol=5e-3)  # default: rtol=1e-7
            np.testing.assert_allclose(v1, v2, rtol=5e-3)

    def test_price_mc(self):
        """
        Compare the implied vol of the benchmark cases
        """
        for no in [1, 2, 3]:
            m, p, rv = pf.HestonMcAndersen2008.init_benchmark(no)
            m.set_num_params(n_path=1e5, dt=1/8, rn_seed=123456)
            m.correct_fwd = False

            vol0 = pf.Bsm(None, intr=m.intr, divr=m.divr).impvol(rv['val'], **rv['args_pricing'])

            vol1 = m.vol_smile(**rv['args_pricing'])
            np.testing.assert_allclose(vol0, vol1, atol=5e-3)
            np.testing.assert_allclose(m.result['spot error'], 0, atol=2e-3)

            m, *_ = pf.HestonMcGlassermanKim2011.init_benchmark(no)
            m.set_num_params(n_path=1e5, rn_seed=123456, kk=10)
            m.correct_fwd = False
            vol1 = m.vol_smile(**rv['args_pricing'])
            np.testing.assert_allclose(vol0, vol1, atol=5e-3)
            np.testing.assert_allclose(m.result['spot error'], 0, atol=2e-3)

            m, *_ = pf.HestonMcTseWan2013.init_benchmark(no)
            m.set_num_params(n_path=1e5, rn_seed=123456, dt=1)
            m.correct_fwd = False
            vol1 = m.vol_smile(**rv['args_pricing'])
            np.testing.assert_allclose(vol0, vol1, atol=5e-3)
            np.testing.assert_allclose(m.result['spot error'], 0, atol=2e-3)

            m, *_ = pf.HestonMcChoiKwok2023PoisGe.init_benchmark(no)
            m.correct_fwd = False
            m.set_num_params(n_path=1e5, rn_seed=123456, kk=10, dt=None)
            vol1 = m.vol_smile(**rv['args_pricing'])
            np.testing.assert_allclose(vol0, vol1, atol=5e-3)
            np.testing.assert_allclose(m.result['spot error'], 0, atol=2e-3)

            m.set_num_params(n_path=1e5, rn_seed=123456, kk=1, dt=1/4)
            vol1 = m.vol_smile(**rv['args_pricing'])
            np.testing.assert_allclose(vol0, vol1, atol=5e-3)
            np.testing.assert_allclose(m.result['spot error'], 0, atol=2e-3)


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
