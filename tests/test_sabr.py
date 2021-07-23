import unittest
import copy
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestSabr(unittest.TestCase):
    def test_Hagan2002(self):
        for k in list(range(1, 19)) + [22, 23]:
            m, df, rv = pf.SabrHagan2002.init_benchmark(k)
            # ref = rv['ref']
            # print(f'Sheet {k:02d}: {ref}')
            v1 = np.round(m.vol_for_price(**rv["args_pricing"]), 4)
            v2 = df["IV Hagan"].values
            np.testing.assert_almost_equal(v1, v2)

    def test_SabrNorm(self):
        for k in [22, 23]:
            m, df, rv = pf.SabrNorm.init_benchmark(k)
            v1 = m.price(**rv["args_pricing"])
            m, df, rv = pf.SabrChoiWu2021H.init_benchmark(k)
            v2 = m.price(**rv["args_pricing"])
            np.testing.assert_almost_equal(v1, v2)

    def test_SabrNormATM(self):
        for k in [22, 23]:
            m, df, rv = pf.SabrNorm.init_benchmark(k)
            m.is_atmvol = True
            np.testing.assert_almost_equal(m.vol_smile(0, 0, texp=0.1), m.sigma)
            np.testing.assert_almost_equal(m.vol_smile(0, 0, texp=10), m.sigma)

            m, df, rv = pf.Nsvh1.init_benchmark(k)
            m.is_atmvol = True
            np.testing.assert_almost_equal(m.vol_smile(0, 0, texp=0.1), m.sigma)
            np.testing.assert_almost_equal(m.vol_smile(0, 0, texp=10), m.sigma)

    def test_PaulotBsm(self):
        for k in list(range(1, 19)):
            m, df, rv = pf.SabrChoiWu2021P.init_benchmark(k)
            m._base_beta = 1.0  # For Paulot's BS volatility approximation
            # print(f'Sheet {k:02d}: {ref}')
            v1 = np.round(m.vol_for_price(**rv["args_pricing"]), 4)
            v2 = df["IV HL-P"].values
            np.testing.assert_almost_equal(v1, v2)

    def test_UnCorrChoiWu2021(self):
        # Param Set 19: Table 7 (Case III.C) in Cai et al. (2017). https://doi.org/10.1287/opre.2017.1617
        m, df, rv = pf.SabrUncorrChoiWu2021.init_benchmark(19)
        mass = m.mass_zero(rv["args_pricing"]["spot"], rv["args_pricing"]["texp"])
        p = m.price(**rv["args_pricing"])

        mass2 = 0.7623543217183134
        p2 = np.array(
            [0.04533777, 0.04095806, 0.03889591, 0.03692339, 0.03324944, 0.02992918]
        )

        np.testing.assert_almost_equal(mass, mass2)
        np.testing.assert_almost_equal(p, p2)

    def test_CondMc(self):
        for k in [19, 20]:  # can test 22 (Korn&Tang) also, but difficult to pass
            m, df, rv = pf.SabrCondMc.init_benchmark(k)
            m.set_mc_params(n_path=5e4, dt=0.05, rn_seed=1234)
            p = m.price(**rv["args_pricing"])
            np.testing.assert_almost_equal(p, rv["val"], decimal=4)


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
