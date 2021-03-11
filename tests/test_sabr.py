import unittest
import copy
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestSabr(unittest.TestCase):

    def test_ChoiWu2021(self):
        # Param Set 19: Table 7 (Case III.C) in Cai et al. (2017). https://doi.org/10.1287/opre.2017.1617
        param = {"sigma": 0.4, "vov": 0.6, "rho": 0, "beta": 0.3, 'n_quad': 9}
        fwd, texp = 0.05, 1
        strike = np.array([0.4, 0.8, 1, 1.2, 1.6, 2.0]) * fwd

        m = pf.SabrUncorrChoiWu2021(**param)
        mass = m.mass_zero(fwd, texp)
        p = m.price(strike, fwd, texp)

        mass2 = 0.7623543217183134
        p2 = np.array([
            0.04533777, 0.04095806, 0.03889591, 0.03692339, 0.03324944, 0.02992918
        ])

        np.testing.assert_almost_equal(mass, mass2)
        np.testing.assert_almost_equal(p, p2)


if __name__ == '__main__':
    print(f'Pyfeng loaded from {pf.__path__}')
    unittest.main()
