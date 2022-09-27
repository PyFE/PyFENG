import unittest
import numpy as np
import scipy.integrate as scin
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf
import pyfeng.ex as pfex


class TestOusvKL(unittest.TestCase):
    """
    Tests for Heston MC models
    """
    def test_infinite_sum(self):
        """
        Check the infinite sum
        Returns:

        """
        for mrt, ns in ((1.0, 2), (1.5, 4), (0.5, 6)):
            ### ns should be even number

            n_pi_2 = (np.arange(1, 160001) * np.pi)**2
            An2 = 2 / (mrt**2 + n_pi_2)
            An4 = An2**2
            An6 = An4 * An2
            An2n2 = An2 / n_pi_2
            An6n2 = n_pi_2 * An6

            m = pfex.OusvMcChoi2023KL(1, 1, 0, 1, theta=mrt)

            assert np.isclose(1, sum(An2[ns:])/m._a2sum(mrt, ns=ns), atol=1e-4)
            assert np.isclose(1, sum(An2[ns::2])/m._a2sum(mrt, ns=ns, odd=1), atol=1e-4)
            assert np.isclose(1, sum(An2[ns+1::2])/m._a2sum(mrt, ns=ns, odd=2), atol=1e-4)

            assert np.isclose(1, np.sum(An2n2[ns:])/m._a2overn2sum(mrt, ns=ns))
            assert np.isclose(1, np.sum(An2n2[ns::2])/m._a2overn2sum(mrt, ns=ns, odd=1))
            assert np.isclose(1, np.sum(An2n2[ns+1::2])/m._a2overn2sum(mrt, ns=ns, odd=2))

            assert np.isclose(1, sum(An4[ns:])/m._a4sum(mrt, ns=ns))
            assert np.isclose(1, sum(An4[ns::2])/m._a4sum(mrt, ns=ns, odd=1))
            assert np.isclose(1, sum(An4[ns+1::2])/m._a4sum(mrt, ns=ns, odd=2))

            assert np.isclose(1, sum(An6[ns:])/m._a6sum(mrt, ns=ns))
            assert np.isclose(1, sum(An6[ns::2])/m._a6sum(mrt, ns=ns, odd=1))
            assert np.isclose(1, sum(An6[ns+1::2])/m._a6sum(mrt, ns=ns, odd=2))

            assert np.isclose(1, sum(An6n2[ns:])/m._a6n2sum(mrt, ns=ns))
            assert np.isclose(1, sum(An6n2[ns::2])/m._a6n2sum(mrt, ns=ns, odd=1))
            assert np.isclose(1, sum(An6n2[ns+1::2])/m._a6n2sum(mrt, ns=ns, odd=2))

    def test_integarl_sin_path(self):
        """
        Vol/var integrated with Simpson method should equal to the analytic integral

        Returns:

        """
        sheet_no = 1
        m, p, rv = pfex.OusvMcChoi2023KL.init_benchmark(sheet_no)
        n_sin = 6
        n_path = 50
        m.set_num_params(n_path=n_path, rn_seed=123456, n_sin=4, dt=None)

        n_step = 10000
        t_grid = np.arange(0, n_step + 1) / n_step
        for texp in (0.2, 1.0, 5.0, 10.0):
            zn = np.random.normal(size=(n_sin + 1, n_path))
            vol_path = m.vol_path_sin(t_grid * texp, zn)

            avgvol_simp = scin.simps(vol_path, dx=1, axis=0) / n_step
            avgvar_simp = scin.simps(vol_path**2, dx=1, axis=0) / n_step
            sig_t, avgvar, avgvol = m.cond_states_step(texp, m.sigma, zn=zn)

            assert np.all(np.isclose(1, vol_path[-1, :]/sig_t))
            assert np.all(np.isclose(1, avgvol_simp/avgvol))
            assert np.all(np.isclose(1, avgvar_simp/avgvar))


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
