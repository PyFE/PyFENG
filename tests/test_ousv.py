import unittest
import numpy as np
import scipy.special as spsp
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


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

            m = pf.OusvMcChoi2023KL(1, 1, 0, 1, theta=mrt)

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
        m, p, rv = pf.OusvMcChoi2023KL.init_benchmark(sheet_no)
        n_sin = 6
        n_path = 50
        m.set_num_params(n_path=n_path, rn_seed=123456, n_sin=4, dt=None)

        n_step = 10000
        t_grid = np.arange(0, n_step + 1) / n_step
        for texp in (0.2, 1.0, 5.0, 10.0):
            zn = np.random.normal(size=(n_sin + 1, n_path))
            vol_path = m.vol_path_sin(t_grid * texp, zn)

            avgvol_simp = np.trapz(vol_path, dx=1, axis=0) / n_step
            avgvar_simp = np.trapz(vol_path**2, dx=1, axis=0) / n_step
            sig_t, avgvar, avgvol = m.cond_states_step(texp, m.sigma, zn=zn)

            assert np.all(np.isclose(1, vol_path[-1, :]/sig_t))
            assert np.all(np.isclose(1, avgvol_simp/avgvol))
            assert np.all(np.isclose(1, avgvar_simp/avgvar))

    def test_MomentsIntVariance(self):
        """
        Unconditional mean/var == E(conditional)
        """
        m = pf.OusvMcChoi2023KL(sigma=1, vov=0.75, mr=2.5)
        zz, ww = spsp.roots_hermitenorm(31)
        ww /= np.sqrt(2 * np.pi)

        for texp in (1.0, 3.0, 5.0):
            for sigma0 in m.sigma * np.array([0.5, 1, 2]):
                ### Unconditional mean of avgvol and avgvar
                mvol, _ = m.avgvol_mv(texp, sigma0)
                mvar, _ = m.avgvar_mv(texp, sigma0)

                ### Weighted average of conditional mean of avgvol and avgvar
                sig_t = m.vol_step(texp, sigma0, zz)
                cvol, cvar = m.cond_avgvolvar_m(texp, sigma0, sig_t)

                np.testing.assert_allclose(mvol, np.sum(cvol * ww))
                np.testing.assert_allclose(mvar, np.sum(cvar * ww))

                mvol, _ = m.avgvol_mv(texp, sigma0 - m.theta, nz_theta=True)
                mvar, _ = m.avgvar_mv(texp, sigma0 - m.theta, nz_theta=True)

                sig_t = m.vol_step(texp, sigma0 - m.theta, zz, nz_theta=True)
                cvol, cvar = m.cond_avgvolvar_m(texp, sigma0 - m.theta, sig_t, nz_theta=True)

                np.testing.assert_allclose(mvol, np.sum(cvol * ww))
                np.testing.assert_allclose(mvar, np.sum(cvar * ww))


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
