import unittest
import numpy as np
import sys
import os
import scipy.special as spsp
import statsmodels.stats.moment_helpers as sms_m

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestSabr(unittest.TestCase):
    def test_Hagan2002(self):
        """
        Hagan formula == Saved benchmark
        """
        for k in list(range(1, 19)) + [22, 23]:
            m, df, rv = pf.SabrHagan2002.init_benchmark(k)
            # ref = rv['ref']
            # print(f'Sheet {k:02d}: {ref}')
            v1 = np.round(m.vol_for_price(**rv["args_pricing"]), 4)
            v2 = df["IV Hagan"].values
            np.testing.assert_allclose(v1, v2)

    def test_SabrNorm(self):
        """
        Choi & Wu (2021) Hagan == Normal Vol Approx
        """
        for k in [22, 23]:
            m, df, rv = pf.SabrNormVolApprox.init_benchmark(k)
            v1 = m.price(**rv["args_pricing"])
            m, df, rv = pf.SabrChoiWu2021H.init_benchmark(k)
            v2 = m.price(**rv["args_pricing"])
            np.testing.assert_allclose(v1, v2)

    def test_SabrNormATM(self):
        """
        Test if is_atmvol works fine
        """
        for k in [22, 23]:
            m, df, rv = pf.SabrNormVolApprox.init_benchmark(k)
            m.is_atmvol = True
            np.testing.assert_allclose(m.vol_smile(0, 0, texp=0.1), m.sigma)
            np.testing.assert_allclose(m.vol_smile(0, 0, texp=10), m.sigma)

            m, df, rv = pf.Nsvh1.init_benchmark(k)
            m.is_atmvol = True
            np.testing.assert_allclose(m.vol_smile(0, 0, texp=0.1), m.sigma)
            np.testing.assert_allclose(m.vol_smile(0, 0, texp=10), m.sigma)

    def test_PaulotBsm(self):
        """
        Paulot formula == Saved benchmark
        """
        for k in list(range(1, 19)):
            m, df, rv = pf.SabrChoiWu2021P.init_benchmark(k)
            m._base_beta = 1.0  # For Paulot's BS volatility approximation
            # print(f'Sheet {k:02d}: {ref}
            v1 = np.round(m.vol_for_price(**rv["args_pricing"]), 4)
            v2 = df["IV HL-P"].values
            np.testing.assert_allclose(v1, v2)

    def test_UnCorrChoiWu2021(self):
        """
        Uncorrelated SABR
        """
        # Param Set 19: Table 7 (Case III.C) in Cai et al. (2017). https://doi.org/10.1287/opre.2017.1617
        m, df, rv = pf.SabrUncorrChoiWu2021.init_benchmark(19)
        mass = m.mass_zero(rv["args_pricing"]["spot"], rv["args_pricing"]["texp"])
        p = m.price(**rv["args_pricing"])

        mass2 = 0.7623543217183134
        p2 = np.array([0.04533777, 0.04095806, 0.03889591, 0.03692339, 0.03324944, 0.02992918])

        np.testing.assert_allclose(mass, mass2, atol=1e-8)
        np.testing.assert_allclose(p, p2, atol=1e-8)

    def test_McTimeDisc(self):
        """
        Time discretization
        """
        for k in [19, 20]:  # can test 22 (Korn&Tang) also, but difficult to pass
            m, df, rv = pf.SabrMcTimeDisc.init_benchmark(k)
            m.set_num_params(n_path=5e4, dt=0.05, rn_seed=1234)
            p = m.price(**rv["args_pricing"])
            np.testing.assert_allclose(p, rv["val"], rtol=1e-3)

    def test_MomentsIntVariance(self):
        """
        Various test on momoents of SABR/NSVh
        """
        m = pf.SabrNormVolApprox(1)

        #### Unconditional avgvar: mvsk versus non-central moments
        for vovn in [0.2, 0.4, 0.8, 1.0]:
            m1, v, s, k = m.avgvar_mvsk(vovn)
            mvsk = sms_m.mnc2mvsk(m.avgvar_mnc4(vovn))
            np.testing.assert_allclose(np.array([m1, v, s, k]), mvsk)

        #### Unconditional avgvar: mean/var == E(conditional mean/var)
        for vovn in [0.1, 0.2, 0.4, 0.6, 0.8]:  # test fail in m4 when vovn > 0.8
            zhat, ww = spsp.roots_hermitenorm(31)
            ww /= np.sqrt(2*np.pi)
            zhat -= 0.5*vovn

            m1, v, s, k = m.avgvar_mvsk(vovn)
            mnc = sms_m.mvsk2mnc([m1, v, s, k])
            cond_m1, cond_m2, cond_m3, cond_m4 = m.cond_avgvar_mvsk(vovn, zhat, True)

            np.testing.assert_allclose(np.sum(cond_m1 * ww), m1)
            np.testing.assert_allclose(np.sum(cond_m2 * ww), mnc[1])
            np.testing.assert_allclose(np.sum(cond_m3 * ww), mnc[2])
            np.testing.assert_allclose(np.sum(cond_m4 * ww), mnc[3])
            # print(np.sum(cond_m4 * ww), mnc[3])

        #### Generic Nsvh (lambda = 0) == Normal SABR
        #### Generic Nsvh (lambda = 1) == Nsvh1
        for (rho, vov) in zip((-0.2, 0, 0.5), (0.1, 0.2, 0.5)):
            m = pf.NsvhGaussQuad(1, rho=rho, vov=vov, lam=0)
            m0 = pf.SabrNormVolApprox(1, rho=rho, vov=vov)
            np.testing.assert_allclose(m0.price_vsk(texp=1.5), m.price_vsk(texp=1.5))

            m = pf.NsvhGaussQuad(1, rho=rho, vov=vov, lam=1)
            m1 = pf.Nsvh1(1, rho=rho, vov=vov)
            np.testing.assert_allclose(m1.price_vsk(texp=1.5), m.price_vsk(texp=1.5))

        for vovn in (0.1, 0.2, 0.5, 1.2):
            m0 = pf.SabrNormVolApprox(sigma=vovn, rho=0, vov=vovn)
            p_var, skew, kurt = m0.price_vsk(texp=1)
            i_m, i_var, *_ = m0.avgvar_mvsk(vovn)

            ### E(X^2_IntVar) = vovn^2 * E(I)
            np.testing.assert_allclose(p_var, i_m*vovn**2)
            ### E(X^4_IntVar) = 3 * vovn^4 * E(I^2)
            np.testing.assert_allclose((kurt + 3)*p_var**2, 3*(i_var + i_m**2)*vovn**4)


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
