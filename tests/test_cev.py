import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestCevGreeks(unittest.TestCase):
    """
    Consistency checks: analytic Greeks vs. numeric Greeks for the CEV model.
    Tests both β < 1 (absorbing boundary) and β > 1 cases, with and without
    rates/dividends, and for both call and put options.
    """

    # Tolerances for analytic vs. numeric Greeks.
    # Central-difference error is O(h²·d³C/dS³), so short-expiry / large-curvature
    # cases need a wider band.  The default step is h = spot*0.001 for delta/gamma
    # and h = 0.001 for vega.
    DELTA_TOL = 5e-4
    GAMMA_TOL = 5e-4
    VEGA_TOL  = 5e-4

    # ------------------------------------------------------------------ helpers

    def _check_greeks(self, m, strike, spot, texp, cp, *, delta_tol=None, gamma_tol=None, vega_tol=None):
        delta_tol = delta_tol or self.DELTA_TOL
        gamma_tol = gamma_tol or self.GAMMA_TOL
        vega_tol  = vega_tol  or self.VEGA_TOL

        delta1 = m.delta(strike=strike, spot=spot, texp=texp, cp=cp)
        delta2 = m.delta_numeric(strike=strike, spot=spot, texp=texp, cp=cp)
        self.assertAlmostEqual(delta1, delta2, delta=delta_tol,
                               msg=f"delta mismatch: analytic={delta1:.6g}, numeric={delta2:.6g}")

        gamma1 = m.gamma(strike=strike, spot=spot, texp=texp, cp=cp)
        gamma2 = m.gamma_numeric(strike=strike, spot=spot, texp=texp, cp=cp)
        self.assertAlmostEqual(gamma1, gamma2, delta=gamma_tol,
                               msg=f"gamma mismatch: analytic={gamma1:.6g}, numeric={gamma2:.6g}")

        vega1 = m.vega(strike=strike, spot=spot, texp=texp, cp=cp)
        vega2 = m.vega_numeric(strike=strike, spot=spot, texp=texp, cp=cp)
        self.assertAlmostEqual(vega1, vega2, delta=vega_tol,
                               msg=f"vega mismatch: analytic={vega1:.6g}, numeric={vega2:.6g}")

    # ------------------------------------------------------------------ fixed cases

    def test_beta_lt1_atm(self):
        """ATM, β = 0.5, no rates."""
        m = pf.Cev(sigma=0.4, beta=0.5)
        self._check_greeks(m, strike=100, spot=100, texp=1.0, cp=1)
        self._check_greeks(m, strike=100, spot=100, texp=1.0, cp=-1)

    def test_beta_lt1_otm_itm(self):
        """OTM and ITM, β = 0.5, with rates and dividends."""
        m = pf.Cev(sigma=0.3, beta=0.5, intr=0.05, divr=0.02)
        for strike in [80, 90, 110, 120]:
            for cp in (1, -1):
                self._check_greeks(m, strike=strike, spot=100, texp=1.0, cp=cp)

    def test_beta_lt1_short_texp(self):
        """Short expiry, β = 0.3.
        Large gamma at short texp inflates the O(h²) FD error; wider tolerance needed."""
        m = pf.Cev(sigma=0.5, beta=0.3, intr=0.03)
        for strike in [95, 100, 105]:
            self._check_greeks(m, strike=strike, spot=100, texp=0.1, cp=1,
                               delta_tol=2e-3, gamma_tol=2e-3)

    def test_beta_lt1_long_texp(self):
        """Long expiry, β = 0.7."""
        m = pf.Cev(sigma=0.2, beta=0.7, intr=0.02, divr=0.01)
        for strike in [80, 100, 120]:
            self._check_greeks(m, strike=strike, spot=100, texp=5.0, cp=1)

    def test_beta_near_zero(self):
        """β close to 0 (near-Normal)."""
        m = pf.Cev(sigma=20.0, beta=0.05)
        for strike in [90, 100, 110]:
            self._check_greeks(m, strike=strike, spot=100, texp=1.0, cp=1,
                               vega_tol=1e-3)

    def test_beta_near_one(self):
        """β close to 1 (near-lognormal)."""
        m = pf.Cev(sigma=0.2, beta=0.95)
        for strike in [90, 100, 110]:
            self._check_greeks(m, strike=strike, spot=100, texp=1.0, cp=1)

    def test_is_fwd(self):
        """is_fwd=True: spot is treated as the forward price."""
        m = pf.Cev(sigma=0.3, beta=0.5, intr=0.05, divr=0.02, is_fwd=True)
        for strike in [90, 100, 110]:
            self._check_greeks(m, strike=strike, spot=100, texp=1.0, cp=1)

    # ------------------------------------------------------------------ random sweep

    def test_random_params_beta_lt1(self):
        """Random parameters, β < 1, 200 cases."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            spot   = rng.uniform(80, 120)
            strike = rng.uniform(70, 140)
            sigma  = rng.uniform(0.1, 2.0)
            beta   = rng.uniform(0.05, 0.95)
            texp   = rng.uniform(0.1, 5.0)
            intr   = rng.uniform(0.0, 0.1)
            divr   = rng.uniform(0.0, 0.1)
            cp     = 1 if rng.random() > 0.5 else -1
            is_fwd = rng.random() > 0.5

            m = pf.Cev(sigma=sigma, beta=beta, intr=intr, divr=divr, is_fwd=is_fwd)

            # Skip deep OTM/ITM where price ≈ 0 and numerics are unreliable
            fwd = m.forward(spot, texp)
            price = m.price(strike, spot, texp, cp)
            if price < 1e-6:
                continue

            with self.subTest(spot=spot, strike=strike, sigma=sigma, beta=beta,
                              texp=texp, intr=intr, divr=divr, cp=cp, is_fwd=is_fwd):
                self._check_greeks(m, strike=strike, spot=spot, texp=texp, cp=cp)


if __name__ == "__main__":
    print(f"Pyfeng loaded from {pf.__path__}")
    unittest.main()
