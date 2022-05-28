import abc
import numpy as np
import scipy.stats as spst
from . import sv_abc as sv
from . import bsm


class HestonABC(sv.SvABC, abc.ABC):
    model_type = "Heston"

    def var_mv(self, var0, dt):
        """
        Mean and variance of the variance V(t+dt) given V(0) = var_0

        Args:
            var0: initial variance
            dt: time step

        Returns:
            mean, variance
        """

        expo = np.exp(-self.mr * dt)
        m = self.theta + (var0 - self.theta)*expo
        s2 = var0*expo + self.theta*(1 - expo)/2
        s2 *= self.vov**2 * (1 - expo) / self.mr
        return m, s2

    def avgvar_mv(self, var0, texp):
        """
        Mean and variance of the average variance given V(0) = var_0.
        Appnedix B in Ball & Roma (1994)

        Args:
            var0: initial variance
            texp: time step

        Returns:
            mean, variance
        """

        mr_t = self.mr * texp
        e_mr = np.exp(-mr_t)
        x0 = var0 - self.theta
        vovn = self.vov * np.sqrt(texp)  # normalized vov

        m = self.theta + x0 * (1 - e_mr)/mr_t

        var = (self.theta - 2*x0*e_mr) + \
              (var0 - 2.5*self.theta + (2*self.theta + (0.5*self.theta - var0)*e_mr)*e_mr)/mr_t
        var *= (vovn/mr_t)**2

        return m, var


class HestonUncorrBallRoma1994(HestonABC):
    """
    Ball & Roma (1994)'s approximation pricing formula for European options under uncorrelated (rho=0) Heston model.
    Up to 2nd order is implemented.

    See Also: OusvUncorrBallRoma1994, GarchUncorrBaroneAdesi2004
    """

    order = 2

    def price(self, strike, spot, texp, cp=1):

        if not np.isclose(self.rho, 0.0):
            print(f"Pricing ignores rho = {self.rho}.")

        avgvar, var = self.avgvar_mv(self.sigma, texp)

        m_bs = bsm.Bsm(np.sqrt(avgvar), intr=self.intr, divr=self.divr)
        price = m_bs.price(strike, spot, texp, cp)

        if self.order == 2:
            price += 0.5 * var * m_bs.d2_var(strike, spot, texp, cp)
        elif self.order > 2:
            raise ValueError(f"Not implemented for approx order: {self.order}")

        return price
