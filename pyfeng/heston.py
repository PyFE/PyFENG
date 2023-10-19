import abc
import numpy as np
from . import sv_abc as sv
from . import bsm


class HestonABC(sv.SvABC, abc.ABC):
    model_type = "Heston"
    var_process = True

    def chi_dim(self):
        """
        Noncentral Chi-square (NCX) distribution's degree of freedom

        Returns:
            degree of freedom (scalar)
        """
        chi_dim = 4 * self.theta * self.mr / self.vov**2
        return chi_dim

    def chi_lambda(self, dt):
        """
        Noncentral Chi-square (NCX) distribution's noncentrality parameter

        Returns:
            noncentrality parameter (scalar)
        """
        chi_lambda = 4 * self.sigma * self.mr / self.vov**2 / np.expm1(self.mr*dt)
        return chi_lambda

    def phi_exp(self, texp):
        exp = np.exp(-self.mr*texp/2)
        phi = 4*self.mr / self.vov**2 / (1/exp - exp)
        return phi, exp

    def var_mv(self, dt, var0=None):
        """
        Mean and variance of the variance V(t+dt) given V(0) = var_0

        Args:
            var0: initial variance
            dt: time step

        Returns:
            mean, variance
        """
        if var0 is None:
            var0 = self.sigma

        e_mr = np.exp(-self.mr*dt)
        m = self.theta + (var0 - self.theta)*e_mr
        s2 = var0*e_mr + self.theta*(1 - e_mr)/2
        s2 *= self.vov**2*(1 - e_mr)/self.mr
        return m, s2

    def avgvar_mv(self, texp, var0=None):
        """
        Mean and variance of the average variance given V(0) = var_0.
        Appnedix B in Ball & Roma (1994)

        Args:
            var0: initial variance
            texp: time step

        Returns:
            mean, variance
        """

        ### We take var0 as argument instead of using self.sigma
        ### because it is used in var_step_qe method in HestonMcAndersen2008

        if var0 is None:
            var0 = self.sigma

        mr_t = self.mr*texp
        e_mr = np.exp(-mr_t)
        phi = (1 - e_mr)/mr_t
        x0 = var0 - self.theta
        mean = self.theta + x0*phi
        var = (self.theta - 2*x0*e_mr) + (var0 - 2.5*self.theta + (var0 - self.theta/2)*e_mr)*phi
        var *= (self.vov/mr_t)**2 * texp
        return mean, var

    def strike_var_swap_analytic(self, texp, dt):
        """
        Analytic fair strike of variance swap. Eq (11) in Bernard & Cui (2014)

        Args:
            texp: time to expiry
            dt: observation time step (e.g., dt=1/12 for monthly) For continuous monitoring, set dt=0

        Returns:
            Fair strike

        References:
            - Bernard C, Cui Z (2014) Prices and Asymptotics for Discrete Variance Swaps. Applied Mathematical Finance 21:140–173. https://doi.org/10.1080/1350486X.2013.820524

        """

        var0 = self.sigma

        ### continuously monitored fair strike (same as mean of avgvar)
        mr_t = self.mr*texp
        e_mr_t = np.exp(-mr_t)
        x0 = var0 - self.theta
        strike = self.theta + x0*(1 - e_mr_t)/mr_t

        if not np.all(np.isclose(dt, 0.0)):
            ### adjustment for discrete monitoring
            mr_h = self.mr * dt
            e_mr_h = np.exp(-mr_h)

            tmp = self.theta - 2*(self.intr - self.divr)
            strike += tmp*dt/4 * (tmp + 2*x0*(1 - e_mr_t)/mr_t)

            tmp = self.vov / self.mr
            strike += self.theta * tmp * (tmp/4 - self.rho) * (1 - (1-e_mr_h)/mr_h)
            strike += x0 * tmp * (tmp/2 - self.rho) * (1 - e_mr_t)/mr_t * (1 + mr_h/(1 - 1/e_mr_h))
            strike += (tmp**2*(self.theta - 2*var0) + 2*x0**2/self.mr) * (1 - e_mr_t**2)/(8*mr_t) * (1-e_mr_h)/(1+e_mr_h)

        return strike


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

        avgvar, var = self.avgvar_mv(texp)

        m_bs = bsm.Bsm(np.sqrt(avgvar), intr=self.intr, divr=self.divr)
        price = m_bs.price(strike, spot, texp, cp)

        if self.order == 2:
            price += 0.5*var*m_bs.d2_var(strike, spot, texp, cp)
        elif self.order > 2:
            raise ValueError(f"Not implemented for approx order: {self.order}")

        return price
