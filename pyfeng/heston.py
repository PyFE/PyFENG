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

    def strike_var_swap_analytic_pctreturn(self, texp, dt):
        """
        Analytic fair strike of variance swap. Proposition3.2 in Bernard & Cui (2014)

        Args:
            texp: time to expiry
            dt: observation time step (e.g., dt=1/12 for monthly) For continuous monitoring, set dt=0
                must in the form of 1/n for some integer n

        Returns:
            Fair strike

        References:
            - Bernard C, Cui Z (2014) Prices and Asymptotics for Discrete Variance Swaps. Applied Mathematical Finance 21:140–173. https://doi.org/10.1080/1350486X.2013.820524

        """
        n = int(texp/dt)
        delta = dt
        kappa = self.mr
        theta = self.theta
        gamma = self.vov
        rho = self.rho
        V0 = self.sigma
        r = self.intr
        S0 = 1

        
        alpha = 2*kappa*theta/gamma**2-1

        def d(u):
            res = np.sqrt((kappa-gamma*rho*u)**2 + (u - u*2)*gamma**2)
            return res

        def g(u):
            res = (kappa - gamma*rho*u - d(u))/(kappa - gamma*rho*u + d(u))
            return res

        def q(u):
            res = ((kappa - gamma*rho*u - d(u))/(gamma**2))*(1 - np.exp(-d(u)*delta))/(1 - g(u)*np.exp(-d(u)*delta))
            return res

        def eta(u):
            res = (2*kappa)/((gamma**2)*(1 - np.exp(-u*kappa)))
            return res

        def M(u,t):
            e1 = np.exp((kappa*theta/gamma**2)*((kappa - gamma*rho*u - d(u))*t - 2*np.log((1 - g(u)*np.exp(-d(u)*t))/(1 - g(u)))))
            e2 = np.exp(V0*(kappa-gamma*rho*u-d(u))*(1 - np.exp(-d(u)*t))/(gamma**2*(1 - g(u)*np.exp(-d(u)*t))))
            res = (S0**u)*e1*e2
            return res

        def ai(ti):
            e1 = np.exp(q(2)*V0*(eta(ti)*np.exp(-kappa*ti)/(eta(ti)-q(2))-1))
            e2 = (eta(ti)/(eta(ti)-q(2)))**(alpha+1)
            res = np.exp(2*r*delta)*M(2, delta)*e1*e2/(S0**2)
            return res

        def K(n):
            res = 0
            a = np.exp(2*r*delta)*M(2, delta)/S0**2
            res += a
            for i in range(1,n):
                ti = i*delta
                res += ai(ti)
            res = res/texp + (n-2*n*np.exp(r*delta))/texp
            return res

        return K(n)

    def strike_var_swap_analytic_ZhuLian(self, texp, dt):
        """
        Analytic fair strike of variance swap. eq (2.34) Lian & Zhu (2011)

        Args:
            texp: time to expiry
            dt: observation time step (e.g., dt=1/12 for monthly) For continuous monitoring, set dt=0
                must in the form of 1/n for some integer n

        Returns:
            Fair strike

        References:
            - Song-Ping Zhu, Guang-Hua Lian (2011) A Closed-form Exact Solution for Pricing Variance Swaps with Stochastic Volatility

        """
        kappa = self.mr
        theta = self.theta
        rho = self.rho
        sigma_V = self.vov
        v0 = self.sigma
        r = self.intr
        N = int(texp/dt)
        T = texp

        def C_D_calculation():
            tilde_a = kappa - 2 * rho * sigma_V
            tilde_b = np.sqrt(tilde_a ** 2 - 2 * sigma_V ** 2)
            tilde_g = (tilde_a / sigma_V) ** 2 - 1 + (tilde_a / sigma_V) * np.sqrt((tilde_a / sigma_V) ** 2 - 2)
        
            term1 = r * dt
            term2 = (kappa * theta) / (sigma_V ** 2)
            term3 = (tilde_a + tilde_b) * dt
            term4 = 2 * np.log((1 - tilde_g * np.exp(tilde_b * dt)) / (1 - tilde_g))
        
            C = term1 + term2 * (term3 - term4)

            D = ((tilde_a + tilde_b) / sigma_V ** 2) * ((1 - np.exp(tilde_b * dt)) / (1 - tilde_g * np.exp(tilde_b * dt)))
            return C, D
        
        def f():
            C, D = C_D_calculation()
            return np.exp(C + D * v0) + np.exp(-r * dt) - 2

        def sum_fi():
            C, D = C_D_calculation()
            sum = 0
            for i in range(2, N + 1):
                c_i = 2 * kappa / (sigma_V ** 2 * (1 - np.exp(-kappa * (i - 1) * dt)))
                term1 = np.exp(C + c_i * np.exp(-kappa * (i - 1) * dt) / (c_i - D) * D * v0)
                term2 = (c_i / (c_i - D)) ** (2 * kappa * theta / (sigma_V ** 2))
                sum += term1 * term2 + np.exp(-r * dt) - 2
            return sum
    
        
        f_v0 = f()
        sum_fi_v0 = sum_fi()
        K_var = (np.exp(r * dt) / T) * (f_v0 + sum_fi_v0) 
        return K_var


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
