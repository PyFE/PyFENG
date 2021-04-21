import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv


class OusvIft(sv.SvABC):
    """
    The implementation of Schobel & Zhu (1998)'s inverse FT pricing formula for European
    options the Ornstein-Uhlenbeck driven stochastic volatility process.

    References: Schöbel, R., & Zhu, J. (1999). Stochastic Volatility With an Ornstein–Uhlenbeck Process: An Extension. Review of Finance, 3(1), 23–46. https://doi.org/10.1023/A:1009803506170

    Examples:
        >>> import pyfeng as pf
        >>> model = pf.OusvIft(0.2, mr=4, vov=0.1, rho=-0.7, intr=0.09531)
        >>> model.price(100, 100, texp=np.array([1, 5, 10]))
        array([13.21493, 40.79773, 62.76312])
        >>> model = pf.OusvIft(0.25, mr=8, vov=0.3, rho=-0.6, intr=0.09531)
        >>> model.price(np.array([90, 100, 110]), 100, texp=1)
        array([21.41873, 15.16798, 10.17448])
    """

    def D_B_C(self, s1, s2, s3, texp):
        # implement the formula for D(t,T), B(t,T), C(t,T) in paper appendix
        mr, theta, vov = self.mr, self.theta, self.vov

        gamma1 = np.sqrt(2*vov**2*s1 + mr**2)
        gamma2 = (mr - 2*vov**2*s3)/gamma1
        gamma3 = mr**2*theta - s2*vov**2
        sinh = np.sinh(gamma1*texp)
        cosh = np.cosh(gamma1*texp)
        sincos = sinh + gamma2*cosh
        cossin = cosh + gamma2*sinh
        ktg3 = mr*theta*gamma1 - gamma2*gamma3
        s2g3 = vov**2*gamma1**3

        D_t_T = (mr - gamma1*sincos/cossin) / vov**2
        B_t_T = ((ktg3+gamma3*sincos)/cossin - mr*theta*gamma1) / (vov**2*gamma1)
        C_t_T = -0.5*np.log(cossin)+0.5*mr*texp+((mr*theta*gamma1)**2-gamma3**2) \
                / (2*s2g3)*(sinh/cossin-gamma1*texp)+ktg3*gamma3/s2g3*((cosh-1)/cossin)

        return D_t_T, B_t_T, C_t_T

    def f_1(self, phi, fwd, texp):
        # implement the formula (11) in paper
        mr, theta, vov, rho = self.mr, self.theta, self.vov, self.rho

        s1 = -0.5*(1+1j*phi)**2*(1-rho**2) + 0.5*(1+1j*phi)*(1-2*mr*rho/vov)
        s2 = (1+1j*phi)*mr*theta*rho/vov
        s3 = 0.5*(1+1j*phi)*rho/vov

        D, B, C = self.D_B_C(s1, s2, s3, texp)
        res = np.exp(1j*phi*np.log(fwd) - 0.5*rho*(1+1j*phi)*(self.sigma**2/vov+vov*texp)) * \
              np.exp(0.5*D*self.sigma**2 + B*self.sigma+C)
        return res

    def f_2(self, phi, fwd, texp):
        # implement the formula (12) in paper
        mr, theta, vov, rho = self.mr, self.theta, self.vov, self.rho

        s1 = 0.5*phi**2*(1-rho**2)+0.5*(1j*phi)*(1-2*mr*rho/vov)
        s2 = (1j*phi)*mr*theta*rho/vov
        s3 = 0.5*(1j*phi)*rho/vov
        D, B, C = self.D_B_C(s1, s2, s3, texp)
        res = np.exp(1j*phi*np.log(fwd) - 0.5*1j*phi*rho*(self.sigma**2/vov + vov*texp)) * \
              np.exp(0.5*D*self.sigma**2 + B*self.sigma + C)
        return res

    def price(self, strike, spot, texp, cp=1):
        # implement the formula (13) and (14) in paper
        fwd, df, _ = self._fwd_factor(spot, texp)

        log_k = np.log(strike)
        J, h = 100001, 0.001
        phi = (np.arange(J)[:, None] + 1)*h  # shape=(J,1)
        ff1 = 0.5 + 1/np.pi*scint.simps((self.f_1(phi, fwd, texp)*np.exp(-1j*phi*log_k)/(1j*phi)).real, dx=h, axis=0)
        ff2 = 0.5 + 1/np.pi*scint.simps((self.f_2(phi, fwd, texp)*np.exp(-1j*phi*log_k)/(1j*phi)).real, dx=h, axis=0)

        price = np.where(cp > 0, fwd*ff1 - strike*ff2, strike*(1-ff2) - fwd*(1-ff1))
        if len(price) == 1:
            price = price[0]

        return df*price
