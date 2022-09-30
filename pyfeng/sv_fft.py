import numpy as np
import abc
import scipy.fft as spfft
import scipy.special as scsp
import scipy.interpolate as spinterp
import scipy.integrate as spint
import functools
from . import opt_abc as opt
from . import sv_abc as sv
from . import ousv
from . import heston

class FftABC(opt.OptABC, abc.ABC):
    n_x = 2**12  # number of grid. power of 2 for FFT
    x_lim = 200  # integratin limit

    @abc.abstractmethod
    def mgf_logprice(self, xx, texp):
        """
        Moment generating function (MGF) of log price. (forward = 1)

        Args:
            xx: dummy variable
            texp: time to expiry

        Returns:
            MGF value at xx
        """
        return NotImplementedError

    def charfunc_logprice(self, x, texp):
        """
        Characteristic function of log price

        Args:
            x:
            texp:

        Returns:

        """
        return self.mgf_logprice(1j*x, texp)

    def price(self, strike, spot, texp, cp=1):
        fwd, df, divf = self._fwd_factor(spot, texp)

        kk = strike/fwd
        log_kk = np.log(kk)

        dx = self.x_lim/self.n_x
        xx = np.arange(self.n_x + 1)[:, None]*dx  # the final value x_lim is excluded
        yy = (np.exp(-log_kk*xx*1j)*self.mgf_logprice(xx*1j + 0.5, texp)).real/(xx**2 + 0.25)
        int_val = spint.simpson(yy, dx=dx, axis=0)
        if np.isscalar(kk):
            int_val = int_val[0]
        price = np.where(cp > 0, 1, kk) - np.sqrt(kk)/np.pi*int_val
        return df*fwd*price

    @functools.lru_cache(maxsize=16)
    def fft_interp(self, texp, *args, **kwargs):
        """ FFT method based on the Lewis expression

        References:
            https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/1.3%20Fourier%20transform%20methods.ipynb
        """
        dx = self.x_lim/self.n_x
        xx = np.arange(self.n_x)*dx  # the final value x_lim is excluded

        weight = np.ones(self.n_x)  # Simpson weights
        weight[1:-1:2] = 4
        weight[2:-1:2] = 2
        weight *= dx/3

        dk = 2*np.pi/self.x_lim
        b = self.n_x*dk/2
        ks = -b + dk*np.arange(self.n_x)

        integrand = np.exp(-1j*b*xx)*self.mgf_logprice(xx*1j + 0.5, texp)/(xx**2 + 0.25)*weight
        # CF: integrand = np.exp(-1j*b*xx)*self.cf(xx - 0.5j, texp)*1/(xx**2 + 0.25)*weight
        integral_value = (self.n_x/np.pi)*spfft.ifft(integrand).real

        obj = spinterp.interp1d(ks, integral_value, kind='cubic')
        return obj

    def price_fft(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        kk = strike/fwd
        log_kk = np.log(kk)

        # self.params_hash(), self.n_x, self.x_lim are only used for cache key
        spline_cub = self.fft_interp(texp, k1=self.params_hash(), k2=self.n_x, k3=self.x_lim)
        price = np.where(cp > 0, 1, kk) - np.sqrt(kk)*spline_cub(-log_kk)
        return df*fwd*price


class BsmFft(FftABC):
    """
    Option pricing under Black-Scholes-Merton (BSM) model using fast fourier transform (FFT).

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmFft(sigma=0.2, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([15.71362027,  9.69251556,  5.52948647,  2.94558375,  1.4813909 ])
    """

    def mgf_logprice(self, uu, texp):
        val = -0.5*self.sigma**2*texp*uu*(1 - uu)
        # CF: val = -0.5 * self.sigma**2 * texp * uu * (1j + uu)
        return np.exp(val)


class VarGammaFft(sv.SvABC, FftABC):

    def mgf_logprice(self, uu, texp):
        volvar = self.vov*self.sigma**2
        mu = np.log(1 - self.theta*self.vov - 0.5*volvar)  # /self.vov
        # CF: rv = 1j*mu*uu - np.log(1 + (-1j*self.theta*self.vov + 0.5*volvar*uu)*uu)
        rv = mu*uu - np.log(1 + (-self.theta*self.vov - 0.5*volvar*uu)*uu)
        np.exp(texp/self.vov*rv, out=rv)
        return rv


class ExpNigFft(sv.SvABC, FftABC):

    def mgf_logprice(self, uu, texp):
        """

        Args:
            uu:
            texp:

        Returns:

        """
        volvar = self.vov*self.sigma**2
        mu = -1 + np.sqrt(1 - 2*self.theta*self.vov - volvar)
        rv = mu*uu + 1 - np.sqrt(1 + (-2*self.theta*self.vov - volvar*uu)*uu)
        # CF: rv = 1j*mu*uu + 1 - np.sqrt(1 + (-2j*self.theta*self.vov + volvar*uu)*uu)
        np.exp(texp/self.vov*rv, out=rv)
        return rv


class HestonFft(heston.HestonABC, FftABC):
    """
    Heston model option pricing with FFT

    References:
        - Lewis AL (2000) Option valuation under stochastic volatility: with Mathematica code. Finance Press

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.array([60, 70, 100, 140])
        >>> sigma, vov, mr, rho, texp, spot = 0.04, 1, 0.5, -0.9, 10, 100
        >>> m = pf.HestonFft(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.32997507, 35.8497697, 13.08467014, 0.29577444
        array([44.32997507, 35.8497697 , 13.08467014,  0.29577444])
    """

    def mgf_logprice(self, uu, texp):
        """
        Log price MGF under the Heston model.
        We use the characteristic function in Eq (2.8) of Lord & Kahl (2010) that is
        continuous in branch cut when complex log is evaluated.

        References:
            - Heston SL (1993) A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options. The Review of Financial Studies 6:327–343. https://doi.org/10.1093/rfs/6.2.327
            - Lord R, Kahl C (2010) Complex Logarithms in Heston-Like Models. Mathematical Finance 20:671–694. https://doi.org/10.1111/j.1467-9965.2010.00416.x
        """
        var_0 = self.sigma
        vov2 = self.vov**2

        beta = self.mr - self.vov*self.rho*uu
        dd = np.sqrt(beta**2 + vov2*uu*(1 - uu))
        gg = (beta - dd)/(beta + dd)
        exp = np.exp(-dd*texp)
        tmp1 = 1 - gg*exp

        mgf = self.mr*self.theta*((beta - dd)*texp - 2*np.log(tmp1/(1 - gg))) + var_0*(beta - dd)*(1 - exp)/tmp1
        return np.exp(mgf/vov2)


class OusvFft(ousv.OusvABC, FftABC):
    """
    OUSV model option pricing with FFT

    """

    def mgf_logprice(self, uu, texp):
        """
        Log price MGF under the OUSV model.
        We use the characteristic function in Eq (4.14) of Lord & Kahl (2010) that is
        continuous in branch cut when complex log is evaluated.

        Returns:
            MGF value at uu

        References:
            - Lord R, Kahl C (2010) Complex Logarithms in Heston-Like Models. Mathematical Finance 20:671–694. https://doi.org/10.1111/j.1467-9965.2010.00416.x
        """

        var_0 = self.sigma**2
        sigma_0 = self.sigma

        # equivalent Heston params when theta=0
        mr_h, vov_h, theta_h = 2*self.mr, 2*self.vov, self.vov**2/(2*self.mr)
        vov2_h = 4*self.vov**2

        beta = mr_h - self.rho*vov_h*uu
        dd = np.sqrt(beta**2 + vov2_h*uu*(1 - uu))
        gg = (beta - dd)/(beta + dd)

        exp_h = np.exp(-0.5*dd*texp)
        exp = exp_h**2
        tmp1 = 1 - gg*exp

        # Heston model part
        mgf = mr_h*theta_h*((beta - dd)*texp - 2*np.log(tmp1/(1 - gg))) + var_0*(beta - dd)*(1 - exp)/tmp1
        mgf /= vov2_h

        # Additional part for OUSV
        bb = (1 - exp_h)**2/tmp1
        aa = 0.5*self.mr*self.theta/dd**2
        aa *= beta*(dd*texp - 4) + dd*(dd*texp - 2) + 4*((dd**2 - 2*beta**2)/(beta + dd)*exp + 2*beta*exp_h)/tmp1

        mgf += 0.5*self.theta/theta_h*(beta - dd)/dd*(aa + bb*sigma_0)

        return np.exp(mgf)

    def mgf_logprice_schobelzhu1998(self, uu, texp):
        """
        MGF from Eq. (13) in Schobel & Zhu (1998).
        This form suffers discontinuity in complex log branch cut. Should not be used for pricing.

        Args:
            uu: dummy variable
            texp: time to expiry

        Returns:
            MGF value at uu

        References:
            - Schöbel R, Zhu J (1999) Stochastic Volatility With an Ornstein–Uhlenbeck Process: An Extension. Rev Financ 3:23–46. https://doi.org/10.1023/A:1009803506170

        """
        mr, theta, vov, rho = self.mr, self.theta, self.vov, self.rho

        # CF: s1 = 0.5*uu*(uu*(1 - rho**2) + 1j*(1 - 2*mr*rho/vov))
        # CF: s2 = 1j*uu*mr*theta*rho/vov
        # CF: s3 = 0.5j*uu*rho/vov
        s1 = 0.5*uu*((1 - 2*mr*rho/vov) - (1 - rho**2)*uu)
        s2 = uu*mr*theta*rho/vov
        s3 = 0.5*uu*rho/vov

        gamma1 = np.sqrt(2*vov**2*s1 + mr**2)
        gamma2 = (mr - 2*vov**2*s3)/gamma1
        gamma3 = mr**2*theta - s2*vov**2
        sinh = np.sinh(gamma1*texp)
        cosh = np.cosh(gamma1*texp)
        sincos = sinh + gamma2*cosh
        cossin = cosh + gamma2*sinh
        ktg3 = mr*theta*gamma1 - gamma2*gamma3
        s2g3 = vov**2*gamma1**3

        D = (mr - gamma1*sincos/cossin)/vov**2
        B = ((ktg3 + gamma3*sincos)/cossin - mr*theta*gamma1)/(
                vov**2*gamma1
        )
        C = (
                -0.5*np.log(cossin)
                + 0.5*mr*texp
                + ((mr*theta*gamma1)**2 - gamma3**2)
                /(2*s2g3)
                *(sinh/cossin - gamma1*texp)
                + ktg3*gamma3/s2g3*((cosh - 1)/cossin)
        )

        # CF: res = -0.5*1j*uu*rho*(self.sigma**2/vov + vov*texp)
        res = -0.5*uu*rho*(self.sigma**2/vov + vov*texp)
        res += (D/2*self.sigma + B)*self.sigma + C
        return np.exp(res)


class Sv32Fft(sv.SvABC, FftABC):
    """
    3/2 model option pricing with Fourier inversion

    References:
        - Lewis AL (2000) Option valuation under stochastic volatility: with Mathematica code. Finance Press

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> sigma, mr, theta, vov, rho = 0.06, 20.48, 0.218, 3.20, -0.99
        >>> strike, spot, texp = np.array([95, 100, 105]), 100, 0.5
        >>> m = pf.Sv32Fft(sigma, vov=vov, mr=mr, rho=rho, theta=theta)
        >>> m.price(strike, spot, texp)
        array([11.7235,  8.9978,  6.7091])
    """

    @staticmethod
    def hyp1f1_complex(a, b, x):
        """
        Confluent hypergeometric function 1F1 (scipy.special.hyp1f1) taking complex values of a and b

        Args:
            a: parameter (real or complex)
            b: parameter (real or complex)
            x: argument (real or complex)

        Returns:
            function value

        References:
            - https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html#scipy.special.hyp1f1
        """
        inc = a / b * x
        ret = 1 + inc

        for kk in np.arange(1, 1024):
            inc *= (a + kk) / (b + kk) / (kk + 1) * x
            ret += inc

        return ret

    def mgf_logprice(self, uu, texp):
        """
        Log price MGF under the 3/2 SV model from Lewis (2000) or Carr & Sun (2007).

        In the formula in Lewis (2000, p54), ik should be replaced by -ik.

        References:
            - Eq. (73)-(75) in Carr P, Sun J (2007) A new approach for option pricing under stochastic volatility. Rev Deriv Res 10:87–150. https://doi.org/10.1007/s11147-007-9014-6
            - p 54 in Lewis AL (2000) Option valuation under stochastic volatility: With Mathematica code. Finance Press, Newport Beach, CA
        """
        vov2 = self.vov**2

        mu = 0.5 + (self.mr - uu*self.rho*self.vov)/vov2
        c_tilde = uu*(1 - uu)/vov2
        delta = np.sqrt(mu**2 + c_tilde)
        alpha = -mu + delta
        beta = 1 + 2*delta

        mr_new = self.mr * self.theta
        XX = 2*mr_new/(self.vov**2 * self.sigma)/(np.exp(mr_new * texp) - 1)
        ret = scsp.gamma(beta - alpha) * scsp.rgamma(beta) * np.power(XX, alpha) * self.hyp1f1_complex(alpha, beta, -XX)

        # log version
        # when beta is too big, scsp.gamma(beta) goes to infty.
        #ret = np.exp(scsp.loggamma(beta - alpha) - scsp.loggamma(beta) + alpha * np.log(XX)) * self.hyp1f1_complex(alpha, beta, -XX)

        return ret
