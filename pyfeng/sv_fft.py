import numpy as np
import abc
import scipy.fft as spfft
import scipy.interpolate as spinterp
from . import opt_abc as opt
from . import sv_abc as sv


class FftABC(opt.OptABC, abc.ABC):

    n_x = 2**12  # number of grid. power of 2 for FFT
    x_lim = 200  # integratin limit

    @abc.abstractmethod
    def cf(self, x, texp):
        return NotImplementedError

    def price(self, strike, spot, texp, cp=1):
        """ FFT method based on the Lewis expression

        References:
            https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/1.3%20Fourier%20transform%20methods.ipynb
        """
        fwd, df, divf = self._fwd_factor(spot, texp)

        kk = strike / fwd
        log_kk = np.log(kk)

        dx = self.x_lim / self.n_x
        xx = np.arange(self.n_x) * dx  # the final value x_lim is excluded

        weight = np.ones(self.n_x)  # Simpson weights
        weight[1:-1:2] = 4
        weight[2:-1:2] = 2
        weight *= dx/3

        dk = 2*np.pi / self.x_lim
        b = self.n_x*dk / 2
        ks = -b + dk*np.arange(self.n_x)

        integrand = np.exp(-1j*b*xx)*self.cf(xx - 0.5j, texp)*1/(xx**2 + 0.25)*weight
        integral_value = (self.n_x / np.pi) * spfft.ifft(integrand).real

        spline_cub = spinterp.interp1d(ks, integral_value, kind='cubic')
        price = np.where(cp > 0, 1, kk) - np.sqrt(kk)*spline_cub(-log_kk)
        return df * fwd * price


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
    def cf(self, uu, texp):
        val = -0.5 * self.sigma**2 * texp * uu * (1j + uu)
        return np.exp(val)


class HestonFft(sv.SvABC, FftABC):
    """
    Heston model option pricing with FFT

    References:
        - Lewis AL (2000) Option valuation under stochastic volatility: with Mathematica code. Finance Press

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.array([60, 100, 140])
        >>> spot = 100
        >>> sigma, vov, mr, rho, texp = 0.04, 1, 0.5, -0.9, 10
        >>> m = pf.HestonFft(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.price(strike, spot, texp)
        array([44.32997493, 13.08467014,  0.29573228])
    """

    model_type = "Heston"

    def cf(self, uu, texp):
        """
        Heston characteristic function as proposed by Schoutens (2004)

        References:
            - https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/1.4%20SDE%20-%20Heston%20model.ipynb
        """
        var_0 = self.sigma
        xi = self.mr - self.vov*self.rho*uu*1j
        dd = np.sqrt(xi**2 + self.vov**2 * uu * (1j + uu))
        g2 = (xi - dd) / (xi + dd)
        exp = np.exp(-dd*texp)
        cf = self.mr*self.theta*((xi - dd)*texp - 2*np.log((1 - g2*exp)/(1 - g2))) \
            + var_0*(xi - dd)*(1 - exp)/(1 - g2*exp)
        cf /= self.vov**2
        return np.exp(cf)
