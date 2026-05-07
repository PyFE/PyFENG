import numpy as np
import abc
import scipy.fft as spfft
import scipy.special as spsp
import scipy.interpolate as spinterp
import scipy.integrate as spint
import functools
from . import ousv
from . import heston
from . import rheston
from . import sv32
from .bsm import Bsm
from .opt_abc import OptABC
from .params import VarGammaParams, NigParams, GarchParams, CgmyParams

class FftABC(OptABC):
    n_x = 2**12  # number of grid. power of 2 for FFT
    x_lim = 200  # integratin limit

    @abc.abstractmethod
    def logp_mgf(self, uu, texp):
        """
        Moment generating function (MGF) of log price. (forward = 1)

        Args:
            uu: dummy variable
            texp: time to expiry

        Returns:
            MGF value at uu
        """
        raise NotImplementedError

    def logp_cf(self, x, texp):
        """
        Characteristic function of log price

        Args:
            x:
            texp:

        Returns:

        """
        return self.logp_mgf(1j*x, texp)

    def price(self, strike, spot, texp, cp=1):
        fwd, df, divf = self._fwd_factor(spot, texp)

        kk = strike/fwd
        log_kk = np.log(kk)

        dx = self.x_lim/self.n_x
        xx = np.arange(self.n_x + 1)[:, None]*dx  # the final value x_lim is excluded
        yy = (np.exp(-log_kk*xx*1j)*self.logp_mgf(xx*1j + 0.5, texp)).real/(xx**2 + 0.25)
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

        integrand = np.exp(-1j*b*xx)*self.logp_mgf(xx*1j + 0.5, texp)/(xx**2 + 0.25)*weight
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


class BsmFft(Bsm, FftABC):
    """
    Option pricing under Black-Scholes-Merton (BSM) model using fast fourier transform (FFT).

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmFft(sigma=0.2, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([15.71362027,  9.69251556,  5.52948647,  2.94558375,  1.4813909 ])
    """

    price_analytic = Bsm.price
    price = FftABC.price


class VarGammaABC(VarGammaParams, OptABC):
    """
    Abstract base for Variance Gamma models — provides ``logp_mgf``.

    Parameters, constraints, and the precomputed ``_mgf1_correction`` are
    defined in :class:`~pyfeng.params.VarGammaParams`.

    References:
        - Madan DB, Carr PP, Chang EC (1998) The Variance Gamma Process and Option
          Pricing. European Finance Review 2:79–105.
          https://doi.org/10.1023/A:1009703431535
        - Madan DB, Seneta E (1990) The Variance Gamma (V.G.) Model for Share
          Market Returns. Journal of Business 63:511–524.
          https://doi.org/10.1086/296519
    """

    def logp_mgf(self, uu, texp):
        """
        MGF of log price under the Variance Gamma model.

        The MGF of :math:`\\log(S_T/F_T)` at argument :math:`u` is

        .. math::

            \\exp\\!\\left(\\frac{T}{\\nu}\\left[
                u\\ln\\!\\left(1 - \\theta\\nu - \\tfrac{1}{2}\\sigma^2\\nu\\right)
                - \\ln\\!\\left(1 - \\theta\\nu u - \\tfrac{1}{2}\\sigma^2\\nu u^2\\right)
            \\right]\\right).

        ``self._mgf1_correction`` stores :math:`\\kappa_{\\mathrm{VG}}(1)\\nu = -\\omega\\nu`
        (the raw CGF at :math:`u=1`, scaled by :math:`\\nu`), so the correction
        term is ``-self._mgf1_correction * uu``.

        Args:
            uu: MGF argument (scalar or array). Real values give the MGF;
                purely imaginary values ``uu = 1j * xi`` give the characteristic
                function.
            texp: time to expiry.

        Returns:
            MGF value(s) at ``uu``, same shape as ``uu``.
        """
        volvar = self.nu * self.sigma**2
        # CF: rv = 1j*self._mgf1_correction*uu - np.log(1 + (-1j*self.theta*self.nu + 0.5*volvar*uu)*uu)
        rv = -self._mgf1_correction * uu - np.log1p((-self.theta * self.nu - 0.5 * volvar * uu) * uu)
        np.exp(texp/self.nu*rv, out=rv)
        return rv


class VarGammaFft(VarGammaABC, FftABC):
    """
    Variance Gamma (VG) model option pricing with FFT.

    Inherits ``logp_mgf`` from :class:`VarGammaABC`.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.arange(80, 121, 10)
        >>> m = pf.VarGammaFft(0.2, nu=0.3, theta=-0.1)
        >>> m.price(strike, 100, 1.0)
    """


class NigABC(NigParams, OptABC):
    """
    Abstract base for Normal Inverse Gaussian (NIG) models — provides ``logp_mgf``.

    Parameters, constraints, and the precomputed ``_mgf1_correction`` are
    defined in :class:`~pyfeng.params.NigParams`.

    References:
        - Barndorff-Nielsen OE (1997) Normal Inverse Gaussian Distributions and
          Stochastic Volatility Modelling. Scandinavian Journal of Statistics
          24:1–13. https://doi.org/10.1111/1467-9469.00045
        - Barndorff-Nielsen OE (1998) Processes of Normal Inverse Gaussian Type.
          Finance and Stochastics 2:41–68.
          https://doi.org/10.1007/s007800050032
    """

    def logp_mgf(self, uu, texp):
        """
        MGF of log price under the NIG model.

        The MGF of :math:`\\log(S_T/F_T)` at argument :math:`u` is

        .. math::

            \\exp\\!\\left(\\frac{T}{\\nu}\\left[
                \\left(\\sqrt{1 - 2\\theta\\nu - \\sigma^2\\nu} - 1\\right) u
                + 1 - \\sqrt{1 - 2\\theta\\nu u - \\sigma^2\\nu u^2}
            \\right]\\right).

        ``self._mgf1_correction`` stores :math:`\\kappa_{\\mathrm{NIG}}(1)\\nu = -\\omega\\nu`
        (the raw CGF at :math:`u=1`, scaled by :math:`\\nu`), so the correction
        term is ``-self._mgf1_correction * uu``.

        Args:
            uu: MGF argument (scalar or array). Real values give the MGF;
                purely imaginary values ``uu = 1j * xi`` give the characteristic
                function.
            texp: time to expiry.

        Returns:
            MGF value(s) at ``uu``, same shape as ``uu``.
        """
        volvar = self.nu * self.sigma**2
        rv = -self._mgf1_correction * uu + 1 - np.sqrt(1 + (-2 * self.theta * self.nu - volvar * uu) * uu)
        # CF: rv = 1j*self._mgf1_correction*uu + 1 - np.sqrt(1 + (-2j*self.theta*self.nu + volvar*uu)*uu)
        np.exp(texp/self.nu*rv, out=rv)
        return rv

    def logp_cum4(self, texp):
        """
        Analytic cumulants of log(S_T/F) for the NIG model.

        The CGF is K(u) = (T/ν)[ωνu + 1 − √(1 − 2θνu − σ²νu²)].
        Differentiating h = √g where g = 1 − 2θνu − σ²νu² via h² = g
        (so g‴ = g⁴ = 0) yields the recursion hʰ = −h′ʰ⁻¹ terms only,
        giving closed-form values at u = 0 (g = 1):

            ω  = −_mgf1_correction / ν       (martingale correction)
            c1 = T · (ω + θ)
            c2 = T · (σ² + νθ²)
            c3 = 3Tν · θ(σ² + νθ²)  =  3νθ · c2      (zero iff θ = 0)
            c4 = 3Tν · (σ² + νθ²)(σ² + 5νθ²)

        Returns:
            (c1, c2, c3, c4)
        """
        nu, sig2, th = self.nu, self.sigma**2, self.theta
        nth2 = nu * th**2
        omega = -self._mgf1_correction / nu
        c2 = texp * (sig2 + nth2)
        c3 = 3.0 * nu * th * c2                                   # = 3νθ · c2
        c4 = 3.0 * nu * c2 * (sig2 + 5.0 * nth2)                 # c2/texp = sig2+nth2
        c1 = texp * (omega + th)
        return float(c1), float(c2), float(c3), float(c4)


class ExpNigFft(NigABC, FftABC):
    """
    Normal Inverse Gaussian (NIG) model option pricing with FFT.

    Inherits ``logp_mgf`` from :class:`NigABC`.
    Also accessible as ``NigFft`` (preferred name).

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.arange(80, 121, 10)
        >>> m = pf.ExpNigFft(0.2, nu=0.3, theta=-0.1)
        >>> m.price(strike, 100, 1.0)
    """


NigFft = ExpNigFft


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


class RoughHestonFft(rheston.RoughHestonABC, FftABC):
    """
    Rough Heston model option pricing with FFT.

    Inherits ``logp_mgf`` (and both solver methods) from :class:`RoughHestonABC`.

    Method 1 (default): Adam's method — El Euch & Rosenbaum (2019).
    Method 2: Fast Hybrid approximation — Callegaro, Grasselli & Pagès (2020).

    Examples:
        >>> import numpy as np
        >>> import pyfeng.ex as pfex
        >>> strike = np.array([60, 70, 100, 140])
        >>> sigma, vov, mr, rho, texp, spot, theta, alpha = 0.0392, 0.1, 0.3156, -0.681, 1, 100, 0.3156, 0.62
        >>> m = pfex.RoughHestonFft(sigma, vov=vov, mr=mr, rho=rho, alpha=alpha)
        >>> m.price(strike, spot, texp)

        >>> strike = np.linspace(0.8, 1.2, 9)
        >>> sigma, vov, mr, rho, texp, spot, theta, alpha = 0.0392, 0.331, 0.1, -0.681, 1/12, 1, 0.3156, 0.62
        >>> m = pfex.RoughHestonFft(sigma, vov=vov, mr=mr, rho=rho, alpha=alpha, theta=theta)
        >>> m.price(strike, spot, texp)
        array([2.00004861e-01, 1.50105032e-01, 1.01136892e-01, 5.67278209e-02,
               2.39151266e-02, 6.81264421e-03, 1.19548373e-03, 1.19636483e-04,
               6.36294510e-06])
    """

    method = 1
    x_lim = 200  # integration limit



class OusvFft(ousv.OusvABC, FftABC):
    """
    OUSV model option pricing with FFT

    """


class Sv32Fft(sv32.Sv32ABC, FftABC):
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


class CgmyABC(CgmyParams, OptABC):
    """
    Abstract base for CGMY models — provides ``logp_mgf``.

    The CGMY process (Carr, Geman, Madan, Yor) is a four-parameter pure-jump
    Lévy process whose Lévy measure is

    .. math::

        \\nu(dx) = C\\left[
            \\frac{e^{-Mx}}{x^{1+Y}}\\,\\mathbf{1}_{x>0}
            + \\frac{e^{Gx}}{|x|^{1+Y}}\\,\\mathbf{1}_{x<0}
        \\right]dx.

    The per-unit-time cumulant generating function (CGF) is

    .. math::

        \\kappa(u) = C\\,\\Gamma(-Y)
            \\left[(M-u)^Y - M^Y + (G+u)^Y - G^Y\\right],
        \\quad -G < u < M,

    and the martingale correction is :math:`\\omega = -\\kappa(1)`, so that
    :math:`E[S_T] = F_T`.

    Note:
        The :math:`\\Gamma(-Y)` factor is singular at :math:`Y = 0, 1, 2, \\ldots`;
        these integer values require special limiting treatment.

    References:
        - Carr P, Geman H, Madan DB, Yor M (2002) The Fine Structure of Asset
          Returns: An Empirical Investigation. Journal of Business 75:305–332.
          https://doi.org/10.1086/338705
        - Ballotta L, Kyriakou I (2014) Monte Carlo Simulation of the CGMY
          Process and Option Pricing. Journal of Futures Markets 34:1095–1121.
          https://doi.org/10.1002/fut.21647
    """

    def logp_mgf(self, uu, texp):
        """
        MGF of log price under the CGMY model.

        The MGF of :math:`\\log(S_T/F_T)` at argument :math:`u` is

        .. math::

            \\exp\\!\\Bigl(T\\,\\bigl[\\kappa(u) + \\omega\\, u\\bigr]\\Bigr)

        where :math:`\\kappa(u) = C\\,\\Gamma(-Y)[(M-u)^Y - M^Y + (G+u)^Y - G^Y]`
        is the CGF and :math:`\\omega = -\\kappa(1)` is the martingale correction.
        ``self._mgf1_correction`` stores :math:`\\kappa(1)/[C\\,\\Gamma(-Y)]`
        (the raw-CGF factor at :math:`u=1`; precomputed as ``_mgf1_correction``
        and scaled by the fused constant ``_gam_Y_C`` :math:`= C\\,\\Gamma(-Y)`),
        so the correction term in the exponent is
        :math:`-\\text{\_gam\_Y\_C}\\cdot\\text{\_mgf1\_correction}\\cdot u\\cdot T = \\omega\\,u\\,T`.
        The martingale condition :math:`\\text{MGF}(1) = 1` holds because
        :math:`\\kappa(1) + \\omega = 0` by construction.

        Args:
            uu: MGF argument (scalar or array). Real values give the MGF;
                purely imaginary values ``uu = 1j * xi`` give the characteristic
                function. Must satisfy :math:`-G < \\operatorname{Re}(u) < M`.
            texp: time to expiry.

        Returns:
            MGF value(s) at ``uu``, same shape as ``uu``.

        References:
            - Eq (19) in Ballotta L, Kyriakou I (2014) Monte Carlo Simulation
              of the CGMY Process and Option Pricing. Journal of Futures Markets
              34:1095–1121. https://doi.org/10.1002/fut.21647
        """
        rv = np.power(self.M - uu, self.Y) - self._M_pow_Y + np.power(self.G + uu, self.Y) - self._G_pow_Y
        np.exp(self._gam_Y_C * texp * (-self._mgf1_correction*uu + rv), out=rv)
        return rv


class CgmyFft(CgmyABC, FftABC):
    """
    CGMY model option pricing with FFT.

    Inherits ``logp_mgf`` from :class:`CgmyABC`.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.arange(80, 121, 10)
        >>> m = pf.CgmyFft(C=1.0, G=5.0, M=10.0, Y=0.5)
        >>> m.price(strike, 100, 1.0)
    """


class GarchFftWuMaWang2012(GarchParams, FftABC):
    """
    GARCH diffusion model option pricing with approximate Fourier inversion

    References:
        - Wu X-Y, Ma C-Q, Wang S-Y (2012) Warrant pricing under GARCH diffusion model. Economic Modelling 29:2237–2244. https://doi.org/10.1016/j.econmod.2012.06.020
        - Lewis AL (2000) Option valuation under stochastic volatility: with Mathematica code. Finance Press

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> sigma, mr, theta, vov, rho = 0.06, 20.48, 0.218, 3.20, -0.99
        >>> strike, spot, texp = np.array([95, 100, 105]), 100, 0.5
        >>> m = pf.GarchFftWuMaWang2012(sigma, vov=vov, mr=mr, rho=rho, theta=theta)
        >>> m.price(strike, spot, texp)
        array([11.7235,  8.9978,  6.7091])
    """

    def logp_mgf_old(self, uu, texp):
        """
        Approximated log price MGF under the GARCH diffusion model.
        Line-by-line implementation of Lemma 2.1. of Wu Et al. (2012).

        Args:
            uu: dummy variable
            texp: time-to-expiry

        References:
            - Wu X-Y, Ma C-Q, Wang S-Y (2012) Warrant pricing under GARCH diffusion model. Economic Modelling 29:2237–2244. https://doi.org/10.1016/j.econmod.2012.06.020

        Returns:
            MGF value
        """
        zeta = -0.5 * (uu - uu ** 2)
        gg = self.mr - (3 / 2) * self.rho * self.vov * np.sqrt(self.theta) * uu
        dd = np.sqrt(gg ** 2 - 4 * self.vov ** 2 * zeta * self.theta)

        #### C ####
        term2_num = 2 * dd - (dd - gg) * (1 - np.exp(-dd * texp))
        term2_den = 2 * dd
        term2_log = 2 * np.log(term2_num / term2_den)

        term2 = - (1 / (2 * self.theta * self.vov ** 2)) * (
                self.mr * self.theta - 1 / 2 * self.rho * self.vov * uu * self.theta ** (3 / 2)) * (
                        term2_log + (dd - gg) * texp)

        term3_num = (dd ** 2 - gg ** 2) * (dd - gg) * texp + (
                dd - gg) ** 3 * np.exp(-dd * texp) * texp - 4 * dd * (
                            dd - gg) * (1 - np.exp(-dd * texp))
        term3_den = 2 * dd - (dd - gg) * (1 - np.exp(-dd * texp))
        term3_log = -4 * gg * np.log(term2_num / term2_den)
        term3 = - (1 / (8 * self.vov ** 2)) * (term3_log + term3_num / term3_den)
        C =  term2 + term3

        #### D ####
        num = 2 * zeta * (1 - np.exp(-dd * texp))
        den = 2 * dd - (dd - gg) * (1 - np.exp(-dd * texp))
        D =  num / den

        return np.exp(C + D * self.sigma)


    def logp_mgf(self, uu, texp):
        """
        Approximated Log price MGF under the GARCH diffusion model.
        Refined implementation of Lemma 2.1. of Wu Et al. (2012).

        Args:
            uu: dummy variable
            texp: time-to-expiry

        References:
            - Wu X-Y, Ma C-Q, Wang S-Y (2012) Warrant pricing under GARCH diffusion model. Economic Modelling 29:2237–2244. https://doi.org/10.1016/j.econmod.2012.06.020

        Returns:
            MGF value
        """

        zeta = uu - uu**2    # - zeta of the paper
        uu_etc = self.rho * self.vov * np.sqrt(self.theta) * uu

        gg = self.mr - (3/2)*uu_etc
        dd = np.sqrt(gg**2 + 2*self.vov**2 * self.theta * zeta)
        dd_m_gg = 2*self.vov**2 * self.theta * zeta / (dd + gg)   # dd - gg
        avgexp = -np.expm1(-dd*texp) / dd

        ### Calculation of C term
        tmp = 1 - 0.5*dd_m_gg * avgexp    ##  (2d - (d-g)(1-e^{-d tau})) / 2d
        log_tmp = np.log(tmp)

        C = -(self.mr - 0.5*uu_etc) * dd_m_gg * texp - (self.mr + 0.5*uu_etc) * log_tmp
        term3_num = (2*self.vov**2 * self.theta * zeta) + dd_m_gg**2 * np.exp(-dd * texp) - 4 * dd**2 * avgexp
        C -= (term3_num * dd_m_gg * texp / (2*dd) / tmp) / 4  # term3
        ### End of Calculation of C term

        ### Calculation of D term
        D = 0.5 * avgexp / tmp * zeta
        ### End of Calculation of C term

        return np.exp(C * (0.5/self.vov**2) - D * self.sigma)
        # BSM = exp(-0.5*self.sigma**2*texp*uu*(1 - uu))