import abc
import numpy as np
import scipy.special as spsp
from .bsm import Bsm
from .norm import Norm
from .opt_abc import OptABC
from .params import SvParams, VarGammaParams, NigParams

class SubordBmABC(SvParams, OptABC):

    # To do:
    # merge with SabrCondDistABC in sabr_int.py  use cond_spot_sigma()
    # use quad.py functions
    # unified initializer, seperate model vs numerical params: set_num_param()
    #

    sv_param = True

    n_quad = 7
    nu = None

    def __init__(self, sigma, vov=0.01, rho=0.0, n_quad=7, intr=0.0, divr=0.0, is_fwd=False, sv_param=True):
        super().__init__(sigma, vov=vov, rho=rho, mr=0.0, intr=intr, divr=divr, is_fwd=is_fwd)
        self.n_quad = n_quad
        self.sv_param = sv_param

    @abc.abstractmethod
    def quad(self, texp, vov):
        raise NotImplementedError

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        sigma2 = self.sigma**2

        if self.sv_param:
            rho2 = self.rho**2
            rhoc = np.sqrt(1-rho2)
            var, w = self.quad(texp, self.vov**2)
            fwd_ratio = np.exp(self.rho*self.sigma/self.vov*(var - texp) - 0.5*sigma2*rho2*var)
            vol_bsm = self.sigma * rhoc * np.sqrt(var / texp)
        else:
            theta = self.rho  # rho playing the role of theta
            v = self.vov  # alpha playing the role of v (variance rate)
            var, w = self.quad(texp, v)
            fwd_ratio = np.exp(theta*(var - texp) + 0.5*sigma2*var)
            vol_bsm = self.sigma * np.sqrt(var / texp)

        fwd_ratio_mean = sum(w*fwd_ratio)
        self.nu = -np.log(fwd_ratio_mean)
        fwd_ratio /= fwd_ratio_mean  # Make sure E(fwd_ratio) = 1.0

        strike_fwd = np.atleast_1d(strike/fwd)
        fwd_arr = fwd * np.ones_like(strike_fwd)

        price = np.zeros_like(strike_fwd)

        for k in range(len(price)):
            price_arr = fwd_arr[k] * Bsm.price_formula(
                strike_fwd[k], fwd_ratio, vol_bsm, texp, cp=cp)
            price[k] = np.sum(price_arr * w)

        return np.exp(-self.intr * texp) * price

    def vol_smile(self, strike, spot, texp, model='bsm', cp=1):
        if model.lower() == 'bsm':
            base_model = Bsm(None, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)
        elif model.lower() == 'norm':
            base_model = Norm(None, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)
        else:
            base_model = None

        price = self.price(strike, spot, texp, cp=cp)
        vol = base_model.impvol(price, strike, spot, texp, cp=cp)
        return vol


class VarGammaQuad(SubordBmABC):

    def quad(self, texp, var_rate):
        alpha = texp/var_rate
        x, w = spsp.roots_genlaguerre(self.n_quad, alpha - 1.0)
        x *= var_rate
        w /= np.sum(w)
        return x, w


class ExpNigQuad(SubordBmABC):

    def quad(self, texp, var_rate):
        z, w = spsp.roots_hermitenorm(self.n_quad)
        # We are creating quadrature IG(t,t^2/nu) = t * IG(1,t/nu)
        mu = 1.0
        lam = texp / var_rate
        fac = 0.5 * mu / lam

        y_hat = np.square(z) * fac
        w *= np.sqrt(2.0 / np.pi)   # 2.0 multiplied to the normal weight: sum(w)=2

        x_2 = 1 + y_hat + np.sqrt(y_hat * (2.0 + y_hat))
        x_1 = 1 / x_2
        p = 1 / (1 + x_1)
        ind_half = int(self.n_quad / 2)
        x_ig = np.concatenate((x_1[:ind_half], x_2[ind_half:])) * texp
        w_ig = np.concatenate((p[:ind_half], 1 - p[ind_half:])) * w

        return x_ig, w_ig


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

        Args:
            uu: MGF argument (scalar or array).
            texp: time to expiry.

        Returns:
            MGF value(s) at ``uu``, same shape as ``uu``.
        """
        volvar = self.nu * self.sigma**2
        rv = -self._mgf1_correction * uu - np.log1p((-self.theta * self.nu - 0.5 * volvar * uu) * uu)
        np.exp(texp/self.nu*rv, out=rv)
        return rv


class NigABC(NigParams, OptABC):
    """
    Abstract base for Normal Inverse Gaussian (NIG) models — provides ``logp_mgf``
    and analytic ``logp_cum4``.

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

        Args:
            uu: MGF argument (scalar or array).
            texp: time to expiry.

        Returns:
            MGF value(s) at ``uu``, same shape as ``uu``.
        """
        volvar = self.nu * self.sigma**2
        rv = -self._mgf1_correction * uu + 1 - np.sqrt(1 + (-2 * self.theta * self.nu - volvar * uu) * uu)
        np.exp(texp/self.nu*rv, out=rv)
        return rv

    def logp_cum4(self, texp):
        """
        Analytic cumulants of log(S_T/F) for the NIG model.

        From the CGF K(u) = (T/ν)[ωνu + 1 − √(1 − 2θνu − σ²νu²)]:

            c1 = T · (ω + θ)
            c2 = T · (σ² + νθ²)
            c3 = 3Tν · θ(σ² + νθ²)  =  3νθ · c2
            c4 = 3Tν · (σ² + νθ²)(σ² + 5νθ²)

        Returns:
            (c1, c2, c3, c4)
        """
        nu, sig2, th = self.nu, self.sigma**2, self.theta
        nth2 = nu * th**2
        omega = -self._mgf1_correction / nu
        c2 = texp * (sig2 + nth2)
        c3 = 3.0 * nu * th * c2
        c4 = 3.0 * nu * c2 * (sig2 + 5.0 * nth2)
        c1 = texp * (omega + th)
        return float(c1), float(c2), float(c3), float(c4)
