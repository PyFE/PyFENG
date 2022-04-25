import abc
import numpy as np
import scipy.special as spsp
from .bsm import Bsm
from .norm import Norm
from . import sv_abc as sv

class SubordBmABC(sv.SvABC):

    # To do:
    # merge with SabrCondDistABC in sabr_int.py  use cond_spot_sigma()
    # use quad.py functions
    # unified initializer, seperate model vs numerical params: set_num_param()
    #

    mr = 0.0
    sv_param = True

    n_quad = 7
    nu = None

    def __init__(self, sigma, vov=0.01, rho=0.0, n_quad=7, intr=0.0, divr=0.0, is_fwd=False, sv_param=True):
        super().__init__(sigma, vov, rho, None, None, intr, divr, is_fwd=is_fwd)
        self.n_quad = n_quad
        self.sv_param = sv_param

    @abc.abstractmethod
    def quad(self, texp, vov):
        return NotImplementedError

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
