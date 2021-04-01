# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10
@author: jaehyuk

"""

import abc
import copy
import os
import pandas as pd
import numpy as np
import scipy.optimize as spop

from . import opt_smile_abc as smile
from . import bsm
from . import norm
from . import cev


class SabrABC(smile.OptSmileABC, abc.ABC):
    vov, beta, rho = 0.0, 1.0, 0.0
    _base_beta = None

    def __init__(self, sigma, vov=0.0, rho=0.0, beta=1.0, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            beta: elasticity parameter. 0.5 by default
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)
        self.vov = vov
        self.rho = rho
        self.beta = beta

    def params_kw(self):
        params = super().params_kw()
        extra = {"vov": self.vov, "beta": self.beta, "rho": self.rho}
        return {**params, **extra}  # Py 3.9, params | extra

    def _variables(self, fwd, texp):
        betac = 1.0 - self.beta
        alpha = self.sigma / np.power(fwd, betac) #if self.beta > 0.0 else self.sigma
        rho2 = self.rho * self.rho
        rhoc = np.sqrt(1.0 - rho2)
        vovn = self.vov * np.sqrt(np.maximum(texp, 1e-64))
        return alpha, betac, rhoc, rho2, vovn

    def _m_base(self, vol):
        """
        The model of the vol_for_price

        Args:
            vol:

        Returns: model
        """
        vol_beta = self.beta if self._base_beta is None else self._base_beta

        if np.isclose(vol_beta, 1):
            return bsm.Bsm(vol, intr=self.intr, divr=self.divr)
        elif np.isclose(vol_beta, 0):
            return norm.Norm(vol, intr=self.intr, divr=self.divr)
        else:
            return cev.Cev(vol, beta=vol_beta, intr=self.intr, divr=self.divr)

    def vol_smile(self, strike, spot, texp, cp=1, model=None):
        if model is None:
            model = 'norm' if np.isclose(self.beta, 0) else 'bsm' if self.beta > 0.0 else None

        return super().vol_smile(strike, spot, texp, cp=1, model=model)

    @classmethod
    def init_benchmark(cls, set_no=None):
        """
        Initiate a SABR model with stored benchmark parameter sets

        Args:
            set_no: set number

        Returns:
            Dataframe of all test cases if set_no = None
            (model, Dataframe of result, params) if set_no is specified

        References:
            Antonov, Alexander, Konikov, M., & Spector, M. (2013). SABR spreads its wings. Risk, 2013(Aug), 58–63.
            Antonov, Alexandre, Konikov, M., & Spector, M. (2019). Modern SABR Analytics. Springer International Publishing. https://doi.org/10.1007/978-3-030-10656-0
            Antonov, Alexandre, & Spector, M. (2012). Advanced analytics for the SABR model. Available at SSRN. https://ssrn.com/abstract=2026350
            Cai, N., Song, Y., & Chen, N. (2017). Exact Simulation of the SABR Model. Operations Research, 65(4), 931–951. https://doi.org/10.1287/opre.2017.1617
            Korn, R., & Tang, S. (2013). Exact analytical solution for the normal SABR model. Wilmott Magazine, 2013(7), 64–69. https://doi.org/10.1002/wilm.10235
            Lewis, A. L. (2016). Option valuation under stochastic volatility II: With Mathematica code. Finance Press.
            von Sydow, L., ..., Haentjens, T., & Waldén, J. (2018). BENCHOP - SLV: The BENCHmarking project in Option Pricing – Stochastic and Local Volatility problems. International Journal of Computer Mathematics, 1–14. https://doi.org/10.1080/00207160.2018.1544368
        """
        this_dir, _ = os.path.split(__file__)
        file = os.path.join(this_dir, 'data/sabr_benchmark.xlsx')
        df_param = pd.read_excel(file, sheet_name='Param', index_col='Sheet')

        if set_no is None:
            return df_param
        else:
            df_val = pd.read_excel(file, sheet_name=str(set_no))
            param = df_param.loc[set_no]
            args_model = {k: param[k] for k in ('sigma', 'vov', 'rho', 'beta')}
            args_pricing = {k: param[k] for k in ('texp', 'spot')}

            assert(df_val.columns[0] == 'k' or df_val.columns[0] == 'K')
            args_pricing['strike'] = df_val.values[:, 0]
            if df_val.columns[0] == 'k':
                args_pricing['strike'] *= param['fwd']

            val = df_val[param['col_name']].values
            is_iv = param['col_name'].startswith('IV')

            m = cls(**args_model)

            param_dict = {
                'args_pricing': args_pricing,
                'ref': param['Reference'], 'val': val, 'is_iv': is_iv
            }

            return m, df_val, param_dict


class SabrVolApproxABC(SabrABC):
    """
    Abstract class for SABR models with volatility approximation (asymptotic expansion)
    """
    approx_order = 1  # order in texp: 0: leading order, 1: first order, -1: reserved for 2/(1+exp(-2*eps)) smoothing 

    @staticmethod
    def _vv(zz, rho):
        return np.sqrt(1 + zz * (zz + 2 * rho))

    @staticmethod
    def _int_inv_locvol(strike, beta, fwd=1.0):
        """
        (int from F to K x^-beta dx)/(K-F)
            = 1/(1-beta) * (K^(1-beta) - F^(1-beta))/(K-F)
            = F^(-beta)/(1-beta) * (k^(1-beta) - 1)/(k-1) where k = K/F

        Args:
            strike:
            beta:
            fwd:

        Returns:

        """

        assert np.isscalar(beta)
        beta = float(beta)  # np.power complains if power is negative integer
        betac = 1.0 - beta
        kk = strike / fwd

        # fall-back indices and values first
        ind_atm = np.fabs(kk - 1.0) < 1e-6
        val = 1 - beta / 2 * (kk - 1) * (1 - (1 + beta) / 3 * (kk - 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            if abs(betac) < 1e-4:
                val = np.where(ind_atm, val, np.log(kk) / (kk - 1))
            else:
                val = np.where(ind_atm, val, (np.power(kk, betac) - 1) / betac / (kk - 1))

        return val

    @staticmethod
    def _hh(zz, rho):
        """
        H(z) in the paper

        Args:
            zz: array
            rho: scalar

        Returns: H(z)

        """
        rho2 = rho * rho
        # initalization with expansion for for small |zz|
        xx_zz = 1 - zz * ((rho / 2) - zz * ((1 / 2 * rho2 - 1 / 6) - 1 / 8 * (5 * rho2 - 3) * rho * zz))

        yy = SabrVolApproxABC._vv(zz, rho)
        eps = 1e-5

        with np.errstate(divide='ignore', invalid='ignore'):  # suppress error for zz=0
            # replace negative zz
            xx_zz = np.where(zz > -eps, xx_zz, np.log((1 - rho) / (yy - (zz + rho))) / zz)
            # replace positive zz
            xx_zz = np.where(zz < eps, xx_zz, np.log((yy + (zz + rho)) / (1 + rho)) / zz)

        return 1.0 / xx_zz

    @abc.abstractmethod
    def vol_for_price(self, strike, spot, texp):
        """
        Equivalent volatility of the SABR model

        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry

        Returns:
            equivalent volatility
        """
        pass
    
    def price(self, strike, spot, texp, cp=1):
        vol = self.vol_for_price(strike, spot, texp)
        m_vol = self._m_base(vol)
        price = m_vol.price(strike, spot, texp, cp=cp)
        return price

    def impvol(self, price, strike, spot, texp, cp=1, setval=False):
        model = copy.copy(self)

        vol = self._m_base().impvol(price, strike, spot, texp, cp=cp)

        def iv_func(_sigma):
            model.sigma = _sigma
            return model.vol_for_price(strike, spot, texp) - vol

        sigma = spop.brentq(iv_func, 0, 10)

        if setval:
            self.sigma = sigma
        return sigma

    def vol_smile(self, strike, spot, texp, cp=1, model=None):
        if model is None:
            model = 'norm' if np.isclose(self.beta, 0) else 'bsm' if self.beta > 0.0 else None

        vol_beta = self.beta if self._base_beta is None else self._base_beta
        if (model.lower() == 'bsm' and np.isclose(vol_beta, 1)) or \
                (model.lower() == 'norm' and np.isclose(vol_beta, 0)):
            vol = self.vol_for_price(strike, spot, texp)
        else:
            vol = super().vol_smile(strike, spot, texp, cp=cp, model=model)
        return vol


class SabrHagan2002(SabrVolApproxABC):
    """
    SABR model with Hagan's implied volatility approximation for 0<beta<=1.

    References:
        Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing Smile Risk.
        Wilmott, September, 84–108.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.SabrHagan2002(sigma=2, vov=0.2, rho=-0.3, beta=0.5)
        >>> m.vol_for_price(np.arange(80, 121, 10), 100, 1.2)
        array([0.21976016, 0.20922027, 0.200432  , 0.19311113, 0.18703486])
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([22.04862858, 14.56226187,  8.74170415,  4.72352155,  2.28891776])
    """
    _base_beta = 1.0  # should not be changed

    def vol_for_price(self, strike, spot, texp):
        # fwd, spot, sigma may be either scalar or np.array.
        # texp, vov, rho, beta should be scholar values

        if texp <= 0.0:
            return 0.0
        fwd, _, _ = self._fwd_factor(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(spot, texp)
        betac2 = betac**2

        log_kk = np.log(fwd/strike)
        log_kk2 = log_kk*log_kk
        pow_kk = np.power(strike/fwd, betac/2)

        pre1 = pow_kk*(1 + betac2/24*log_kk2*(1 + betac2/80*log_kk2))

        term02 = (2 - 3*rho2)/24 * self.vov**2
        term11 = alpha*self.vov*self.rho*self.beta/4/pow_kk
        term20 = (betac*alpha/pow_kk)**2 / 24

        if self.approx_order == 0:
            vol = 1.0
        else:
            vol = 1.0 + texp*(term02 + term11 + term20)

        zz = pow_kk*log_kk*self.vov/np.maximum(alpha, np.finfo(float).eps)
        hh = self._hh(-zz, self.rho)  # note we pass -zz becaues hh(zz) definition is different
        vol *= alpha*hh/pre1   # bsm vol
        return vol

    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp=1, setval=False, is_vol=True):
        """
        Given option prices or normal vols at 3 strikes, compute the sigma, vov, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving
        you may use spop.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        """
        model = copy.copy(self)

        if is_vol:
            vol3 = price_or_vol3
        else:
            vol3 = self._m_base().impvol(price_or_vol3, strike3, spot, texp)

        def iv_func(x):
            model.sigma = np.exp(x[0])
            model.vov = np.exp(x[1])
            model.rho = np.tanh(x[2])
            return model.vol_for_price(strike3, spot, texp=texp) - vol3

        sol = spop.root(iv_func, np.array([vol3[1], -1, 0.0]))
        params = {"sigma": np.exp(sol.x[0]), "vov": np.exp(sol.x[1]), "rho": np.tanh(sol.x[2])}

        if setval:
            self.sigma, self.vov, self.rho = params["sigma"], params["vov"], params["rho"]

        return params

    
class SabrChoiWu2021H(SabrVolApproxABC, smile.MassZeroABC):
    """
    The CEV volatility approximation of the SABR modelbased on Theorem 1 of Choi & Wu (2019)

    References:
        Choi, J., & Wu, L. (2019). The equivalent constant-elasticity-of-variance (CEV) volatility
        of the stochastic-alpha-beta-rho (SABR) model.
        ArXiv:1911.13123 [q-Fin]. http://arxiv.org/abs/1911.13123

        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.SabrChoiWu2021H(sigma=2, vov=0.2, rho=-0.3, beta=0.5)
        >>> m.vol_for_price(np.arange(80, 121, 10), 100, 1.2)
        array([2.07833214, 2.03698255, 2.00332   , 1.97692259, 1.95735019])
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([22.04897988, 14.56240351,  8.74169054,  4.72340753,  2.28876105])
    """

    def __init__(self, sigma, vov=0.0, rho=0.0, beta=1.0, intr=0.0, divr=0.0, is_fwd=False, vol_beta=None):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            beta: elasticity parameter. 0.5 by default
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
            vol_beta: the beta for the volatility to choose _m_vol. If None (by default) vol_beta = beta
        """
        self.vol_beta = vol_beta
        super().__init__(sigma, vov, rho, beta, intr, divr, is_fwd)

    def vol_for_price(self, strike, spot, texp):
        # fwd, spot, sigma may be either scalar or np.array.
        # texp, vov, rho, beta should be scholar values

        vol_beta = self.beta if self.vol_beta is None else self.vol_beta
        vol_betac = 1.0 - vol_beta

        fwd, _, _ = self._fwd_factor(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)

        kk = strike / fwd  # standardized strike

        vov_over_alpha_safe = self.vov/np.maximum(alpha, np.finfo(float).eps)
        tmp = self._int_inv_locvol(kk, self.beta)
        qq_ratio = 1.0 if self.vol_beta is None else self._int_inv_locvol(kk, vol_beta) / tmp

        qq = tmp * (kk-1.0)
        zz = vov_over_alpha_safe * qq  # zeta = (vov/sigma0) q
        hh = self._hh(zz, self.rho)
        # term02: O(vov^2)
        term02 = (2 - 3*rho2)/24 * self.vov**2

        # term11: O(alpha*vov)
        # C(k)-C(1)/(k-1). Notice that 1/beta comes from int_inv_locvol
        term11 = self.rho*self.beta/4 * self.vov*alpha * self._int_inv_locvol(kk, betac)

        # term20: O(alpha^2)
        if self.vol_beta is None or np.isclose(self.beta, vol_beta):
            term20 = 0.0
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                ## Override ATM (qq=0)
                term20 = np.where(
                    np.fabs(qq) < 1e-6,
                    (np.square(betac) - np.square(vol_betac)) / 24 * np.square(alpha),
                    (0.5*(self.beta-self.vol_beta)*np.log(kk) - np.log(qq_ratio)) * np.square(alpha)/qq
                )
        #else:
        #    raise ValueError('Cannot handle this vol_beta different from beta')

        ## Return
        if self.approx_order == 0:
            vol = 1.0
        else:
            vol = 1.0 + texp*(term20 + term11 + term02)

        pre_fac = alpha * np.power(fwd, vol_betac) * qq_ratio
        vol *= pre_fac*hh
        return vol

    def mass_zero(self, spot, texp, log=False):
        assert(self.vol_beta is None)
        vol_cev = self.vol_for_price(0.0, spot, texp)
        cev_m = cev.Cev(sigma=vol_cev, beta=self.beta)
        mass = cev_m.mass_zero(spot, texp, log=log)
        #print(vol_cev, mass)
        return mass

    def mass_zero_t0(self, spot, texp):
        """
        Limit value of -T log(M_T) as T -> 0, where M_T is the mass at zero.
            See Corollary 3.1 of Choi & Wu (2019)

        Args:
            spot: spot (or forward) price

        Returns:
            - lim_{T->0} T log(M_T)
        """
        fwd = self.forward(spot, texp)
        alpha, betac, rhoc, rho2, _ = self._variables(fwd, texp)
        hh = self._hh(-self.vov / (alpha * betac), self.rho)
        t0 = 0.5/(betac*alpha*hh)**2
        return t0


class SabrChoiWu2021P(SabrChoiWu2021H, smile.MassZeroABC):
    """
    The CEV volatility approximation of the SABR modelbased on Theorem 2 of Choi & Wu (2019)

    References:
        Choi, J., & Wu, L. (2019). The equivalent constant-elasticity-of-variance (CEV) volatility
        of the stochastic-alpha-beta-rho (SABR) model.
        ArXiv:1911.13123 [q-Fin]. http://arxiv.org/abs/1911.13123

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.SabrChoiWu2021P(sigma=2, vov=0.2, rho=-0.3, beta=0.5)
        >>> m.vol_for_price(np.arange(80, 121, 10), 100, 1.2)
        array([2.07761123, 2.03665311, 2.00332   , 1.97718783, 1.95781579])
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([22.0470526 , 14.56114825,  8.74169054,  4.72447547,  2.29018838])
    """

    def vol_for_price(self, strike, spot, texp):
        # fwd, spot, sigma may be either scalar or np.array.
        # texp, vov, rho, beta should be scholar values

        vol_beta = self.beta if self.vol_beta is None else self.vol_beta
        vol_betac = 1.0 - vol_beta

        fwd, _, _ = self._fwd_factor(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)

        kk = strike / fwd  # standardized strike

        # explicitly make np.array even if args are all scalar or list
        if isinstance(kk, float):
            kk = np.array([kk])

        ## Eq 32 (leading order)
        vov_over_alpha_safe = self.vov/np.maximum(alpha, np.finfo(float).eps)

        # qq_ratio = qq_vol_beta / qq_beta
        tmp = self._int_inv_locvol(kk, self.beta)
        qq_ratio = 1.0 if self.vol_beta is None else self._int_inv_locvol(kk, vol_beta) / tmp

        zz = vov_over_alpha_safe * tmp * (kk - 1.0)  # zeta = (vov/sigma0) q
        hh = self._hh(zz, self.rho)
        v_m = self._vv(zz, self.rho)

        if abs(betac) < 0.001:
            gg_diff = 0.5*self.beta*self.rho/(1.0-rho2) * (v_m - 1.0 - self.rho*zz)
        else:
            t1 = (v_m + self.rho + zz)/rhoc  # array
            t2 = (1 + self.rho)/rhoc  # scalar

            eta = self.vov/(betac*alpha)*np.power(kk, betac) * (rhoc/v_m)
            eta2_m_1 = eta*eta - 1.0
            sqrt_eta2_m_1 = np.sqrt(abs(eta2_m_1))

            num_t1 = self.rho + (eta-rhoc)*t1
            num_t2 = self.rho + (eta-rhoc)*t2

            gg1 = np.arctan(t1)
            gg2 = np.arctan(t2) * np.ones_like(t1)  # t2 is a scalar, so vectorize

            eps = 1e-6

            ind = (eta2_m_1 > eps)
            gg1[ind] -= eta[ind] / sqrt_eta2_m_1[ind] * np.arctan(num_t1[ind] / sqrt_eta2_m_1[ind])
            gg2[ind] -= eta[ind] / sqrt_eta2_m_1[ind] * np.arctan(num_t2[ind] / sqrt_eta2_m_1[ind])

            ind = (eta2_m_1 < -eps)
            gg1[ind] += eta[ind] / sqrt_eta2_m_1[ind] \
                        * 0.5*np.log(abs( (num_t1[ind]+sqrt_eta2_m_1[ind])/(num_t1[ind]-sqrt_eta2_m_1[ind]) ))

            # as eta->0 (k->0) the numerator diverge. but the gg2 value should be 0.
            ind = (eta2_m_1 < -eps)*(eta > eps)
            gg2[ind] += eta[ind] / sqrt_eta2_m_1[ind] \
                        * 0.5*np.log(abs( (num_t2[ind]+sqrt_eta2_m_1[ind])/(num_t2[ind]-sqrt_eta2_m_1[ind]) ))
            # when eta is very small, the term above is zero, so do nothing.

            ind = (abs(eta2_m_1) <= eps)
            gg1[ind] += 1.0 / num_t1[ind]
            gg2[ind] += 1.0 / num_t2[ind]

            gg_diff = self.rho*self.beta/(rhoc*betac) * (gg2 - gg1)

        zz2_safe = np.maximum(np.square(zz), np.finfo(float).eps)
        tmp = 0 if self.vol_beta is None else 0.5*(self.beta-self.vol_beta)*np.log(kk) - np.log(qq_ratio)

        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ## Override ATM (qq=0)
            term20 = np.where(
                np.fabs(qq) < 1e-6,
                (np.square(betac) - np.square(vol_betac)) / 24 * np.square(alpha),
                (0.5 * (self.beta - self.vol_beta) * np.log(kk) - np.log(qq_ratio)) * np.square(alpha) / qq
            )
        """
        order1 = np.square(self.vov*hh)/zz2_safe *\
                 (tmp + 0.5*np.log(v_m/np.square(hh)) + gg_diff)

        ## Override ATM (z=0)
        ind_atm = (abs(zz) < 1e-6)
        order1[ind_atm] = (np.square(betac)-np.square(vol_betac))/24*alpha*alpha \
            + ((self.rho*self.beta/4)*alpha + (2 - 3*rho2)/24*self.vov)*self.vov  # RHS scalar

        # return value
        if self.approx_order == 0:
            vol = 1.0
        else:
            vol = 1.0 + order1*texp

        # order0 = (z_base' / z_base) * hh
        vol *= alpha * np.power(fwd, vol_betac) * qq_ratio * hh
        return vol[0] if vol.size == 1 else vol


class SabrLorig2017(SabrVolApproxABC):
    """
    Third-order BSM volatilty approximation of the SABR model by Lorig et al. (2017)

    References:
        Lorig, M., Pagliarani, S., & Pascucci, A. (2017). Explicit Implied Volatilities
        for Multifactor Local-Stochastic Volatility Models.
        Mathematical Finance, 27(3), 926–960. https://doi.org/10.1111/mafi.12105
    """

    _base_beta = 1.0  # should not be changed
    lv_factor = 1.0
    approx_order = 3

    def vol_for_price(self, strike, spot, texp):
        # fwd, spot, sigma may be either scalar or np.array.
        # texp, vov, rho, beta should be scholar values

        fwd, _, _ = self._fwd_factor(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(fwd, texp)

        kk = strike / fwd  # standardized strike

        kmx = np.log(kk)
        kmx_2 = kmx*kmx
        kmx_3 = kmx*kmx_2

        t_2 = texp*texp if self.approx_order > 1 else 0.0
        t_3 = texp*t_2 if self.approx_order > 2 else 0.0

        betac_2 = betac*betac
        betac_3 = betac*betac_2

        rho = self.rho
        vov = self.vov
        vov_2 = vov*vov
        vov_3 = vov*vov_2

        s0 = alpha
        s0_2 = s0*s0
        s0_3 = s0*s0_2
        s0_4 = s0*s0_3
        s0_5 = s0*s0_4

        s10 = s0/2*(-betac)*kmx
        s01 = vov/4 * ( 2*kmx*rho + texp*s0*(-vov+rho*s0) )

        s20 = s0*betac_2 * (texp/24*s0_2 - t_2/96*s0_4 + kmx_2/12) * self.lv_factor

        s11 = -betac*vov
        s11 *= texp/4*rho*s0_2 - t_2/48*rho*s0_4 - texp/24*s0*(3*vov-5*rho*s0)*kmx

        s02 = vov_2
        s02 *= texp/24*(8-3*rho2)*s0 + t_2/96*s0*(5*vov_2 + 2*s0*((6*rho2-2)*s0 - 7*vov*rho)) \
            - texp/24*rho*(vov-3*rho*s0)*kmx + (2-3*rho2)/(12*s0)*kmx_2

        s30 = -betac_3*s0_3*kmx
        s30 *= texp/16 - 5/192*t_2*s0_2

        s21 = betac_2*vov
        s21 *= t_2/96*s0_3*(7*rho*s0 - 3*vov) + t_3/384*s0_5*(5*vov-7*rho*s0) \
            + 13/48*texp*rho*s0_2*kmx - 13/192*t_2*rho*s0_4*kmx \
            - texp/48*s0*(vov-3*rho*s0)*kmx_2

        s12 = -betac*vov_2
        s12 *= t_2/48*rho*s0_2*(13*rho*s0-7*vov) + t_3/192*rho*s0_4*(5*vov-7*rho*s0) \
            + texp/48*(3*rho2+8)*s0*kmx \
            + t_2/192*s0*(5*vov_2 - 22*vov*rho*s0 + 4*(5*rho2-3)*s0_2)*kmx \
            + texp/24*rho2*s0*kmx_2 + (3*rho2-2)/(24*s0)*kmx_3

        s03 = vov_3
        s03 *= t_2/96*s0*(3*vov*(rho2-4) + rho*(26-9*rho2)*s0) \
            + t_3/384*s0*(s0*(19*vov_2*rho + 2*s0*(vov*(8-21*rho2) + rho*(15*rho2-11)*s0)) - 3*vov_3) \
            + texp/48*rho*(3*rho2-2)*kmx + t_2/192*rho*(vov_2+6*s0*(vov*rho+(1-2*rho2)*s0))*kmx \
            - texp/16*rho*(rho2-1)*kmx_2 + rho*(6*rho2-5)/(24*s0_2)*kmx_3

        vol = s0 + (s10+s01) + (s20+s11+s02) + (s30+s21+s12+s03)

        return vol
