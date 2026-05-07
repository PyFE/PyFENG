import numpy as np
from . import bsm
from .opt_abc import OptABC
from .params import SviParams


class Svi(SviParams, OptABC):
    """
    Stochastic Volatility-inspired (SVI) model by Gatheral.

    Parameters are defined in :class:`~pyfeng.params.SviParams`.

    References
        - Gatheral J, Jacquier A (2013) Arbitrage-free SVI volatility surfaces. arXiv:12040646 [q-fin]
    """

    def base_model(self, sigma=None):
        return bsm.Bsm(sigma, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)

    def vol_for_price(self, strike, spot, texp):
        fwd = self.forward(spot, texp)
        money = np.log(strike / fwd) - self.shift
        vol = np.sqrt(self.sigma + self.vov * (self.rho * money + np.sqrt(money**2 + self.smooth**2)))
        return vol

    @classmethod
    def init_from_heston(cls, sigma, vov=0.8, rho=-0.7, mr=0.5, theta=None, texp=1.0, intr=0.0, divr=0.0, is_fwd=False):
        """
        SVI initalization with equivalent Heston model by Gatheral & Jacquier (2011)

        Args:
            sigma: Heston sigma
            vov: Heston vov
            rho: Heston rho
            mr: Heston mr
            theta: Heston theta
            texp: time to expiry
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.

        Returns: model

        References:
            - Gatheral J, Jacquier A (2011) Convergence of Heston to SVI. Quantitative Finance 11:1129–1132. https://doi.org/10.1080/14697688.2010.550931
        """

        theta = sigma if theta is None else theta

        rhoc2 = 1 - rho**2
        w1 = np.sqrt((2*mr - rho*vov)**2 + vov**2 * rhoc2) - (2*mr - rho*vov)
        w1 *= (4*mr*theta) / (vov**2 * rhoc2)
        w2 = sigma / (mr * theta)
        sigma_ = w1 * rhoc2 / 2
        vov_ = (w1 * w2) / (2 * texp)
        shift = - rho * texp / w2
        smooth = np.sqrt(rhoc2) * texp / w2
        return cls(sigma_, vov=vov_, rho=rho, smooth=smooth, shift=shift, intr=intr, divr=divr, is_fwd=is_fwd)

    def price(self, strike, spot, texp, cp=1):
        vol = self.vol_for_price(strike, spot, texp)
        m_vol = self.base_model(vol)
        return m_vol.price(strike, spot, texp, cp=cp)
