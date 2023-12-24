import abc
import numpy as np
from . import bsm
from . import norm
from . import opt_abc as opt
import scipy.stats as spst


class OptSmileABC(opt.OptABC, abc.ABC):
    """
    Abstract class to model with volatility smile
    """

    def _m_smile(self, model="bsm", is_fwd=None):
        if is_fwd is None:
            is_fwd = self.is_fwd
        if model.lower() == "bsm":
            base_model = bsm.Bsm(None, intr=self.intr, divr=self.divr, is_fwd=is_fwd)
        elif model.lower() == "norm":
            base_model = norm.Norm(None, intr=self.intr, divr=self.divr, is_fwd=is_fwd)
        else:
            base_model = None
        return base_model

    def vol_smile(self, strike, spot, texp, cp=None, model="bsm"):
        """
        Equivalent volatility smile for a given model

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
            model: {'bsm', 'norm'} 'bsm' (by default) for Black-Scholes-Merton, 'norm' for Bachelier

        Returns:
            volatility smile under the specified model
        """
        base_model = self._m_smile(model)
        if cp is None:
            fwd = self.forward(spot, texp)
            cp = np.where(strike > fwd, 1, -1)  # make option out-of-the-money
        price = self.price(strike, spot, texp, cp=cp)
        vol = base_model.impvol(price, strike, spot, texp, cp=cp)
        return vol


class MassZeroABC(opt.OptABC, abc.ABC):
    """
    Implied volatility asymptotics of De Marco et al. (2017) given the positive mass at zero.

    References:
          - De Marco, S., Hillairet, C., & Jacquier, A. (2017). Shapes of Implied Volatility with Positive Mass at Zero. SIAM Journal on Financial Mathematics, 8(1), 709–737. https://doi.org/10.1137/14098065X
    """

    @abc.abstractmethod
    def mass_zero(self, spot, texp, log=False):
        """
        Probability mass absorbed at the zero boundary (K=0)

        Args:
            spot: spot (or forward) price
            texp: time to expiry
            log: log value if True

        Returns:
            (log) probability mass at zero
        """
        return NotImplementedError

    def vol_from_mass_zero(self, strike, spot, texp, mass=None):
        """
        Implied volatility from positive mass at zero from DMHJ (2017)
        If mass is given, use the given value. If None (by default), compute model implied value.

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            mass: probability mass at zero (None by default)

        Returns:
            implied BSM volatility

        References:
              De Marco, S., Hillairet, C., & Jacquier, A. (2017). Shapes of Implied Volatility with
              Positive Mass at Zero. SIAM Journal on Financial Mathematics, 8(1), 709–737.
              https://doi.org/10.1137/14098065X
        """

        # Perhaps we should return Nan for k >= 1
        if mass is None:
            mass = self.mass_zero(spot, texp)

        fwd = self.forward(spot, texp)
        kk = strike / fwd
        tmp = np.sqrt(2 * np.abs(np.log(kk)))
        leading = tmp / np.sqrt(texp)

        qq = spst.norm.ppf(mass)
        vol = 1 + (qq + 0.5 * ((2 + qq ** 2) + qq / tmp) / tmp) / tmp
        vol *= leading
        return vol

    def price_from_mass_zero(self, strike, spot, texp, cp=1, mass=None):
        vol = self.vol_from_mass_zero(strike, spot, texp, mass=mass)
        base_model = bsm.Bsm(vol)
        price = base_model.price(strike, spot, texp, cp=cp)
        return price
