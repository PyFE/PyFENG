import abc

from . import bsm
from . import norm
from . import opt_abc as opt


class OptSmileABC(opt.OptABC, abc.ABC):
    """
    Abstract class to handle volatility smile
    """
    def _vol_smile_model(self, model='bsm'):
        if model.lower() == 'bsm':
            base_model = bsm.Bsm(None, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)
        elif model.lower() == 'norm':
            base_model = norm.Norm(None, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)
        else:
            base_model = None
        return base_model

    def vol_smile(self, strike, spot, texp, cp=1, model='bsm'):
        """
        Equivalent volatility smile for a given model

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
            model: {'bsm', 'norm'} 'bsm' for Black-Scholes-Merton, 'norm' for Bachelier (normal)

        Returns:
            volatility smile under the specified model
        """
        base_model = self._vol_smile_model(model)
        price = self.price(strike, spot, texp, cp=cp)
        vol = base_model.impvol(price, strike, spot, texp, cp=cp)
        return vol
