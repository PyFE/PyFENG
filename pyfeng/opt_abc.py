import abc
import copy
import numpy as np
import scipy.optimize as sopt
import scipy.stats as spst
from typing import ClassVar

class OptABC(abc.ABC):
    model_type: ClassVar[str]
    sigma, intr, divr = None, 0.0, 0.0
    is_fwd = False

    IMPVOL_TOL = 1e-10
    IMPVOL_MAXVOL = 99.99

    def __init__(self, sigma, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        self.sigma = sigma
        self.intr = intr
        self.divr = divr
        self.is_fwd = is_fwd

    def params_kw(self):
        """
        Model parameters in dictionary
        """
        params = {
            "sigma": self.sigma,
            "intr": self.intr,
            "divr": self.divr,
            "is_fwd": self.is_fwd,
        }
        return params

    def params_hash(self):
        dct = self.params_kw()
        return hash((frozenset(dct.keys()), frozenset(dct.values())))

    @classmethod
    def from_param(cls, other):
        """
        Create a new instance by copying parameters from another model parameters of the same type.

        Args:
            other: source param instance

        Returns:
            New instance of cls with parameters copied from other

        Raises:
            TypeError: if source and target are not the same model type
        """
        cls_type = getattr(cls, 'model_type', None)
        other_type = getattr(type(other), 'model_type', None)

        if isinstance(cls_type, str) and isinstance(other_type, str):
            # SV-family models: compare model_type string (allows cross-algorithm copies)
            if cls_type != other_type:
                raise TypeError(
                    f"Model type mismatch: source is '{other_type}' ({type(other).__name__}) "
                    f"but target is '{cls_type}' ({cls.__name__})."
                )
        elif not isinstance(other, cls):
            # Simple models (BSM, Norm, etc.): require class compatibility
            raise TypeError(
                f"Cannot copy from '{type(other).__name__}' to '{cls.__name__}'."
            )

        return cls(**other.params_kw())


    def forward(self, spot, texp):
        """
        Forward price

        Args:
            spot: spot price
            texp: time to expiry

        Returns:
            forward price
        """
        if self.is_fwd:
            return np.array(spot)
        else:
            return np.array(spot) * np.exp((self.intr - self.divr) * np.array(texp))

    def _fwd_factor(self, spot, texp):
        """
        Forward, discount factor, dividend factor

        Args:
            spot: spot (or forward) price
            texp: time to expiry

        Returns:
            (forward, discounting factor, dividend factor)
        """
        df = np.exp(-self.intr * np.array(texp))
        if self.is_fwd:
            divf = 1
            fwd = np.array(spot)
        else:
            divf = np.exp(-self.divr * np.array(texp))
            fwd = np.array(spot) * divf / df
        return fwd, df, divf

    @abc.abstractmethod
    def price(self, strike, spot, texp, cp=1):
        """
        Call/put option price.

        Args:
            strike: strike price.
            spot: spot (or forward) price.
            texp: time to expiry.
            cp: 1/-1 for call/put option.

        Returns:
            option price
        """
        raise NotImplementedError

    def impvol_brentq(self, price, strike, spot, texp, cp=1, setval=False):
        """
        Implied volatility using Brent's method. Slow but robust implementation.

        Args:
            price: option price
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put
            setval: if True, sigma is set with the solved implied volatility

        Returns:
            implied volatility
        """

        fwd, df, _ = self._fwd_factor(spot, texp)

        kk = strike / fwd  # strike / fwd
        price_std = price / df / fwd  # forward price / fwd

        model = copy.copy(self)
        model.sigma = 1e-64
        p_min = model.price(kk, 1, texp, cp)
        model.sigma = self.IMPVOL_MAXVOL
        p_max = model.price(kk, 1, texp, cp)

        scalar_output = np.isscalar(price) and np.isscalar(p_min)
        ones_like = np.ones_like(np.atleast_1d(price + p_min))

        sigma = np.empty(ones_like.shape).flatten()
        sigma.fill(np.nan)
        price_flat = (ones_like * price_std).flatten()
        p_min = (ones_like * p_min).flatten()
        p_max = (ones_like * p_max).flatten()
        texp_flat = (ones_like * texp).flatten()
        kk_flat = (ones_like * kk).flatten()
        cp_flat = (ones_like * cp).flatten()

        def iv_func(_sigma):
            model.sigma = _sigma
            return model.price(_strike, 1.0, _texp, _cp) - _price

        for k in range(len(sigma)):
            _cp = cp_flat[k]
            _texp = texp_flat[k]
            _strike = kk_flat[k]
            _price = price_flat[k]

            if np.abs(_price - p_min[k]) < self.IMPVOL_TOL:
                sigma[k] = 0.0
            elif np.abs(_price - p_max[k]) < self.IMPVOL_TOL:
                sigma[k] = self.IMPVOL_MAXVOL
            elif _price < p_min[k] or p_max[k] < _price:
                sigma[k] = np.nan
            else:
                sigma[k] = sopt.brentq(iv_func, 0.0, 10)
            """
                if time_value < -self.IMPVOL_TOL:
                    warn_msg = f'Negative time value: [%d] %g, strike:%f, price:%f' % (
                        k, time_value, kk_flat[k], price_flat[k])
                    warnings.warn(warn_msg, Warning)
            """

        if scalar_output:
            sigma = sigma[0]
        else:
            sigma = sigma.reshape(ones_like.shape)
        if setval:
            self.sigma = sigma
        return sigma

    ####
    impvol = impvol_brentq

    def _delta_shock(self, strike=100, spot=100, texp=1, cp=1):
        """
        Shock size for `delta_numeric`, `gamma_numeric`, and `vanna_numeric`

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            shock size
        """
        return spot * 0.001  # 10 bps of spot price

    def delta_numeric(self, strike, spot, texp, cp=1):
        """
        Option model delta (sensitivity to price) by finite difference

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            delta value
        """
        h = self._delta_shock(strike, spot, texp, cp)
        delta = (
            self.price(strike, spot + h, texp, cp)
            - self.price(strike, spot - h, texp, cp)
        ) / (2 * h)
        return delta

    def gamma_numeric(self, strike, spot, texp, cp=1):
        """
        Option model gamma (2nd derivative to price) by finite difference

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            Delta with numerical derivative
        """
        h = self._delta_shock(strike, spot, texp, cp)
        gamma = (
            self.price(strike, spot + h, texp, cp)
            - 2 * self.price(strike, spot, texp, cp)
            + self.price(strike, spot - h, texp, cp)
        ) / (h * h)
        return gamma

    def _vega_shock(self, strike=100, spot=100, texp=1, cp=1):
        """
        Shock size for `vega_numeric`, `volga_numeric`, and `vanna_numeric`

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            vega shock size
        """
        return 0.001  # 0.1% shock

    def vega_numeric(self, strike, spot, texp, cp=1):
        """
        Option model vega (sensitivity to volatility) by finite difference

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            vega value
        """
        h = self._vega_shock(strike, spot, texp, cp)
        model = copy.copy(self)
        model.sigma += h
        p_up = model.price(strike, spot, texp, cp)
        model.sigma -= 2 * h
        p_dn = model.price(strike, spot, texp, cp)

        vega = (p_up - p_dn) / (2 * h)
        return vega

    def volga_numeric(self, strike, spot, texp, cp=1):
        """
        Option model volga (2nd derivative to volatility) by finite difference

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            volga value
        """
        h = self._vega_shock(strike, spot, texp, cp)
        model = copy.copy(self)

        p_0 = model.price(strike, spot, texp, cp)
        model.sigma += h
        p_up = model.price(strike, spot, texp, cp)
        model.sigma -= 2 * h
        p_dn = model.price(strike, spot, texp, cp)

        volga = (p_up + p_dn - 2 * p_0) / (h * h)
        return volga

    def vanna_numeric(self, strike, spot, texp, cp=1):
        """
        Option model vanna (cross-derivative to price and volatility) by finite difference

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            vanna value
        """
        h = self._delta_shock(strike, spot, texp, cp)
        vega_up = self.vega_numeric(strike, spot + h, texp, cp)
        vega_dn = self.vega_numeric(strike, spot - h, texp, cp)

        vanna = (vega_up - vega_dn) / (2 * h)
        return vanna

    def _theta_shock(self, strike=100, spot=100, texp=1, cp=1):
        """
        Shock size for `theta_numeric`

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            shock size
        """
        return np.minimum(1 / 3652.5, texp)  # 1/10 day

    def theta_numeric(self, strike, spot, texp, cp=1):
        """
        Option model thegta (sensitivity to time-to-maturity) by finite difference

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            theta value
        """
        dt = self._delta_shock(strike, spot, texp, cp)/100
        theta = self.price(strike, spot, texp - dt, cp) - self.price(strike, spot, texp + dt, cp)
        theta /= 2*dt
        return theta

    def pdf_numeric(self, strike, spot, texp, cp=-1, h=0.001):
        """
        Probability density function (PDF) at `strike`

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            PDF values
        """
        fwd = self.forward(spot, texp)
        kk = strike / fwd
        kk_arr = np.array([kk - h, kk, kk + h]).flatten()
        price = self.price(kk_arr, 1.0, texp, cp=cp)
        price = price.reshape(3, -1)
        pdf = (price[2] + price[0] - 2.0 * price[1]) / (h * h)
        return pdf

    def cdf_numeric(self, strike, spot, texp, cp=-1, h=0.001):
        """
        Cumulative distribution function (CDF) at `strike`

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            CDF values
        """
        fwd = self.forward(spot, texp)
        #kk = strike / fwd
        strike_arr = np.array([strike - h, strike + h]).flatten()
        price = self.price(strike_arr, fwd, texp, cp=cp)
        price = price.reshape(2, -1)
        cdf = np.sign(cp)*(price[0] - price[1]) / (2*h)
        return cdf

    def _m_smile(self, model="bsm", is_fwd=None):
        """
        Construct a base BSM or Bachelier model instance for smile conversion.

        Args:
            model: {'bsm', 'norm'}
            is_fwd: if None, use self.is_fwd

        Returns:
            Bsm or Norm model instance (sigma=None), or None for unknown model
        """
        from . import bsm, norm  # lazy imports to avoid circular dependency
        if is_fwd is None:
            is_fwd = self.is_fwd
        if model.lower() == "bsm":
            return bsm.Bsm(None, intr=self.intr, divr=self.divr, is_fwd=is_fwd)
        elif model.lower() == "norm":
            return norm.Norm(None, intr=self.intr, divr=self.divr, is_fwd=is_fwd)
        else:
            return None

    def vol_smile(self, strike, spot, texp, cp=None, model="bsm"):
        """
        Equivalent volatility smile for a given model.

        Computes option prices via ``self.price()`` and inverts them to
        implied volatility under the requested base model.

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put. If None, OTM convention is used.
            model: {'bsm', 'norm'} base model for implied vol inversion

        Returns:
            implied volatility smile
        """
        base_model = self._m_smile(model)
        if cp is None:
            fwd = self.forward(spot, texp)
            cp = np.where(strike > fwd, 1, -1)  # OTM convention
        price = self.price(strike, spot, texp, cp=cp)
        return base_model.impvol(price, strike, spot, texp, cp=cp)



class OptAnalyticABC(OptABC):
    """
    Option model with analytic price and greeks are available
    """

    THROW_NEGATIVE_TEXP = False

    @staticmethod
    @abc.abstractmethod
    def price_formula(strike, spot, sigma, texp, cp=1, *args, **kwargs):
        """
        Call/put option pricing formula (abstract/static method)

        Args:
            strike: strike price
            spot: spot (or forward) price
            sigma: model volatility
            texp: time to expiry
            cp: 1/-1 for call/put option
            *args:
            **kwargs:

        Returns:
            vanilla option price
        """
        raise NotImplementedError

    def price(self, strike, spot, texp, cp=1):
        if self.THROW_NEGATIVE_TEXP:
            assert ~np.any(texp < 0)

        return self.price_formula(
            strike=strike, spot=spot, texp=texp, cp=cp, **self.params_kw()
        )

    @abc.abstractmethod
    def delta(self, strike, spot, texp, cp=1):
        """
        Option model delta (sensitivity to price).

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            delta value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def gamma(self, strike, spot, texp, cp=1):
        """
        Option model gamma (2nd derivative to price).

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            gamma value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def vega(self, strike, spot, texp, cp=1):
        """
        Option model vega (sensitivity to volatility).

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            vega value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def theta(self, strike, spot, texp, cp=1):
        """
        Option model theta (sensitivity to time-to-maturity).

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            theta value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cdf(self, strike, spot, texp, cp=-1):
        """
        Cumulative distribution function of the final asset price.

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: -1 (default) for left-tail CDF, -1 for right-tail CDF (i.e., survival function)

        Returns:
            CDF value
        """
        raise NotImplementedError


class MassZeroABC(OptABC, abc.ABC):
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
        raise NotImplementedError

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
