import abc
import copy
import numpy as np
import scipy.optimize as sopt


class OptABC(abc.ABC):
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
        pass

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

        scalar_output = np.isscalar(price) & np.isscalar(p_min)
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
        return np.minimum(1 / 365.25, texp)  # one day

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
        dt = self._delta_shock(strike, spot, texp, cp)
        theta = self.price(strike, spot, texp - dt, cp) - self.price(
            strike, spot, texp, cp
        )
        theta /= dt
        return theta

    def pdf_numeric(self, strike, spot, texp, cp=-1, h=0.001):
        """
        Probability density functin (PDF) at `strike`

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            probability densitiy
        """
        fwd = spot * (1.0 if self.is_fwd else np.exp(texp * (self.intr - self.divr)))
        kk = strike / fwd
        kk_arr = np.array([kk - h, kk, kk + h]).flatten()
        price = self.price(kk_arr, 1, texp, cp=cp)
        price = price.reshape(3, -1)
        pdf = (price[2] + price[0] - 2.0 * price[1]) / (h * h)
        return pdf

    # create aliases
    delta = delta_numeric
    gamma = gamma_numeric
    vega = vega_numeric
    vanna = vanna_numeric
    volga = volga_numeric
    theta = theta_numeric


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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass


class OptMaABC(OptABC, abc.ABC):

    n_asset = 1
    rho = None
    cor_m = np.diag([1.0])
    cov_m = np.diag([1.0])
    chol_m = np.diag([1.0])

    def __init__(self, sigma, cor=None, intr=0.0, divr=0.0, is_fwd=False):
        """

        Args:
            sigma: model volatilities of `n_asset` assets. (n_asset, ) array
            cor: correlation. If matrix with shape (n_asset, n_asset), used as it is.
                If scalar, correlation matrix is constructed with all same off-diagonal values.
            intr: interest rate (domestic interest rate)
            divr: vector of dividend/convenience yield (foreign interest rate) 0-D or (n_asset, ) array
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        sigma = np.atleast_1d(sigma)
        self.n_asset = len(sigma)

        super().__init__(sigma, intr, divr, is_fwd)

        if self.n_asset == 1:
            if cor is not None:
                print(f"Ignoring cor={cor} for a single asset")
            self.rho = None
            self.cor_m = np.array([[1.0]])
        elif np.isscalar(cor):
            self.cor_m = cor * np.ones((self.n_asset, self.n_asset)) + (
                1 - cor
            ) * np.eye(self.n_asset)
            self.rho = cor
        else:
            assert cor.shape == (self.n_asset, self.n_asset)
            self.cor_m = cor
            if self.n_asset == 2:
                self.rho = cor[0, 1]

        self.cov_m = sigma * self.cor_m * sigma[:, None]
        self.chol_m = np.linalg.cholesky(self.cov_m)

    def price(self, strike, spot, texp, cp=1):
        """
        Call/put option price.

        Args:
            strike: strike price.
            spot: spot (or forward) prices for assets.
                Asset dimension should be the last, e.g. (n_asset, ) or (N, n_asset)
            texp: time to expiry.
            cp: 1/-1 for call/put option.

        Returns:
            option price
        """
        pass
