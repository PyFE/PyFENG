import numpy as np
from . import opt_abc as opt


class BsmNdMc(opt.OptMaABC):
    """
    Monte-Carlo simulation of multiasset (N-d) BSM (geometric Brownian Motion)

    Examples:
        >>> import pyfeng as pf
        >>> spot = np.ones(4)*100
        >>> sigma = np.ones(4)*0.4
        >>> texp = 5
        >>> payoff = lambda x: np.fmax(np.mean(x,axis=1) - strike, 0) # Basket option
        >>> strikes = np.arange(80, 121, 10)
        >>> m = pf.BsmNdMc(sigma, cor=0.5, rn_seed=1234)
        >>> m.simulate(tobs=[texp], n_path=20000)
        >>> p = []
        >>> for strike in strikes:
        >>>    p.append(m.price_european(spot, texp, payoff))
        >>> np.array(p)
        array([36.31612946, 31.80861014, 27.91269315, 24.55319506, 21.62677625])
    """

    spot = np.ones(2)
    sigma = np.ones(2)*0.1

    # MC params
    n_path = 100
    rn_seed = None
    rng = None
    antithetic = True

    # path
    path, tobs = None, None

    def __init__(self, sigma, cor=None, intr=0.0, divr=0.0, rn_seed=None):
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)
        super().__init__(sigma, cor=cor, intr=intr, divr=divr, is_fwd=False)

    def set_mc_params(self, n_path, rn_seed=None, antithetic=True):
        self.n_path = n_path
        self.rn_seed = rn_seed
        self.antithetic = antithetic

    def _bm_incr(self, tobs, n_path=None):
        """
        Calculate incremental Brownian Motions

        Args:
            tobs: array of observation times
            n_path: number of paths. If None (default), use the stored one.
            store: if True (default), save the result to self.path_stored

        Returns:
            price path (time, path, asset)
        """
        dt = np.diff(np.atleast_1d(tobs), prepend=0)
        n_t = len(dt)

        n_path = self.n_path if n_path is None else n_path

        if self.antithetic:
            # generate random number in the order of path, time, asset and transposed
            # in this way, the same paths are generated when increasing n_path
            bm_incr = self.rng.normal(size=(n_path//2, n_t, self.n_asset)).transpose((1, 0, 2))
            bm_incr *= np.sqrt(dt[:, None, None])
            bm_incr = np.dot(bm_incr, self.chol_m.T)
            bm_incr = np.stack([bm_incr, -bm_incr], axis=2).reshape((n_t, n_path, self.n_asset))
        else:
            bm_incr = np.random.randn(n_path, n_t, self.n_asset).transpose((1, 0, 2)) * np.sqrt(dt[:, None, None])
            bm_incr = np.dot(bm_incr, self.chol_m.T)

        return bm_incr

    def simulate(self, tobs, n_path=None, store=True):
        """
        Simulate the price paths and store in the class.
        The initial prices are normalized to 1.

        Args:
            tobs: array of observation times
            n_path: number of paths. If None (default), use the stored one.
            store: if True (default), save the result to self.path_stored

        Returns:
            price path (time, path, asset)
        """
        # (n_t, n_path, n_asset) * (n_asset, n_asset)
        path = self._bm_incr(tobs, n_path)
        # Add drift and convexity
        dt = np.diff(np.atleast_1d(tobs), prepend=0)
        path += (self.intr - self.divr - 0.5*self.sigma**2)*dt[:, None, None]
        np.cumsum(path, axis=0, out=path)
        np.exp(path, out=path)

        if store:
            self.n_path = n_path
            self.path = path
            self.tobs = tobs

        return path

    def price_european(self, spot, texp, payoff):
        """
        The European price of that payoff at the expiry.

        Args:
            spot: array of spot prices
            texp: time-to-expiry
            payoff: payoff function applicable to the time-slice of price path

        Returns:
            The MC price of the payoff
        """
        if self.path is None:
            raise ValueError('Simulated paths are not available. Run simulate() first.')

        # check if texp is in tobs
        ind, *_ = np.where(np.isclose(self.tobs, texp))
        if len(ind) == 0:
            raise ValueError(f'Stored path does not contain t = {texp}')

        path = self.path[ind[0], ] * spot
        price = np.exp(-self.intr * texp) * np.mean(payoff(path), axis=0)
        return price


class NormNdMc(BsmNdMc):
    """
    Monte-Carlo simulation of multiasset (N-d) Normal/Bachelier model (arithmetic Brownian Motion)

    Examples:
        >>> import pyfeng as pf
        >>> spot = np.ones(4)*100
        >>> sigma = np.ones(4)*0.4
        >>> texp = 5
        >>> payoff = lambda x: np.fmax(np.mean(x,axis=1) - strike, 0) # Basket option
        >>> strikes = np.arange(80, 121, 10)
        >>> m = pf.NormNdMc(sigma*spot, cor=0.5, rn_seed=1234)
        >>> m.simulate(tobs=[texp], n_path=20000)
        >>> p = []
        >>> for strike in strikes:
        >>>    p.append(m.price_european(spot, texp, payoff))
        >>> np.array(p)
        array([39.42304794, 33.60383167, 28.32667559, 23.60383167, 19.42304794])
    """

    def simulate(self, tobs, n_path=None, store=True):
        path = self._bm_incr(tobs, n_path)
        np.cumsum(path, axis=0, out=path)

        if store:
            self.n_path = n_path
            self.path = path
            self.tobs = tobs

        return path

    def price_european(self, spot, texp, payoff):
        if self.path is None:
            raise ValueError('Simulated paths are not available. Run simulate() first.')

        # check if texp is in tobs
        ind, *_ = np.where(np.isclose(self.tobs, texp))
        if len(ind) == 0:
            raise ValueError(f'Stored path does not contain t = {texp}')

        path = self.path[ind[0], ] + spot
        price = np.exp(-self.intr * texp) * np.mean(payoff(path), axis=0)
        return price