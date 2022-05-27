import numpy as np
import pyfeng.multiasset as ma


class BsmNdMc(ma.OptMaABC):
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
        >>> m.simulate(n_path=20000, tobs=[texp])
        >>> p = []
        >>> for strike in strikes:
        >>>    p.append(m.price_european(spot, texp, payoff))
        >>> np.array(p)
        array([36.31612946, 31.80861014, 27.91269315, 24.55319506, 21.62677625])
    """

    spot = np.ones(2)
    sigma = np.ones(2) * 0.1

    # MC params
    rn_seed = None
    rng = None
    antithetic = True

    # tobs and path stored in the class
    n_path = 0
    path = tobs = None

    def __init__(self, sigma, cor=None, intr=0.0, divr=0.0, rn_seed=None, antithetic=True):
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)
        self.antithetic = antithetic
        super().__init__(sigma, cor=cor, intr=intr, divr=divr, is_fwd=False)

    def _bm_incr(self, tobs, n_path):
        """
        Calculate incremental Brownian Motions

        Args:
            tobs: array of observation times
            n_path: number of paths to simulate

        Returns:
            price path (time, path, asset)
        """
        dt = np.diff(np.atleast_1d(tobs), prepend=0)
        n_t = len(dt)

        n_path_gen = n_path // 2 if self.antithetic else n_path

        # generate random number in the order of path, time, asset and transposed
        # in this way, the same paths are generated when increasing n_path
        bm_incr = self.rng.standard_normal((n_path_gen, n_t, self.n_asset)).transpose((1, 0, 2))
        np.multiply(bm_incr, np.sqrt(dt[:, None, None]), out=bm_incr)
        bm_incr = np.dot(bm_incr, self.chol_m.T)
        if self.antithetic:
            bm_incr = np.stack([bm_incr, -bm_incr], axis=2).reshape(
                (n_t, n_path, self.n_asset)
            )

        return bm_incr

    def simulate(self, tobs, n_path, store=True):
        """
        Simulate the price paths and store in the class.
        The initial prices are normalized to 0 and spot should be multiplied later.

        Args:
            tobs: array of observation times
            n_path: number of paths to simulate
            store: if True (default), store path, tobs, and n_path in the class

        Returns:
            price path (time, path, asset) if store is False
        """

        # (n_t, n_path, n_asset) * (n_asset, n_asset)
        tobs = np.atleast_1d(tobs)
        path = self._bm_incr(tobs=tobs, n_path=n_path)
        # Add drift and convexity
        dt = np.diff(tobs, prepend=0)
        path += (self.intr - self.divr - 0.5 * self.sigma ** 2) * dt[:, None, None]
        np.cumsum(path, axis=0, out=path)
        np.exp(path, out=path)

        if store:
            self.n_path = n_path
            self.path = path
            self.tobs = tobs
        else:
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
        if self.n_path == 0:
            raise ValueError("Simulated paths are not available. Run simulate() first.")

        # check if texp is in tobs
        ind, *_ = np.where(np.isclose(self.tobs, texp))
        if len(ind) == 0:
            raise ValueError(f"Stored tobs does not contain t={texp}")

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

    def simulate(self, tobs, n_path, store=True):
        """
        Simulate the price paths and store in the class.
        The initial prices are normalized to 0 and spot should be added later.

        Args:
            tobs: array of observation times
            n_path: number of paths to simulate
            store: if True (default), store path, tobs, and n_path in the class

        Returns:
            price path (time, path, asset) if store is False
        """
        tobs = np.atleast_1d(tobs)
        path = self._bm_incr(tobs, n_path)
        np.cumsum(path, axis=0, out=path)

        if store:
            self.n_path = n_path
            self.path = path
            self.tobs = tobs
        else:
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
        if self.n_path == 0:
            raise ValueError("Simulated paths are not available. Run simulate() first.")

        # check if texp is in tobs
        ind, *_ = np.where(np.isclose(self.tobs, texp))
        if len(ind) == 0:
            raise ValueError(f"Stored tobs does not contain t={texp}")

        path = self.path[ind[0], ] + spot
        price = np.exp(-self.intr * texp) * np.mean(payoff(path), axis=0)
        return price
