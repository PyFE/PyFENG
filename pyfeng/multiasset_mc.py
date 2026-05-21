import numpy as np
from .params import MaParams


class BsmNdMc(MaParams):
    """
    Monte-Carlo simulation of multiasset (N-d) BSM (geometric Brownian Motion)

    Examples:
        >>> import pyfeng as pf
        >>> spot = np.ones(4) * 100
        >>> sigma = np.ones(4) * 0.4
        >>> texp = 5
        >>> payoff = lambda x: np.fmax(np.mean(x, axis=1) - strike, 0)  # Basket option
        >>> strikes = np.arange(80, 121, 10)
        >>> m = pf.BsmNdMc(sigma, rho=0.5).configure(n_path=20000, rn_seed=1234).simulate([texp])
        >>> p = [m.price_european(spot, texp, payoff) for strike in strikes]
        >>> np.array(p)
        array([36.31612946, 31.80861014, 27.91269315, 24.55319506, 21.62677625])
    """

    spot = np.ones(2)
    sigma = np.ones(2) * 0.1

    def __init__(self, sigma, rho=None, cor_m=None, cov_m=None, intr=0.0, divr=0.0, rn_seed=None, antithetic=True):
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)
        self.antithetic = antithetic
        # per-instance simulation state (not class-level, to avoid shared-state bug)
        self.n_path = 0
        self.path = None
        self.tobs = None
        super().__init__(sigma, rho=rho, cor_m=cor_m, cov_m=cov_m, intr=intr, divr=divr, is_fwd=False)

    def configure(self, n_path=None, rn_seed=None, antithetic=None):
        """
        Set numerical simulation parameters.

        Args:
            n_path: number of MC paths
            rn_seed: random seed (reseeds the RNG when provided)
            antithetic: use antithetic variates (default True)

        Returns:
            self  (for method chaining)
        """
        if n_path is not None:
            self.n_path = int(n_path)
        if rn_seed is not None:
            self.rn_seed = rn_seed
            self.rng = np.random.default_rng(rn_seed)
        if antithetic is not None:
            self.antithetic = antithetic
        return self

    def _bm_incr(self, tobs, n_path):
        """
        Calculate incremental Brownian Motions

        Args:
            tobs: array of observation times
            n_path: number of paths to simulate

        Returns:
            BM increments (time, path, asset)
        """
        dt = np.diff(np.atleast_1d(tobs), prepend=0)
        n_t = len(dt)

        n_path_gen = n_path // 2 if self.antithetic else n_path

        # generate random numbers in (path, time, asset) order then transpose;
        # this ensures the same paths are produced when n_path is increased
        bm_incr = self.rng.standard_normal((n_path_gen, n_t, self.n_asset)).transpose((1, 0, 2))
        np.multiply(bm_incr, np.sqrt(dt[:, None, None]), out=bm_incr)
        bm_incr = np.dot(bm_incr, self.chol_m.T)
        if self.antithetic:
            bm_incr = np.stack([bm_incr, -bm_incr], axis=2).reshape(
                (n_t, n_path, self.n_asset)
            )

        return bm_incr

    def simulate(self, tobs, store=True):
        """
        Build simulated price paths.

        The number of paths is taken from ``self.n_path`` (set via
        :meth:`configure`).  The initial price is normalised to 1; multiply
        by ``spot`` when evaluating payoffs.

        Args:
            tobs: observation times (scalar or array)
            store: if True (default), store path and tobs; return self.
                   if False, return the path array without storing.

        Returns:
            self when store=True (for method chaining), or
            path array (time, path, asset) when store=False.

        Raises:
            ValueError: if n_path has not been set via configure().
        """
        if not self.n_path:
            raise ValueError("n_path is not set. Call configure(n_path=...) first.")

        tobs = np.atleast_1d(tobs)
        path = self._bm_incr(tobs=tobs, n_path=self.n_path)
        # drift + convexity correction
        dt = np.diff(tobs, prepend=0)
        path += (self.intr - self.divr - 0.5 * self.sigma ** 2) * dt[:, None, None]
        np.cumsum(path, axis=0, out=path)
        np.exp(path, out=path)

        if store:
            self.path = path
            self.tobs = tobs
            return self
        else:
            return path

    def price_european(self, spot, texp, payoff):
        """
        Price a European payoff using stored paths.

        Args:
            spot: array of spot prices
            texp: time-to-expiry (must be one of the stored tobs)
            payoff: function of the (path, asset) slice at texp

        Returns:
            discounted MC price
        """
        if self.path is None:
            raise ValueError("No paths stored. Call simulate() first.")

        ind, *_ = np.where(np.isclose(self.tobs, texp))
        if len(ind) == 0:
            raise ValueError(f"Stored tobs does not contain t={texp}")

        path = self.path[ind[0], ] * spot
        return np.exp(-self.intr * texp) * np.mean(payoff(path), axis=0)


class NormNdMc(BsmNdMc):
    """
    Monte-Carlo simulation of multiasset (N-d) Normal/Bachelier model (arithmetic Brownian Motion)

    Examples:
        >>> import pyfeng as pf
        >>> spot = np.ones(4) * 100
        >>> sigma = np.ones(4) * 0.4
        >>> texp = 5
        >>> payoff = lambda x: np.fmax(np.mean(x, axis=1) - strike, 0)  # Basket option
        >>> strikes = np.arange(80, 121, 10)
        >>> m = pf.NormNdMc(sigma * spot, rho=0.5).configure(n_path=20000, rn_seed=1234).simulate([texp])
        >>> p = [m.price_european(spot, texp, payoff) for strike in strikes]
        >>> np.array(p)
        array([39.42304794, 33.60383167, 28.32667559, 23.60383167, 19.42304794])
    """

    def simulate(self, tobs, store=True):
        """
        Build simulated price paths (arithmetic BM; add spot to get absolute prices).

        Args:
            tobs: observation times (scalar or array)
            store: if True (default), store path and tobs; return self.
                   if False, return the path array without storing.

        Returns:
            self when store=True (for method chaining), or
            path array (time, path, asset) when store=False.

        Raises:
            ValueError: if n_path has not been set via configure().
        """
        if not self.n_path:
            raise ValueError("n_path is not set. Call configure(n_path=...) first.")

        tobs = np.atleast_1d(tobs)
        path = self._bm_incr(tobs, self.n_path)
        np.cumsum(path, axis=0, out=path)

        if store:
            self.path = path
            self.tobs = tobs
            return self
        else:
            return path

    def price_european(self, spot, texp, payoff):
        """
        Price a European payoff using stored paths.

        Args:
            spot: array of spot prices
            texp: time-to-expiry (must be one of the stored tobs)
            payoff: function of the (path, asset) slice at texp

        Returns:
            discounted MC price
        """
        if self.path is None:
            raise ValueError("No paths stored. Call simulate() first.")

        ind, *_ = np.where(np.isclose(self.tobs, texp))
        if len(ind) == 0:
            raise ValueError(f"Stored tobs does not contain t={texp}")

        path = self.path[ind[0], ] + spot
        return np.exp(-self.intr * texp) * np.mean(payoff(path), axis=0)
