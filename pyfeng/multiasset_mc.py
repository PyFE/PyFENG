import numpy as np
from .params import MaParams
from .opt_abc import OptABC
from .bsm import Bsm
from .multiasset import NormBasket, BsmBasketGeoApprox


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


class BsmBasketMc(MaParams, OptABC):
    """
    Monte Carlo pricer for basket options under the multiasset BSM model.

    The constructor and ``price()`` signature are identical to the analytical
    basket classes in ``multiasset.py``, making this a direct drop-in for
    benchmarking.  Numerical parameters are set via :meth:`configure`.

    Two control variates are available (``cv`` argument of :meth:`price`),
    both using a fixed hedge ratio ``beta = 1``:

    ``cv='geo'`` — geometric basket call as control:
        ``price = E_geo_call + mean(arith_call - geo_call)``

        The geometric basket forward ``E[G_T] = exp(w′log(F) + ½(σ_geo²−w′σ²)T)``
        is computed analytically, and the geometric call is priced via BSM.

    ``cv='norm'`` — Bachelier basket call as control:
        ``price = NormBasket.price + mean(bsm_call - bachelier_call)``

        The Bachelier payoffs are evaluated on the **same** random draws as the
        BSM paths (``S_T^{\\rm Norm} = F + z``, same ``z`` as BSM), so the two
        payoffs are maximally correlated.  :class:`NormBasket` supplies the exact
        analytical expectation.  Effective when ``σ√T`` is not too large.

    Examples:
        >>> import numpy as np, pyfeng as pf
        >>> m = pf.BsmBasketMc(np.ones(4)*0.4, rho=0.5).configure(n_path=100000, rn_seed=42)
        >>> m.price(np.arange(80, 121, 10, dtype=float), np.ones(4)*100, texp=5)
        array([36.47662759, 31.99768938, 28.12016612, 24.76925744, 21.86601311])
    """

    def __init__(self, sigma, rho=None, cor_m=None, cov_m=None, *,
                 weight=None, intr=0.0, divr=0.0, is_fwd=False,
                 rn_seed=None, antithetic=True):
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)
        self.antithetic = antithetic
        self.n_path = 0
        super().__init__(sigma, rho=rho, cor_m=cor_m, cov_m=cov_m,
                         weight=weight, intr=intr, divr=divr, is_fwd=is_fwd)

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
            self.antithetic = bool(antithetic)
        return self

    def price(self, strike, spot, texp, cp=1, cv='geo'):
        """
        Price a European basket call or put.

        Args:
            strike: scalar or array of strikes
            spot:   (n_asset,) array of current spot prices
            texp:   time to expiry
            cp:     +1 for call (default), -1 for put
            cv:     control variate — ``'geo'``, ``'norm'``, or ``None``

        Returns:
            Scalar price when ``strike`` is scalar; ndarray otherwise.
        """
        if not self.n_path:
            raise ValueError("n_path is not set. Call configure(n_path=...) first.")

        fwd, df, _ = self._fwd_df_divf(spot, texp)   # (n_asset,)

        # ── save rng state before drawing (needed for cv='geo' and cv='norm') ─
        if cv in ('geo', 'norm'):
            rng_state = self.rng.bit_generator.state

        # ── simulate terminal asset values ─────────────────────────────────
        n_gen = self.n_path // 2 if self.antithetic else self.n_path
        z = self.rng.standard_normal((n_gen, self.n_asset)) @ self.chol_m.T * np.sqrt(texp)
        if self.antithetic:
            z = np.concatenate([z, -z], axis=0)            # (n_path, n_asset)
        S_T = fwd * np.exp(-0.5 * self.sigma**2 * texp + z)  # (n_path, n_asset)
        B_T = S_T @ self.weight                             # (n_path,) arithmetic basket

        # ── control variate setup (computed once, outside the strike loop) ─
        if cv == 'geo':
            # BsmBasketGeoApproxMc with the rng state restored to before BSM draws,
            # so it generates the same random numbers as the BSM simulation.
            m_geo_mc = BsmBasketGeoApproxMc(
                sigma=self.sigma, cor_m=self.cor_m, weight=self.weight,
                intr=self.intr, divr=self.divr, is_fwd=True,
            ).configure(n_path=self.n_path, antithetic=self.antithetic)
            m_geo_mc.rng.bit_generator.state = rng_state
            # Analytical geometric basket price (exact expectation of the control)
            m_geo = BsmBasketGeoApprox(
                sigma=self.sigma, cor_m=self.cor_m, weight=self.weight,
                intr=self.intr, divr=self.divr, is_fwd=True,
            )
        elif cv == 'norm':
            # ATM normal vol approximation: sigma_norm = fwd * sigma_bsm
            sigma_norm = fwd * self.sigma
            # NormBasketMc with the rng state restored to before BSM draws,
            # so it generates the same random numbers as the BSM simulation.
            m_norm_mc = NormBasketMc(
                sigma=sigma_norm, cor_m=self.cor_m, weight=self.weight,
                intr=self.intr, divr=self.divr, is_fwd=True,
            ).configure(n_path=self.n_path, antithetic=self.antithetic)
            m_norm_mc.rng.bit_generator.state = rng_state
            # Analytical Bachelier basket price (exact expectation of the control)
            m_nb = NormBasket(
                sigma=sigma_norm, cor_m=self.cor_m, weight=self.weight,
                intr=self.intr, divr=self.divr, is_fwd=True,
            )

        # ── price each strike ───────────────────────────────────────────────
        scalar_strike = np.ndim(strike) == 0
        kk = np.atleast_1d(np.asarray(strike, dtype=float)).ravel()

        # BSM MC prices (common to all CV methods)
        bsm_mc = np.array([df * np.maximum(cp * (B_T - k), 0).mean() for k in kk])

        if cv == 'geo':
            # BsmBasketGeoApproxMc.price() generates z once for all strikes (same draws as BSM)
            geo_mc = m_geo_mc.price(kk, fwd, texp, cp)
            geo_exact = m_geo.price(kk, fwd, texp, cp)
            out = bsm_mc - (geo_mc - geo_exact)             # beta = 1
        elif cv == 'norm':
            # NormBasketMc.price() generates z once for all strikes (same draws as BSM)
            norm_mc = m_norm_mc.price(kk, fwd, texp, cp)
            norm_exact = m_nb.price(kk, fwd, texp, cp)
            out = bsm_mc - (norm_mc - norm_exact)           # beta = 1
        else:
            out = bsm_mc

        return float(out[0]) if scalar_strike else out


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


class NormBasketMc(BsmBasketMc):
    """
    Monte Carlo pricer for basket options under the multiasset Bachelier (Normal) model.

    Terminal asset prices follow arithmetic Brownian motion:
    ``S_T = F + z``  where ``z ~ N(0, T · Σ)`` and ``Σ = diag(σ) · C · diag(σ)``
    is the absolute covariance matrix.  ``σ`` must therefore be supplied as
    **absolute** volatilities (e.g. ``sigma = np.ones(4) * 40`` for 40-point vol
    on a spot-100 asset), consistent with :class:`NormBasket` and :class:`NormNdMc`.

    All numerical parameters (``n_path``, ``rn_seed``, ``antithetic``) and the
    constructor signature are inherited from :class:`BsmBasketMc`.

    No control variate is offered: under the Normal model the basket ``B_T = w′S_T``
    is exactly Gaussian, so :class:`NormBasket` already prices it analytically.
    This class is useful as a Monte Carlo cross-check or as a template for
    non-Gaussian extensions of the Normal model.

    Examples:
        >>> import numpy as np, pyfeng as pf
        >>> sigma = np.ones(4) * 40   # absolute vol: 40 pts on spot-100 assets
        >>> m = pf.NormBasketMc(sigma, rho=0.5).configure(n_path=100000, rn_seed=42)
        >>> m.price(np.arange(80, 121, 10, dtype=float), np.ones(4)*100, texp=5)
        array([39.40363659, 33.5644173 , 28.27867498, 23.5644173 , 19.40363659])
    """

    def price(self, strike, spot, texp, cp=1):
        """
        Price a European basket call or put under the Bachelier model.

        Args:
            strike: scalar or array of strikes
            spot:   (n_asset,) array of current spot prices
            texp:   time to expiry
            cp:     +1 for call (default), -1 for put

        Returns:
            Scalar price when ``strike`` is scalar; ndarray otherwise.
        """
        if not self.n_path:
            raise ValueError("n_path is not set. Call configure(n_path=...) first.")

        fwd, df, _ = self._fwd_df_divf(spot, texp)     # (n_asset,)

        # ── simulate terminal asset values (arithmetic BM) ─────────────────
        n_gen = self.n_path // 2 if self.antithetic else self.n_path
        z = self.rng.standard_normal((n_gen, self.n_asset)) @ self.chol_m.T * np.sqrt(texp)
        if self.antithetic:
            z = np.concatenate([z, -z], axis=0)         # (n_path, n_asset)
        S_T = fwd + z                                   # (n_path, n_asset)
        B_T = S_T @ self.weight                         # (n_path,) arithmetic basket

        # ── price each strike ───────────────────────────────────────────────
        scalar_strike = np.ndim(strike) == 0
        kk = np.atleast_1d(np.asarray(strike, dtype=float)).ravel()
        out = np.array([df * np.maximum(cp * (B_T - k), 0).mean() for k in kk])

        return float(out[0]) if scalar_strike else out


class BsmBasketGeoApproxMc(BsmBasketMc):
    """
    Monte Carlo pricer for the approximate geometric basket under the multiasset BSM model.

    Simulates the single-lognormal geometric basket used by :class:`BsmBasketGeoApprox`:

    .. math::

        G_T = (\\mathbf{w}^\\top \\mathbf{F})\\,
              \\exp\\!\\left(-\\tfrac{1}{2}\\sigma_{\\rm geo}^2 T + \\mathbf{w}^\\top z\\right)

    where :math:`\\sigma_{\\rm geo} = \\sqrt{\\mathbf{w}^\\top C\\mathbf{w}}` and
    :math:`\\mathbf{w}^\\top z` is the basket projection of the correlated BM draws.
    This gives :math:`E[G_T] = \\mathbf{w}^\\top \\mathbf{F}`, consistent with the
    arithmetic-forward proxy used by :class:`BsmBasketGeoApprox`.

    This is the MC analogue of :class:`BsmBasketGeoApprox`, in the same way that
    :class:`NormBasketMc` is the MC analogue of :class:`NormBasket`.  Because
    :math:`G_T` is exactly log-normal, results converge to :class:`BsmBasketGeoApprox`.

    All numerical parameters and the constructor signature are inherited from
    :class:`BsmBasketMc`.

    Examples:
        >>> import numpy as np, pyfeng as pf
        >>> sigma = np.ones(4) * 0.4
        >>> m = pf.BsmBasketGeoApproxMc(sigma, rho=0.5).configure(n_path=100000, rn_seed=42)
        >>> m.price(np.arange(80, 121, 10, dtype=float), np.ones(4)*100, texp=5)
        array([36.22924335, 31.72077966, 27.82454772, 24.46179606, 21.554812  ])
    """

    def price(self, strike, spot, texp, cp=1):
        """
        Price a European approximate geometric-basket call or put under BSM.

        Args:
            strike: scalar or array of strikes
            spot:   (n_asset,) array of current spot prices
            texp:   time to expiry
            cp:     +1 for call (default), -1 for put

        Returns:
            Scalar price when ``strike`` is scalar; ndarray otherwise.
        """
        if not self.n_path:
            raise ValueError("n_path is not set. Call configure(n_path=...) first.")

        fwd, df, _ = self._fwd_df_divf(spot, texp)     # (n_asset,)
        fwd_basket = fwd @ self.weight                  # w'F (arithmetic forward as proxy)
        vol_geo = np.sqrt(self.weight @ self.cov_m @ self.weight)

        # ── simulate approximate geometric basket ──────────────────────────
        n_gen = self.n_path // 2 if self.antithetic else self.n_path
        z = self.rng.standard_normal((n_gen, self.n_asset)) @ self.chol_m.T * np.sqrt(texp)
        if self.antithetic:
            z = np.concatenate([z, -z], axis=0)         # (n_path, n_asset)
        # Project z onto basket direction: w'z ~ N(0, vol_geo²·T)
        wz = z @ self.weight                            # (n_path,)
        G_T = fwd_basket * np.exp(-0.5 * vol_geo**2 * texp + wz)  # (n_path,)

        # ── price each strike ───────────────────────────────────────────────
        scalar_strike = np.ndim(strike) == 0
        kk = np.atleast_1d(np.asarray(strike, dtype=float)).ravel()
        out = np.array([df * np.maximum(cp * (G_T - k), 0).mean() for k in kk])

        return float(out[0]) if scalar_strike else out
