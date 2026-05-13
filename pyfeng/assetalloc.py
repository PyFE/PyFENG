import warnings
import numpy as np
import scipy.stats as spst
import scipy.optimize as spop
from dataclasses import dataclass, field

from .params import MaParams


@dataclass
class RiskParity(MaParams):
    """
    Risk parity (equal risk contribution) asset allocation.

    References:
        - Maillard S, Roncalli T, Teïletche J (2010) The Properties of Equally Weighted Risk Contribution Portfolios. The Journal of Portfolio Management 36:60–70. https://doi.org/10.3905/jpm.2010.36.4.060
        - Choi J, Chen R (2022) Improved iterative methods for solving risk parity portfolio. Journal of Derivatives and Quantitative Studies 30. https://doi.org/10.1108/JDQS-12-2021-0031

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> cov = np.array([
                [ 94.868, 33.750, 12.325, -1.178, 8.778 ],
                [ 33.750, 445.642, 98.955, -7.901, 84.954 ],
                [ 12.325, 98.955, 117.265, 0.503, 45.184 ],
                [ -1.178, -7.901, 0.503, 5.460, 1.057 ],
                [ 8.778, 84.954, 45.184, 1.057, 34.126 ]
            ])/10000

        >>> m = pf.RiskParity(cov_m=cov)
        >>> m.fit_ccd()
        array([0.125, 0.047, 0.083, 0.613, 0.132])
        >>> m._result
        {'err': 2.2697290741335863e-07, 'n_iter': 6}

        >>> m = pf.RiskParity(cov_m=cov, budget=np.array([0.1, 0.1, 0.2, 0.3, 0.3]))
        >>> m.fit_ccd()
        array([0.077, 0.025, 0.074, 0.648, 0.176])

        >>> m = pf.RiskParity(cov_m=cov, longshort=np.array([-1, -1, 1, 1, 1]))
        >>> m.fit_ccd()
        array([-0.216, -0.162,  0.182,  0.726,  0.47 ])
    """

    budget: np.ndarray | None = None
    longshort: np.ndarray | int = 1

    _result: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

        # ── budget ────────────────────────────────────────────────────────────
        if self.budget is None:
            self.budget = np.full(self.n_asset, 1 / self.n_asset)
        else:
            budget = np.asarray(self.budget, dtype=float)
            if len(budget) != self.n_asset:
                raise ValueError(f"budget length {len(budget)} does not match n_asset={self.n_asset}.")
            if not np.isclose(np.sum(budget), 1):
                raise ValueError(f"budget must sum to 1, got {np.sum(budget)}.")
            self.budget = budget

        # ── longshort ─────────────────────────────────────────────────────────
        if self.longshort is None:
            self.longshort = np.ones(self.n_asset, dtype=np.int8)
        elif np.isscalar(self.longshort):
            self.longshort = np.full(self.n_asset, np.sign(int(self.longshort)), dtype=np.int8)
        else:
            ls = np.asarray(self.longshort)
            if len(ls) != self.n_asset:
                raise ValueError(f"longshort length {len(ls)} does not match n_asset={self.n_asset}.")
            self.longshort = np.sign(ls).astype(np.int8)

    @classmethod
    def init_random(cls, n_asset=10, zero_ev=0, budget=False):
        """
        Randomly initialize the correlation matrix

        Args:
            n_asset: number of assets
            zero_ev: number of zero eivenvalues. 0 by default
            budget: randomize budget if True. False by default.

        Returns:
            RiskParity model object
        """
        ev = np.zeros(n_asset)
        ev[:n_asset-zero_ev] = np.random.uniform(size=n_asset - zero_ev)
        ev *= n_asset / np.sum(ev)
        cor = spst.random_correlation.rvs(ev, tol=1e-11)
        if not np.allclose(np.diag(cor), 1):
            raise ValueError("Correlation matrix diagonal must be all ones.")

        m = cls(cov_m=cor)
        return m

    def fit_ccd(self, tol=1e-6):
        """
        Risk parity weight using the improved CCD method of Choi and Chen (2022)

        Args:
            tol: error tolerance

        Returns:
            risk parity weight

        References:
            - Choi J, Chen R (2022) Improved iterative methods for solving risk parity portfolio. Journal of Derivatives and Quantitative Studies 30. https://doi.org/10.1108/JDQS-12-2021-0031
        """

        cor = self.cor_m
        ww = np.full(self.n_asset, 1 / np.sqrt(np.sum(cor)))

        for k in range(1, 1024):
            for (i, row) in enumerate(cor):
                a = (np.dot(row, ww) - ww[i]) / 2
                ww[i] = self.longshort[i] * np.sqrt(a * a + self.budget[i]) - a

            # Rescaling step
            cor_ww = cor @ ww
            vv = np.sqrt(np.dot(ww, cor_ww))
            cor_ww /= vv
            ww /= vv

            err = np.max(np.abs(ww * cor_ww - self.budget))
            if err < tol:
                ww /= self.sigma
                ww /= np.sum(ww)
                self._result = {'err': err, 'n_iter': k}
                self.weight = ww
                return ww

        # when not converged
        self._result = {'err': err, 'n_iter': k}
        return None

    def fit_ccd_original(self, tol=1e-6):
        """
        Risk parity weight using original CCD method of Griveau-Billion et al (2013).
        This is implemented for performance comparison. Use fit_ccd() for better performance.

        Args:
            tol: error tolerance

        Returns:
            risk parity weight

        References:
            - Griveau-Billion T, Richard J-C, Roncalli T (2013) A Fast Algorithm for Computing High-dimensional Risk Parity Portfolios. arXiv:13114057 [q-fin]

        """

        cov = self.cov_m
        ww = 1 / self.sigma
        ww /= np.sum(ww)
        cov_ww = cov @ ww
        vv = np.sqrt(np.dot(ww, cov_ww))

        for k in range(1, 1024):
            for i in range(self.n_asset):
                a = (cov_ww[i] - cov[i, i] * ww[i]) / 2
                wwi = (self.longshort[i] * np.sqrt(a * a + cov[i, i] * vv * self.budget[i]) - a) / cov[i, i]
                # update cov_ww, ww[i], and vv
                cov_ww += cov[:, i] * (wwi - ww[i])
                ww[i] = wwi
                vv = np.sqrt(np.dot(ww, cov_ww))

            err = np.max(np.abs(ww * cov_ww / vv - self.budget))
            if err < tol:
                ww /= np.sum(ww)
                self._result = {'err': err, 'n_iter': k}
                self.weight = ww
                return ww

        # when not converged
        self._result = {'err': err, 'n_iter': k}
        return None

    @staticmethod
    def _newton_val(w, cov, bud):
        # w = w/np.sqrt(np.sum(w*w))
        err = (cov @ w) - bud / w
        return err

    @staticmethod
    def _newton_jacobian(w, cov, bud):
        jac = cov + np.diag(bud / (w * w))
        return jac

    def fit_newton(self, tol=1e-6):
        """
        Risk parity weight using the 'improved' Newton method by Choi & Chen (2022).
        This is implemented for performance comparison. Use fit_ccd() for better performance.

        Args:
            tol: error tolerance

        Returns:
            risk parity weight

        References:
            - Spinu F (2013) An Algorithm for Computing Risk Parity Weights. SSRN Electronic Journal. https://doi.org/10.2139/ssrn.2297383
            - Choi J, Chen R (2022) Improved iterative methods for solving risk parity portfolio. Journal of Derivatives and Quantitative Studies 30. https://doi.org/10.1108/JDQS-12-2021-0031
        """
        cor = self.cor_m

        a = 0.5 * (np.sum(cor, axis=1) - 1) / np.sqrt(np.sum(cor))
        w_init = np.sqrt(a * a + self.budget) - a

        sol = spop.root(self._newton_val, w_init, (cor, self.budget), jac=self._newton_jacobian, tol=tol)
        # assert sol.success
        if not sol.success:
            warnings.warn("Newton solver failed to converge.")

        ww = sol.x / self.sigma
        ww /= np.sum(ww)
        err = np.max(np.abs(sol.fun))
        self._result = {'err': err, 'n_iter': sol.nfev}
        self.weight = ww
        return ww


class RiskParitySCA(RiskParity):
    """
    Successive convex approximation for capped risk parity portfolios.

    This class extends :class:`RiskParity` with a single additional model
    parameter, ``w_max``. All covariance, correlation, budget, and long/short
    inputs are inherited from :class:`RiskParity` and :class:`MaParams`.

    The fitted portfolio solves a long-only, equal-budget approximation of
    the volatility risk parity problem with the capped simplex constraints

    ``sum_i w_i = 1`` and ``0 <= w_i <= w_max``.

    For a covariance matrix ``Sigma``, the volatility risk contribution is
    proportional to ``w_i (Sigma w)_i``. Equal risk parity corresponds to
    Feng and Palomar (2015), equations (9) and (13). The squared risk
    contribution dispersion used here follows the formulation in their
    equation (16), while the iterative procedure is a projected-gradient
    implementation inspired by their successive convex approximation
    framework in equations (34)--(39) and Algorithm 1.

    The implementation initializes from :meth:`RiskParity.fit_ccd`, projects
    onto the capped simplex, and then repeats first-order projected steps
    with backtracking line search. It stores diagnostics in ``_result`` using
    the same style as :class:`RiskParity`.

    Args:
        w_max: upper bound on each asset weight. Must satisfy
            ``w_max >= 1 / n_asset``.

    References:
        - Feng Y, Palomar DP (2015). SCRIP: Successive Convex Optimization
          Methods for Risk Parity Portfolio Design. IEEE Transactions on
          Signal Processing 63(19), 5285-5300.
          https://doi.org/10.1109/TSP.2015.2452219
        - Choi J, Chen R (2022). Improved iterative methods for solving risk
          parity portfolio. Journal of Derivatives and Quantitative Studies
          30(2), 114-124. https://doi.org/10.1108/JDQS-12-2021-0031
    """

    tol = 1e-6
    max_iter = 200

    def __init__(self, *args, w_max=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        if not np.allclose(self.budget, 1.0 / self.n_asset):
            raise ValueError("RiskParitySCA currently supports equal budgets only.")
        if np.any(self.longshort != 1):
            raise ValueError(
                "RiskParitySCA currently supports long-only portfolios only."
            )

        self.w_max = float(w_max)
        if not np.isfinite(self.w_max) or self.w_max <= 0.0 or self.w_max > 1.0:
            raise ValueError("w_max must lie in (0, 1].")
        if self.w_max * self.n_asset < 1.0 - 1e-12:
            raise ValueError(
                f"w_max={self.w_max} is infeasible for n_asset={self.n_asset}: "
                "need w_max >= 1/n_asset."
            )

    def fit_sca(self, tol=None):
        """
        Compute constrained risk parity weights by SCA.

        Args:
            tol: optional convergence tolerance override.

        Returns:
            Constrained risk parity weights.

        Raises:
            FloatingPointError: if CCD initialization or SCA fails.
        """
        tol_eff = self.tol if tol is None else float(tol)
        if not np.isfinite(tol_eff) or tol_eff <= 0.0:
            raise ValueError("tol must be a positive finite float.")
        max_iter = int(self.max_iter)
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        w = self.fit_ccd(tol=min(tol_eff, 1e-8))
        if w is None:
            raise FloatingPointError("RiskParity CCD initialization failed to converge.")

        w = self._project_simplex_box(np.clip(w, 0.0, self.w_max), self.w_max)
        current_obj = self._objective(w)
        step = 1.0
        err = np.inf

        for k in range(1, max_iter + 1):
            w_old = w.copy()
            sw = self.cov_m @ w
            rc = w * sw
            resid = rc - rc.mean()
            grad = 2.0 * (sw * resid + w * (self.cov_m @ resid))

            local_step = step
            while True:
                candidate = self._project_simplex_box(
                    w - local_step * grad,
                    self.w_max,
                )
                candidate_obj = self._objective(candidate)
                if current_obj - candidate_obj >= 1e-12 or local_step <= 1e-12:
                    break
                local_step *= 0.5

            w = candidate
            current_obj = candidate_obj
            step = min(local_step * 1.5, 1.0)
            err = float(np.linalg.norm(w - w_old, ord=np.inf))
            if err < tol_eff:
                break

        converged = err < tol_eff
        if not converged:
            raise FloatingPointError("RiskParitySCA failed to converge.")

        self.weight = w
        self._result = {
            "err": err,
            "n_iter": k,
            "objective": current_obj,
            "gap": self._risk_contribution_gap(w),
        }
        return w

    def _objective(self, w):
        rc = self._risk_contributions(w)
        return float(np.sum((rc - rc.mean()) ** 2))

    def _risk_contributions(self, w):
        return w * (self.cov_m @ w)

    def _risk_contribution_gap(self, w):
        rc = self._risk_contributions(w)
        return float(np.max(np.abs(rc - rc.mean())))

    @staticmethod
    def _project_simplex_box(v, u):
        """Project ``v`` onto ``{w : sum(w)=1, 0 <= w <= u}``."""
        v = np.asarray(v, dtype=float)
        if v.ndim != 1:
            raise ValueError("v must be a 1-D array.")
        if not np.all(np.isfinite(v)):
            raise ValueError("v must contain only finite values.")
        if not np.isfinite(u) or u <= 0.0:
            raise ValueError("u must be a positive finite float.")
        n = v.size
        if u * n < 1.0 - 1e-12:
            raise ValueError("Infeasible box for simplex projection.")

        lo, hi = v.min() - u, v.max()
        for _ in range(100):
            tau = 0.5 * (lo + hi)
            w = np.clip(v - tau, 0.0, u)
            total = w.sum()
            if abs(total - 1.0) < 1e-12:
                return w
            if total > 1.0:
                lo = tau
            else:
                hi = tau
        return np.clip(v - 0.5 * (lo + hi), 0.0, u)
