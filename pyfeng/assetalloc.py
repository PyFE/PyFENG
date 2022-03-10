import abc
import numpy as np
import scipy.stats as spst
import scipy.optimize as spop


class AssetAllocABC(abc.ABC):

    n_asset = 1
    rho = None
    sigma = np.array([1.0])
    ret = np.array([0.0])
    cor_m = np.eye(1)
    cov_m = np.eye(1)
    longshort = np.array([1], dtype=np.int8)

    def __init__(self, sigma=None, cor=None, cov=None, ret=None, longshort=1):
        """

        Args:
            sigma: asset volatilities of `n_asset` assets. (n_asset, ) array
            cor: asset correlation. If matrix with shape (n_asset, n_asset), used as it is.
                If scalar, correlation matrix is constructed with all same off-diagonal values.
            cov: asset covariance
            ret: expected return
            longshort: long/short constraint. 1 for long-only, -1 for short, 0 for no constraint
        """

        if cov is None:
            # when sigma and cor are given

            self.sigma = np.atleast_1d(sigma)
            self.n_asset = len(self.sigma)

            if self.n_asset == 1:
                raise ValueError(f"The number of assets should be more than one.")

            if np.isscalar(cor):
                self.cor_m = cor * np.ones((self.n_asset, self.n_asset)) \
                             + (1 - cor) * np.eye(self.n_asset)
                self.rho = cor
            else:
                assert cor.shape == (self.n_asset, self.n_asset)
                self.cor_m = cor
                if self.n_asset == 2:
                    self.rho = cor[0, 1]

            self.cov_m = self.sigma * self.cor_m * sigma[:, None]
        else:
            # When covariance is given directly
            self.n_asset = cov.shape[0]

            self.cov_m = cov
            self.sigma = np.sqrt(np.diag(cov))
            self.cor_m = cov.copy()
            self.cor_m /= self.sigma[:, None]
            self.cor_m /= self.sigma

        if ret is not None:
            self.ret = ret * np.ones(self.n_asset)

        if longshort is None:
            self.longshort = np.full(self.n_asset, 1, dtype=np.int8)  # long-only
        elif np.isscalar(longshort):
            self.longshort = np.full(self.n_asset, np.sign(longshort), dtype=np.int8)  # long-only
        else:
            assert self.n_asset == len(longshort)
            self.longshort = np.sign(longshort, dtype=np.int8)  # long-only


class RiskParity(AssetAllocABC):
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
        
        >>> m = pf.RiskParity(cov=cov)
        >>> m.weight()
        array([0.125, 0.047, 0.083, 0.613, 0.132])
        >>> m._result
        {'err': 2.2697290741335863e-07, 'n_iter': 6}

        >>> m = pf.RiskParity(cov=cov, budget=[0.1, 0.1, 0.2, 0.3, 0.3])
        >>> m.weight()
        array([0.077, 0.025, 0.074, 0.648, 0.176])

        >>> m = pf.RiskParity(cov=cov, longshort=[-1, -1, 1, 1, 1])
        >>> m.weight()
        array([-0.216, -0.162,  0.182,  0.726,  0.47 ])
    """

    budget = None
    _result = {}

    def __init__(self, sigma=None, cor=None, cov=None, ret=None, budget=None, longshort=1):
        """

        Args:
            sigma: asset volatilities of `n_asset` assets. (n_asset, ) array
            cor: asset correlation. If matrix with shape (n_asset, n_asset), used as it is.
                If scalar, correlation matrix is constructed with all same off-diagonal values.
            cov: asset covariance
            ret: expected return
            budget: risk bugget. 1/n_asset if not specified.
            longshort: long/short constraint. 1 for long-only (default), -1 for short, 0 for no constraint
        """
        super().__init__(sigma=sigma, cor=cor, cov=cov, ret=ret, longshort=longshort)
        if budget is None:
            self.budget = np.full(self.n_asset, 1 / self.n_asset)
        else:
            assert self.n_asset == len(budget)
            assert np.isclose(np.sum(budget), 1)
            self.budget = np.array(budget)

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
        assert np.allclose(np.diag(cor), 1)

        m = cls(cov=cor)
        return m

    def weight(self, tol=1e-6):
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
                return ww

        # when not converged
        self._result = {'err': err, 'n_iter': k}
        return None

    def weight_ccd_original(self, tol=1e-6):
        """
        Risk parity weight using original CCD method of Griveau-Billion et al (2013).
        This is implemented for performance comparison. Use weight() for better performance.

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

    def weight_newton(self, tol=1e-6):
        """
        Risk parity weight using the 'improved' Newton method by Choi & Chen (2022).
        This is implemented for performance comparison. Use weight() for better performance.

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
            print("Newton Failed.")

        ww = sol.x / self.sigma
        ww /= np.sum(ww)
        err = np.max(np.abs(sol.fun))
        self._result = {'err': err, 'n_iter': sol.nfev}
        return ww
