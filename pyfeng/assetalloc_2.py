
import abc
import numpy as np
import scipy.stats as spst
import scipy.optimize as spop
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import sys

#import pyfeng as pf
import pyfeng.ex as pfex
from pyfeng.assetalloc import AssetAllocABC


class RiskParity2(AssetAllocABC):
    """
    Risk parity (equal risk contribution) asset allocation with general bounds.

    References:
        -Bai X, Scheinberg K, Tutuncu R (2016) Least-squares approach to risk parity in portfolio selection. Quantitative Finance 16:357–376.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> cov = np.array([
        >>> [1,-0.9,0.6],
        >>> [-0.9,1.0,-0.2],
        >>> [0.6,-0.2,4.0]
        >>> ])
        >>> m = pfex.RiskParity2(cov=cov)
        >>> weight = m.general_risk_parity_with_fixed_theta(a=-1,b=2)
        >>> print(weight)
        # [0.45520246 0.4805526  0.06424494]
        >>> weight = m.general_risk_parity_with_variable_theta(a=-1,b=2)
        >>> print(weight)
        # [0.26339982 0.19102125 0.54557893]
        >>> weight = m.minimum_variance_risk_parity_extended_least_square(rho=1000, beta=0.01, tol=1e-6, itreation_max=100, a=-1,b=2)
        >>> print(weight)
        # [ 0.53233682  0.51447136 -0.04680817]

        >>> weight = m.general_risk_parity_with_fixed_theta(a=-0.05, b=0.35)
        >>> print(weight)
        # [0.35 0.35 0.3 ]
        >>> weight = m.general_risk_parity_with_variable_theta(a=-0.05, b=0.35)
        >>> print(weight)
        # [0.33692795 0.31616628 0.34690577]
        >>> weight = m.minimum_variance_risk_parity_extended_least_square(rho=1000, beta=0.01, tol=1e-6, itreation_max=100, a=-0.05, b=0.35)
        >>> print(weight)
        # [0.35, 0.35, 0.3 ]

    """

    def __init__(self, sigma=None, cor=None, cov=None, ret=None, budget=None, longshort=0):
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
        # if budget is None:
        #     self.budget = np.full(self.n_asset, 1 / self.n_asset)
        # else:
        #     assert self.n_asset == len(budget)
        #     assert np.isclose(np.sum(budget), 1)
        #     self.budget = np.array(budget)

    def weight_general(self,x0,cov_m,bnds):
        """
            Solve the optimization problem for the general risk parity problem with fixed theta.

            Args:
                x0: weights for asset allocation problem.
                cov_m: asset covariance
                bnds: boundary likes (a,b), means that weight x satisfy a<=x<=b

            Returns:
                risk parity weight
        """
        # --formula (16) --
        # fix theta:
        n_assets = len(x0)
        rp = np.full(n_assets, 1 / n_assets)

        list_bnds = []
        for i in range(len(x0)):
            list_bnds.append(bnds)
        bnds = tuple(list_bnds)
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        func = lambda x: np.sum((x * (cov_m @ x) / (x.T @ cov_m @ x) - rp) ** 2 )
        res = minimize(func, x0, method='SLSQP', constraints=cons,bounds=bnds, tol=1e-10)
        weight = res.x
        return weight

    def general_risk_parity_with_fixed_theta(self,a=-1,b=2):
        """
            Least-squares risk parity model with general bounds of Bai et al.(2016)
            ---Implementation of 2.3 formula (16)---

            Args:
                a: lower bound, means that weight x satisfy a<=x
                b: upper bound, means that weight x satisfy x<=b

            Returns:
                risk parity weight

            References:
                -Bai X, Scheinberg K, Tutuncu R (2016) Least-squares approach to risk parity in portfolio selection. Quantitative Finance 16:357–376.
        """
        # implementation of general risk parity formula with fixed theta in 2.3
        bonds = (a, b)
        cor = self.cor_m
        cov_m = self.cov_m
        x0 = np.full(self.n_asset, 1 / np.sqrt(np.sum(cor)))
        return self.weight_general(x0,cov_m,bonds)

    def weight_general_2(self,x0,cov_m,bnds):
        """
            Solve the optimization problem for the general
            risk parity problem with variable theta.

            Args:
                x0: weights for asset allocation problem.
                cov_m: asset covariance
                bnds: boundary likes (a,b), means that weight x satisfy a<=x<=b

            Returns:
                risk parity weight
        """
        # --formula (17) --
        #variable theta:
        n_assets = len(x0)
        list_bnds = []
        for i in range(n_assets):
            list_bnds.append(bnds)
        for i in range(n_assets):
            list_bnds.append((None, None))
        bnds = tuple(list_bnds)

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[:int(len(x) / 2)]) - 1.0}]
        rp = np.full(n_assets, 1 / n_assets)
        rp = (x0.T @ cov_m @ x0) * rp

        func = lambda x: np.sum(
            (x[:int(len(x) / 2)] * (cov_m @ x[:int(len(x) / 2)]) - x[int(len(x) / 2):]) ** 2)

        res = minimize(func, [x0, rp], method='SLSQP', constraints=cons, bounds=bnds, tol=1e-10)
        weight = res.x[:int(len(res.x) / 2)]

        return weight

    def general_risk_parity_with_variable_theta(self,a=-1,b=2):
        """
            Least-squares risk parity model with general bounds of Bai et al.(2016)
            ---Implementation of 2.3 formula (17)---

            Args:
                a: lower bound, means that weight x satisfy a<=x
                b: upper bound, means that weight x satisfy x<=b

            Returns:
                risk parity weight

            References:
                -Bai X, Scheinberg K, Tutuncu R (2016) Least-squares approach to risk parity in portfolio selection. Quantitative Finance 16:357–376.
        """
        # implementation of general risk parity formula with variable theta in 2.3
        bonds = (a, b)
        cor = self.cor_m
        cov_m = self.cov_m
        x0 = np.full(self.n_asset, 1 / np.sqrt(np.sum(cor)))
        return self.weight_general_2(x0, cov_m, bonds)

    def weight_minimum_variance(self,x0,rho,cov_m,bnds):
        """
        Solve the optimization problem for the minimum variance weight extended least square.

        Args:
            x0: weights for asset allocation problem.
            rho: the weight parameter of the convex term in the formula.
            cov_m: asset covariance
            bnds: boundary likes (a,b), means that weight x satisfy a<=x<=b

        Returns:
            risk parity weight
        """
        # --formula (28) --
        n_assets = len(x0)
        list_bnds = []
        for i in range(n_assets):
            list_bnds.append(bnds)
        for i in range(n_assets):
            list_bnds.append((None,None))
        bnds = tuple(list_bnds)

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[:int(len(x)/2)]) - 1.0}]
        rp = np.full(n_assets, 1 / n_assets)
        rp = (x0.T @ cov_m @ x0) * rp

        func = lambda x: np.sum(
            (x[:int(len(x)/2)] * (cov_m @ x[:int(len(x)/2)])
             - x[int(len(x)/2):]) ** 2 + rho * (x[:int(len(x)/2)].T @ cov_m @ x[:int(len(x)/2)]))

        res = minimize(func,[x0,rp], method='SLSQP', constraints=cons, bounds=bnds, tol=1e-10)
        weight = res.x[:int(len(res.x)/2)]

        return weight

    def minimum_variance_risk_parity_extended_least_square(self, rho=1000, beta=0.01, tol=1e-6, itreation_max=100, a=-1, b=2):
        """
        Minimum variance with risk parity using the Extended least-squares models of Bai et al.(2016)
            ---Implementation of 4.1 formula (28)---
        Args:
            rho: the weight parameter of the convex term in the formula.
            beta: the scholar for decreasing rho.
            tol: error tolerance.
            itreation_max: The maximum number of times the weight is calculated
            a: lower bound, means that weight x satisfy a <= x
            b: upper bound, means that weight x satisfy x <= b

        Returns:
            risk parity weight with minimum variance

        References:
            -Bai X, Scheinberg K, Tutuncu R (2016) Least-squares approach to risk parity in portfolio selection. Quantitative Finance 16:357–376.
        """
        # Algorithm 1 in 4.1
        bonds = (a,b)
        cor = self.cor_m
        cov_m = self.cov_m
        x = np.full(self.n_asset, 1 / np.sqrt(np.sum(cor)))
        for k in range(0,itreation_max):
            if(rho >= tol):
                x = self.weight_minimum_variance(x,rho,cov_m,bonds)
                rho = rho * beta
            else:
                x = self.weight_minimum_variance(x, 0, cov_m,bonds)
                break
        return x