

import abc
import numpy as np
import scipy.stats as spst
import scipy.optimize as spop
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import sys

sys.path.insert(0, '/Users/liyafen/Documents/GitHub/PyFENG/pyfeng')
#import pyfeng as pf
import pyfeng.ex as pfex
from assetalloc import AssetAllocABC

class RiskParity2(AssetAllocABC):

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

    def optimization_of_convex_problem(self,x0,rho,cov_m,bnds):
        """
        Solve the optimization of convex problem for the minimum variance weight extended least square.

        Args:
            x: weights for asset allocation problem.
            rho: the weight parameter of the convex term in the formula.
            cov_m:
            bnds: boundary likes (a,b)
        """
        #fix theta:
        # list_bnds = []
        # for i in range(len(x0)):
        #     list_bnds.append(bnds)
        # bnds = tuple(list_bnds)
        # cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        # rp = [0.2,0.2,0.2,0.2,0.2]
        # func = lambda x: np.sum((x * (cov_m @ x) / (x.T @ cov_m @ x) - rp) ** 2 + rho*(x.T @ cov_m @ x))
        # res = minimize(func, x0, method='SLSQP', constraints=cons,bounds=bnds, tol=1e-10)
        # weight = res.x

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
            (x[:int(len(x)/2)] * (cov_m @ x[:int(len(x)/2)]) - x[int(len(x)/2):]) ** 2 + rho * (x[:int(len(x)/2)].T @ cov_m @ x[:int(len(x)/2)]))

        res = minimize(func,[x0,rp], method='SLSQP', constraints=cons, bounds=bnds, tol=1e-10)
        weight = res.x[:int(len(res.x)/2)]

        return weight

    def minimum_variance_weight_extended_least_square(self, rho=1000, beta=0.01, tol=1e-6, itreation_max=100,a=-1,b=2):
        """
        Minimum variance with risk parity using the Extended least-squares models of Bai et al.(2016)

        Args:
            rho: the weight parameter of the convex term in the formula.
            beta: the scholar for decreasing rho.
            tol: error tolerance.

        Returns:
            risk parity weight

        References:
            -Bai X, Scheinberg K, Tutuncu R (2016) Least-squares approach to risk parity in portfolio selection. Quantitative Finance 16:357–376.
        """
        #Algorithm 1 in 4.1

        bonds = (a,b)
        cor = self.cor_m
        cov_m = self.cov_m
        x = np.full(self.n_asset, 1 / np.sqrt(np.sum(cor)))
        for k in range(0,itreation_max):
            if(rho >= tol):
                x = self.optimization_of_convex_problem(x,rho,cov_m,bonds)
                rho = rho * beta
            else:
                x = self.optimization_of_convex_problem(x, 0, cov_m,bonds)
                break
        return x










if __name__ == '__main__':
    import numpy as np
    import pyfeng as pf
    cov = np.array([
        [94.868, 33.750, 12.325, -1.178, 8.778],
        [33.750, 445.642, 98.955, -7.901, 84.954],
        [12.325, 98.955, 117.265, 0.503, 45.184],
        [-1.178, -7.901, 0.503, 5.460, 1.057],
        [8.778, 84.954, 45.184, 1.057, 34.126]
    ]) / 10000


    m = RiskParity2(cov=cov)
    weight = m.minimum_variance_weight_extended_least_square(rho=1000, beta=0.01, tol=1e-6, itreation_max=100,a=-1,b=2)
    print(weight)
    #[ 0.05027209  0.00553736 -0.01230498  0.85650904  0.09998649]
    cov = np.array([
        [1,-0.9,0.6],
        [-0.9,1.0,-0.2],
        [0.6,-0.2,4.0]
    ])

    m = RiskParity2(cov=cov)
    weight = m.minimum_variance_weight_extended_least_square(rho=1000, beta=0.01, tol=1e-6, itreation_max=100, a=-1,
                                                             b=2)
    print(weight)
    #[ 0.53233682  0.51447136 -0.04680817]

