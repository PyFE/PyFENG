import numpy as np
from . import multiasset as ma
from .quad import NdGHQ  # Not sure
from .multiasset import BsmBasket1Bm


class BsmBasketChoi2018(ma.NormBasket):
    """
    Choi (2018)'s pricing method for Basket/Spread/Asian options

    References
        - Choi J (2018) Sum of all Black-Scholes-Merton models: An efficient pricing method for spread, basket, and Asian options. Journal of Futures Markets 38:627–644. https://doi.org/10.1002/fut.21909
    """

    n_quad = None
    lam = 4.0

    def set_num_params(self, n_quad=None, lam=3.0):
        self.n_quad = n_quad
        self.lam = lam

    @staticmethod
    def householder(vv0):
        """
        Returns a Householder reflection (orthonormal matrix) that maps (1,0,...0) to vv0
        
        Args:
            vv0: vector

        Returns:
            Reflection matrix

        References
            - https://en.wikipedia.org/wiki/Householder_transformation
        """
        vv1 = vv0 / np.linalg.norm(vv0)
        vv1[0] -= 1.0

        if abs(vv1[0]) < np.finfo(float).eps*100:
            return np.eye(len(vv1))
        else:
            return np.eye(len(vv1)) + vv1[:, None] * vv1 / vv1[0]

    def v_mat(self, fwd):
        """
        Construct the V matrix

        Args:
            fwd: forward vector of assets

        Returns:
            V matrix
        """

        fwd_wts = fwd * self.weight

        v1 = self.cov_m @ fwd_wts
        v1 /= np.sqrt(np.sum(v1 * fwd_wts))

        thres = 0.01 * self.sigma
        idx = (np.sign(fwd_wts) * v1 < thres)

        if np.any(idx):
            v1[idx] = (np.sign(fwd_wts) * thres)[idx]
            q1 = np.linalg.solve(self.chol_m, v1)
            q1norm = np.linalg.norm(q1)
            q1 /= q1norm
            v1 /= q1norm
        else:
            q1 = self.chol_m.T @ fwd_wts
            q1 /= np.linalg.norm(q1)

        r_mat = self.householder(q1)

        chol_r_mat = self.chol_m @ r_mat[:, 1:]
        svd_u, svd_d, _ = np.linalg.svd(chol_r_mat, full_matrices=False)

        v_mat = np.hstack((v1[:, None], svd_u @ np.diag(svd_d)))
        return v_mat

    def v1_fwd_weight(self, fwd, texp):
        """
        Construct v1, forward array, and weights

        Args:
            fwd: forward vector of assets
            texp: time to expiry

        Returns:
            (v1, f_k, ww)
        """
        v_mat = self.v_mat(fwd) * np.sqrt(texp)
        print(np.round(v_mat, 2))
        v1 = v_mat[:, 0]
        v_mat = v_mat[:, 1:len(self.n_quad)+1].T

        quad = NdGHQ(self.n_quad)
        zz, ww = quad.z_vec_weight()
        f_k = np.exp(zz @ v_mat - 0.5*np.sum(v_mat**2, axis=0))

        return v1, f_k, ww

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)

        v1, f_k, ww = self.v1_fwd_weight(fwd, texp)
        m_1bm = BsmBasket1Bm(sigma=v1, weight=self.weight)

        price = np.zeros_like(strike, dtype=float)
        for k, f_k_row in enumerate(f_k):
            price1 = m_1bm.price(strike, f_k_row * fwd, texp=1.0, cp=cp)
            price += price1 * ww[k]

        return df * price
