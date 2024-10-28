# -*- coding: utf-8 -*-
"""
Created on August 24, 2024

@author: Jaehyuk Choi
"""

import abc
import numpy as np
import scipy.special as spsp
import scipy.integrate as spint
from . import sv_abc as sv
from . import rheston


class RoughHestonMcMaWu2022(rheston.RoughHestonABC, sv.CondMcBsmABC, abc.ABC):
    """
    Simulation using Gaussian quadrature based on Ma and Wu (2022)

    References:
        - Ma & Wu (2022) A fast algorithm for simulation of rough volatility models, Quantitative Finance, 22:3, 447-462, DOI: 10.1080/14697688.2021.1970213
    """

    def get_nodes_weights(self, err_tol, scale_coef):
        """
        Nodes and weights for the entire approximation

        Args:
            err_tol: absolute error tolerance for the approximation of the kernel function
            scale_coef: scaling coefficient of the number of nodes

        Returns:
            x_all: nodes
            omega_all: weights
        """
        M, N, n_o, n_s, n_l = self.get_num_nodes(err_tol, scale_coef)

        x_o, omega_o = self.GaussJacobiQuad(n_o, M)
        i = np.arange(-int(M), 0)
        x_s, omega_s = self.GaussLegendreQuad(n_s, i[:, None])
        j = np.arange(0, int(N) + 1)
        x_l, omega_l = self.GaussLegendreQuad(n_l, j[:, None])

        x_all = np.concatenate((x_o, x_s.reshape(-1), x_l.reshape(-1)))
        omega_all = np.concatenate((omega_o, omega_s.reshape(-1), omega_l.reshape(-1)))

        N_all = int(n_o + M * n_s + (N + 1) * n_l)
        assert N_all == len(x_all)
        assert len(x_all) == len(omega_all)

        return x_all, omega_all

    def get_num_nodes(self, err_tol=1e-4, scale_coef=1):
        """
        Number of nodes for the Gaussian quadrature

        Args:
            err_tol: absolute error tolerance for the approximation of the kernel function
            scale_coef: scaling coefficient of the number of nodes

        References:
            - Algorithm 2 in Ma & Wu (2022)

        Returns:
            M: value of $M$ for the integration on $[0, 2^{-M}]$
            N: value of $N$ for the cutoff of the integral approximation
            n_o: number of nodes for the Gauss-Jacobi quadrature on the interval $[0, 2^{-M}]$
            n_s: number of nodes for the Gauss-Legendre quadrature on the interval $[2^{j}, 2^{j+1}], j=-M, \cdots, -1$
            n_l: number of nodes for the Gauss-Legendre quadrature on the interval $[2^{j}, 2^{j+1}], j=0, \cdots, N$
        """
        M = scale_coef * np.fmax(np.ceil(np.log(self.texp)), 0) + 1
        N = scale_coef * np.ceil(np.log(np.log(1 / err_tol) / self.dt))
        n_o = scale_coef * np.ceil(np.log(1 / err_tol))
        n_s = scale_coef * np.ceil(np.log(1 / err_tol))
        n_l = scale_coef * np.ceil(np.log(1 / err_tol / self.dt))

        return M, N, n_o, n_s, n_l

    #set_num_params(self, n_path=10000, dt=None, rn_seed=None, antithetic=True, kk=1):
    def set_num_params(self, n_path=10000, dt=1/50, rn_seed=None, antithetic=True, texp=1, err_tol=1e-4, scale_coef=1):
        super().set_num_params(n_path, dt, rn_seed, antithetic)

        self.texp = texp
        # Frequently used constants
        self._gamma_a = spsp.gamma(self.alpha)
        self._gamma_1ma = spsp.gamma(1 - self.alpha)
        self._gamma_2ma = spsp.gamma(2 - self.alpha)
        self._gamma_1pa = spsp.gamma(1 + self.alpha)

        self.dt_pow1 = np.power(self.dt, 1 - self.alpha) / self._gamma_2ma
        self.dt_pow2 = np.power(self.dt, 0.5 - self.alpha) / np.sqrt(1 - 2*self.alpha)

        self.x_all, self.omega_all = self.get_nodes_weights(err_tol, scale_coef)
        self.x_exp = np.exp(-self.x_all * self.dt)


    def drift_func(self, var_t):
        """
        The drift term of the rough Heston model $f(var_t) = \kappa (\theta - var_t)$

        Args:
            var_t: volatility at time s

        Returns:
            drift term
        """
        return self.mr * (self.theta - var_t)

    def vol_func(self, var_t):
        """
        The diffusion term of the rough Heston model $g(var_t) = \kappa \epsilon * \sqrt(var_t)$

        Args:
            var_t: volatility at time s
        
        Returns:
            diffusion term
        """
        return self.vov * np.sqrt(var_t)
    
    def term_1(self, t_n, t_k):
        """
        The term $\left[\left(t_n-t_{k-1}\right)^{1-\alpha}-\left(t_n-t_k\right)^{1-\alpha}\right]$

        Args:
            t_n: time at $t_n$
            t_k: time at $t_k$

        References:
            - Equation (8) in Ma & Wu (2022)

        Returns:
            value of the term
        """
        return (t_n - t_k + self.dt) ** (1 - self.alpha) - (t_n - t_k) ** (1 - self.alpha)
    
    def term_2(self, t_n, t_k):
        """
        The term $\left[\frac{\left(t_n-t_{k-1}\right)^{1-2 \alpha}-\left(t_n-t_k\right)^{1-2 \alpha}}{1-2 \alpha}\right]^{1 / 2}$

        Args:
            t_n: time at $t_n$
            t_k: time at $t_k$

        References:
            - Equation (8) in Ma & Wu (2022)

        Returns:
            value of the term
        """
        return np.sqrt(((t_n - t_k + self.dt) ** (1 - 2 * self.alpha) - (t_n - t_k) ** (1 - 2 * self.alpha)) / (1 - 2 * self.alpha))
    
    def GaussJacobiQuad(self, n, M):
        """
        Nodes and weights for the Gauss-Jacobi quadrature

        Args:
            n: number of nodes
            M: value of $M$ for the integration on $[0, 2^{-M}]$
        
        Returns:
            s_o: nodes
            omega_o: weights
        """
        s_o, w_o = spsp.roots_jacobi(n, 0, self.alpha-1)
        s_o = 2.0 ** (-M - 1) * (s_o + 1)
        w_o = 2.0 ** ((-M - 1) * self.alpha) * w_o
        omega_o = w_o / self._gamma_a
        
        return s_o, omega_o
    
    def GaussLegendreQuad(self, n, j):
        """
        Nodes and weights for the Gauss-Legendre quadrature

        Args:
            n: number of nodes
            j: index for the interval

        Returns:
            s_i: nodes
            omega_i: weights
        """
        s_i, w_i = spsp.roots_legendre(n)
        s_i = 2.0 ** (j - 1) * (s_i + 3)
        w_i = s_i ** (self.alpha - 1) * w_i * 2.0 ** (j - 1)
        omega_i = w_i / self._gamma_a

        return s_i, omega_i

    def H_N(self, H_previous, V_previous):
        """
        Update rule of the auxiliary function $H_{l}^{N}(t_n)$

        Args:
            x_all: all nodes of the Gaussian quadrature approximation
            H_previous: $H_{l}^{N}(t_{n-1})$
            V_previous: $V_{t_{n-1}}$
        
        References:
            - Equation (18) in Ma & Wu (2022)

        Returns:
            updated function: $H_{l}^{N}(t_n)$
        """
        return (1 - self.x_exp) / self.x_all * self.drift_func(V_previous)[:, None] + self.x_exp * H_previous
    
    def J_N(self, J_previous, V_previous, Z_t):
        """
        Update rule of the auxiliary function $J_{l}^{N}(t_n)$

        A correction for the paper: in the first line of `J_N`, `np.sqrt((1 - np.exp(-2 * x_all * self.dt)) / (2 * x_all))` is added due to Ito's Isometry

        Args:
            x_all: all nodes of the Gaussian quadrature approximation
            J_previous: $J_{l}^{N}(t_{n-1})$
            V_previous: $V_{t_{n-1}}$
            Z_t: random normal variables for the current time step

        References:
            - Equation (19) in Ma & Wu (2022)

        Returns:
            updated function: $J_{l}^{N}(t_n)$
        """
        # A correction for the paper: in the first line of `J_N`, `np.sqrt((1 - np.exp(-2 * x_all * self.dt)) / (2 * x_all))` is added due to Ito's Isometry
        return np.sqrt(((1 - self.x_exp**2) / (2 * self.x_all))) * (self.vol_func(V_previous) * Z_t)[:, None] + self.x_exp * J_previous
    
    def eta_j(self, N_exp):
        """
        $\eta_j$ for the Multifactor approximation

        Args:
            N_exp: number of factors

        References:
            - Equation (27) in Ma & Wu (2022)

        Returns:
            array of $\eta_j$
        """
        j = np.arange(N_exp + 1)
        return j * N_exp ** (-1 / 5) / self.texp * (np.sqrt(10) * self.alpha / (2 + self.alpha)) ** (2 / 5)
    
    def c_j_gamma_j(self, N_exp):
        """
        $c_j$ and $\gamma_j$ for the Multifactor approximation

        Args:
            N_exp: number of factors

        References:
            - Equation (27) in Ma & Wu (2022)

        Returns:
            tuple of arrays of $c_j$ and $\gamma_j$
        """
        eta = self.eta_j(N_exp)
        cj = (eta[1:] ** self.alpha - eta[:-1] ** self.alpha) / self._gamma_1ma / self._gamma_1pa
        gammaj = (eta[1:] ** (self.alpha + 1) - eta[:-1] ** (self.alpha + 1)) / (eta[1:] ** self.alpha - eta[:-1] ** self.alpha) * self.alpha / (1 + self.alpha)

        return cj, gammaj
    
    def V_tJ(self, V_tj_previous, V_previous, gammaj, Z_t):
        """
        Update rule of the factors $V_{t_{n}}^{\tilde{N}_{\text{exp}}, j, N}$ for the Multifactor approximation

        Args:
            V_tj_previous: $V_{t_{n-1}}^{\tilde{N}_{\text{exp}}, j}$
            V_previous: $V_{t_{n}}^{\tilde{N}_{\text{exp}}, N}$
            gammaj: $\gamma_j$
            Z_t: random normal variables for the current time step

        References:
            - Algorithm 4 in Ma & Wu (2022)

        Returns:
            updated function: $V_{t_{n}}^{\tilde{N}_{\text{exp}}, j, N}$
        """
        return (1 - gammaj * self.dt) * V_tj_previous + (-self.mr * V_previous * self.dt + self.vol_func(V_previous) * np.sqrt(self.dt) * Z_t)[:, None]
    
    def MultifactorApprox(self, Z_t):
        """
        Simulation of the rough Heston model using the Multifactor approximation

        Args:
            Z_t: random normal variables for the simulation of the volatility process

        References:
            - Algorithm 4 in Ma & Wu (2022)

        Returns:
            V_t: simulated volatility process
        """
        N_exp = int(np.ceil(self.n_ts ** (5 / 4)))
        cj, gammaj = self.c_j_gamma_j(N_exp)

        V_t = np.zeros((self.n_ts + 1, self.n_path))
        V_t[0, :] = self.sigma
        V_tj = np.zeros((self.n_path, N_exp))
        for i in range(self.n_ts):
            V_tj = self.V_tJ(V_tj, V_t[i, :], gammaj, Z_t[i, :])
            V_t[i + 1, :] = self.sigma + self.mr * self.theta * (cj / gammaj * (1 - np.exp(-gammaj * self.tgrid[i + 1]))).sum() + (cj * V_tj).sum(axis=1)

        return V_t

    def cond_states_step(self, dt, var_t, H_t, J_t):
        """
        Simulation of the rough Heston model using the Fast algorithm

        Args:
            Z_t: random normal variables for the simulation of the volatility process
            err_tol: absolute error tolerance for the approximation of the kernel function
            scale_coef: scaling coefficient of the number of nodes

        References:
            - Algorithm 2 in Ma & Wu (2022)

        Returns:
            V_t: simulated volatility process
        """
        assert dt == self.dt
        Z = self.rng_spawn[0].standard_normal(size=self.n_path)
        Y_t = np.sqrt(var_t * dt) * Z
        var_t_new = self.sigma + self.dt_pow1 * self.drift_func(var_t) \
                    + ((self.x_exp * self.omega_all * (H_t + J_t)).sum(axis=1) + self.dt_pow2 * self.vol_func(var_t) * Z) / self._gamma_1ma
        H_t = self.H_N(H_t, var_t)
        J_t = self.J_N(J_t, var_t, Z)

        return var_t_new, H_t, J_t, Y_t

    def cond_spot_sigma(self, texp, var_0):
        """
        Returns new forward and volatility conditional on volatility path (e.g., sigma_T, integrated variance)
        The forward and volatility are standardized in the sense that F_0 = 1 and sigma_0 = 1
        Therefore, they should be scaled by the original F_0 and sigma_0 values.
        Volatility, not variance, is returned.

        Args:
            spot: spot price
            V_t: variance paths
            Z_t: the BMs driving the volatility process (needed for integration)
            correct_fwd: martingale preserving control variate

        Returns:
            cond_forward: conditional forward
            cond_sigma: conditional volatility
        """

        assert (texp == self.texp)

        tobs = self.tobs(texp)
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        var_t = np.full(self.n_path, var_0)
        H_t = np.zeros(self.n_path)
        J_t = np.zeros(self.n_path)
        Y_0_T = np.zeros(self.n_path)
        avgvar = 0.5 * dt * var_t

        for i in range(n_dt):
            var_t, H_t, J_t, Y_t = self.cond_states_step(dt[i], var_t, H_t, J_t)
            Y_0_T += Y_t
            if i < n_dt-1:
                avgvar += dt * var_t
            else:
                avgvar += 0.5 * dt * var_t

        avgvar /= texp
        spot_cond = np.exp(self.rho * Y_0_T - 0.5 * self.rho ** 2 * avgvar * texp)
        sigma_cond = np.sqrt((1.0 - self.rho**2) / var_0 * avgvar)  # normalize by initial variance
        # return normalized forward and volatility
        return spot_cond, sigma_cond

    def price_path(self, spot, V_t, W_t, strike, cp=1):
        """
        Stock price paths and prices of European options

        Args:
            spot: spot price
            V_t: simulated volatility process
            W_t: random normal variables for the simulation of the stock price process
            strike: strike price

        Returns:
            S_t: simulated stock price process
            price: prices of European options
        """
        disc_fac = np.exp(-self.texp * self.intr)

        X_t = np.zeros((self.n_ts, self.n_path))
        X_t[0, :] = np.log(spot)
        for i in range(self.n_ts - 1):
            X_t[i + 1, :] = X_t[i, :] + (
                        self.intr - 0.5 * V_t[i, :] * self.dt + np.sqrt(V_t[i, :] * self.dt) * W_t[i, :])

        S_t = np.exp(X_t)

        if isinstance(strike, (int, float)):
            return S_t, disc_fac * np.fmax(0.0, cp * (S_t[-1, :] - strike)).mean()
        elif isinstance(strike, np.ndarray):
            return S_t, disc_fac * np.fmax(0.0, cp * (S_t[-1, :] - strike[:, None])).mean(axis=1)
        else:
            raise ValueError("Strike price must be a scalar or a numpy array")

    def return_var_realized(self, texp, cond):
        return NotImplementedError
        
