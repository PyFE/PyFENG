# -*- coding: utf-8 -*-
"""
Created on May 2, 2024

@author: Enze Zhou, Vahan Geghamyan
"""

import abc
import numpy as np
import scipy.special as spsp
import scipy.integrate as spint
from . import sv_abc as sv
from . import rheston


class RoughHestonMcABC(rheston.RoughHestonABC, sv.CondMcBsmABC, abc.ABC):
    
    def __init__(self, V_0, rho, kappa, epsilon, theta, alpha, intr=0.0, divr=0.0) -> None:
        """
        Args:
            V_0: initial volatility
            rho: correlation
            kappa: mean reversion speed
            epsilon: volatility of volatility
            theta: long term volatility
            alpha: modified Hurst exponent (\alpha = -(H - 0.5))
            S_0: initial stock price
            texp: time to maturity
            intr: interest rate
            divr: dividend rate
        """
        super().__init__(V_0, kappa * epsilon, rho, kappa, theta, alpha, intr, divr)

    def set_num_params(self, texp, n_path=10000, n_ts=1000, rn_seed=None, antithetic=True):
        super().set_num_params(n_path, n_ts, rn_seed, antithetic)

        self.texp = texp
        self.n_ts = int(n_ts)
        self.dt = self.texp / self.n_ts
        self.tgrid = np.linspace(0, self.texp, self.n_ts + 1)

        # Frequently used constants
        self._gamma_a = spsp.gamma(self.alpha)
        self._gamma_1ma = spsp.gamma(1 - self.alpha)
        self._gamma_2ma = spsp.gamma(2 - self.alpha)
        self._gamma_1pa = spsp.gamma(1 + self.alpha)


class RoughHestonMcMaWu2022(RoughHestonMcABC):
    """
    Simulation using Gaussian quadrature based on Ma and Wu (2022)

    References:
        - Ma & Wu (2022) A fast algorithm for simulation of rough volatility models, Quantitative Finance, 22:3, 447-462, DOI: 10.1080/14697688.2021.1970213
    """

    def __init__(self, V_0, rho, kappa, epsilon, theta, alpha, intr=0.0, divr=0.0) -> None:
        super().__init__(V_0, rho, kappa, epsilon, theta, alpha, intr, divr)
        self.V_0 = self.sigma
        self.kappa = self.mr
        self.epsilon = self.vov / self.kappa

    def f(self, V_s):
        """
        The drift term of the rough Heston model $f(V_s) = \kappa (\theta - V_s)$

        Args:
            V_s: volatility at time s

        Returns:
            drift term
        """
        return self.kappa * (self.theta - V_s)

    def g(self, V_s):
        """
        The diffusion term of the rough Heston model $g(V_s) = \kappa \epsilon * \sqrt(V_s)$

        Args:
            V_s: volatility at time s
        
        Returns:
            diffusion term
        """
        return self.kappa * self.epsilon * np.sqrt(V_s)
    
    def random_normals(self):
        """
        Generate random normal variables for the simulation

        Args:
            None

        Returns:
            Z_t1: random normal variables for the simulation of the volatility process
            W_t: random normal variables for the simulation of the stock price process
        """
        Z_t1 = self.rng_spawn[0].normal(size=(self.n_ts, self.n_path))
        W_t = self.rho * Z_t1 + np.sqrt(1 - self.rho**2) * self.rng_spawn[1].normal(size=(self.n_ts, self.n_path))
        return Z_t1, W_t
    
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
    
    def ModifiedEM(self, Z_t):
        """
        Simulation of the rough Heston model using the modified Euler-Maruyama algorithm

        Args:
            Z_t: random normal variables for the simulation of the volatility process
        
        References:
            - Algorithm 1 in Ma & Wu (2022)

        Returns:
            V_t: simulated volatility process
        """
        assert Z_t.shape == (self.n_ts, self.n_path)

        V_t = np.zeros((self.n_ts + 1, self.n_path))
        V_t[0] = self.V_0

        for i in range(self.n_ts):
            V_t[i + 1, :] = self.V_0
            j = np.arange(1, i + 2)
            summation_f = (self.f(V_t[:i + 1, :]) * self.term_1(self.tgrid[i + 1], self.tgrid[j])[:, np.newaxis]).sum(axis=0)
            V_t[i + 1, :] += 1 / self._gamma_2ma * summation_f
            summation_g = (self.g(V_t[:i + 1, :]) * self.term_2(self.tgrid[i + 1], self.tgrid[j])[:, np.newaxis] * Z_t[:i + 1, :]).sum(axis=0)
            V_t[i + 1, :] += 1 / self._gamma_1ma * summation_g

        return V_t
    
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
        s_o, w_o = spsp.roots_jacobi(n, self.alpha-1, 0)
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
        x_s, omega_s = self.GaussLegendreQuad(n_s, i[:, np.newaxis])
        j = np.arange(0, int(N) + 1)
        x_l, omega_l = self.GaussLegendreQuad(n_l, j[:, np.newaxis])

        x_all = np.concatenate((x_o, x_s.reshape(-1), x_l.reshape(-1)))
        omega_all = np.concatenate((omega_o, omega_s.reshape(-1), omega_l.reshape(-1)))

        N_all = int(n_o + M * n_s + (N + 1) * n_l)
        assert N_all == len(x_all)
        assert len(x_all) == len(omega_all)

        return x_all, omega_all
    
    def H_N(self, x_all, H_previous, V_previous):
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
        return 1 / x_all * (1 - np.exp(-x_all * self.dt)) * self.f(V_previous)[:, np.newaxis] + np.exp(-x_all * self.dt) * H_previous
    
    def J_N(self, x_all, J_previous, V_previous, Z_t):
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
        return np.sqrt(((1 - np.exp(-2 * x_all * self.dt)) / (2 * x_all))) * (self.g(V_previous) * Z_t)[:, np.newaxis] + np.exp(-x_all * self.dt) * J_previous
    
    def Fast(self, Z_t, err_tol=1e-4, scale_coef=1):
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
        x_all, omega_all = self.get_nodes_weights(err_tol=err_tol, scale_coef=scale_coef)

        V_t = np.zeros((self.n_ts + 1, self.n_path))
        V_t[0, :] = self.V_0
        N_all = len(x_all)

        H_t = np.zeros((self.n_path, N_all))
        J_t = np.zeros((self.n_path, N_all))

        for i in range(self.n_ts):
            V_t[i + 1, :] = self.V_0 + self.dt ** (1 - self.alpha) / self._gamma_2ma * self.f(V_t[i, :]) \
                            + 1 / self._gamma_1ma * (omega_all * np.exp(-x_all * self.dt) * H_t).sum(axis=1) \
                            + self.dt ** (0.5 - self.alpha) / np.sqrt(1 - 2 * self.alpha) / self._gamma_1ma * self.g(V_t[i, :]) * Z_t[i, :] \
                            + 1 / self._gamma_1ma * (omega_all * np.exp(-x_all * self.dt) * J_t).sum(axis=1)
            H_t = self.H_N(x_all, H_t, V_t[i, :])
            J_t = self.J_N(x_all, J_t, V_t[i, :], Z_t[i, :])
        # A correction for the paper: in the third line of `V_t[i + 1, :]`, `1 / np.sqrt(1 - 2 * self.alpha)` is added due to Ito's Isometry

        return V_t
    
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
        return (1 - gammaj * self.dt) * V_tj_previous + (-self.kappa * V_previous * self.dt + self.g(V_previous) * np.sqrt(self.dt) * Z_t)[:, np.newaxis]
    
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
        V_t[0, :] = self.V_0
        V_tj = np.zeros((self.n_path, N_exp))
        for i in range(self.n_ts):
            V_tj = self.V_tJ(V_tj, V_t[i, :], gammaj, Z_t[i, :])
            V_t[i + 1, :] = self.V_0 + self.kappa * self.theta * (cj / gammaj * (1 - np.exp(-gammaj * self.tgrid[i + 1]))).sum() + (cj * V_tj).sum(axis=1)

        return V_t
    
    def price(self, spot, V_t, W_t, strike, cp=1):
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
            X_t[i + 1, :] = X_t[i, :] + (self.intr - 0.5 * V_t[i, :] * self.dt + np.sqrt(V_t[i, :] * self.dt) * W_t[i, :])

        S_t = np.exp(X_t)

        if isinstance(strike, (int, float)):
            return S_t, disc_fac * np.fmax(0.0, cp * (S_t[-1, :] - strike)).mean()
        elif isinstance(strike, np.ndarray):
            return S_t, disc_fac * np.fmax(0.0, cp * (S_t[-1, :] - strike[:, np.newaxis])).mean(axis=1)
        else:
            raise ValueError("Strike price must be a scalar or a numpy array")
        
    def cond_spot_sigma(self, spot, V_t, Z_t, correct_fwd=False):
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
        div_fac = np.exp(-self.texp * self.divr)
        disc_fac = np.exp(-self.texp * self.intr)
        forward = spot / disc_fac * div_fac

        V_0_T = spint.trapezoid(V_t, x=self.tgrid, axis=0)
        Y_0_T = np.sum(np.sqrt(V_t[:-1] * self.dt) * Z_t, axis=0) # cannot use the trapezoid rule because it is an Ito integral, see below for the trapezoid rule
        # Y_0_T = spint.trapezoid(np.sqrt(V_t), dx=Z_t * np.sqrt(self.dt), axis=0)

        cond_forward = forward * np.exp(self.intr * self.texp + self.rho * Y_0_T - 0.5 * self.rho ** 2 * V_0_T)
        cond_sigma = np.sqrt((1 - self.rho ** 2) * V_0_T / self.texp)

        if correct_fwd:
            forward_mc = np.mean(cond_forward)
            lambda_ = forward * np.exp(self.intr * self.texp) / forward_mc

            return lambda_ * cond_forward, cond_sigma

        else:
            return cond_forward, cond_sigma
        
    def priceCMC(self, spot, V_t, Z_t, strike, correct_fwd=False, cp=1):
        """
        Pricing of European options using conditional Monte Carlo

        Args:
            spot: spot price
            V_t: variance paths
            Z_t: the BMs driving the volatility process (needed for integration)
            strike: strike price
            correct_fwd: martingale preserving control variate

        Returns:
            price_: prices of European options
        """
        cond_forward, cond_sigma = self.cond_spot_sigma(spot, V_t, Z_t, correct_fwd=correct_fwd)
        base_model = self.base_model(vol=cond_sigma)
        if isinstance(strike, (int, float)):
            price_ = base_model.price(strike, spot=cond_forward, texp=self.texp, cp=cp)
        elif isinstance(strike, np.ndarray):
            price_ = base_model.price(strike[:, None], spot=cond_forward, texp=self.texp, cp=cp)
        else:
            raise ValueError("Strike price must be a scalar or a numpy array")

        return np.mean(price_, axis=1)
    
    def return_var_realized(self, texp, cond):
        return NotImplementedError
        