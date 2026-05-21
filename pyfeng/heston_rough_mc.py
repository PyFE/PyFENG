# -*- coding: utf-8 -*-
"""
Created on May 2, 2024

@author: Enze Zhou, Vahan Geghamyan
"""

import abc
import numpy as np
import scipy.special as spsp
import scipy.integrate as spint
from dataclasses import dataclass, field
from .sv_abc import CondMcBsmABC
from .heston_rough import RoughHestonABC


@dataclass
class RoughHestonMcABC(RoughHestonABC, CondMcBsmABC):
    """
    Abstract base class for Monte Carlo simulation of the rough Heston model.

    The rough Heston model is defined by the stochastic Volterra equations (Eqs. 1–3):
        dS_t = r S_t dt + S_t sqrt(V_t) dW_t
        V_t  = V_0 + 1/Γ(1-α) ∫_0^t (t-s)^{-α} κ(θ - V_s) ds
                   + 1/Γ(1-α) ∫_0^t (t-s)^{-α} κε sqrt(V_s) dB_s

    where W and B are correlated Brownian motions with dW dB = ρ dt, and
    α = H - 1/2 ∈ (-1/2, 0) with H the Hurst exponent (H < 1/2 for rough paths).

    Parameters are stored using the parent class convention:
        sigma  → V_0  (initial variance)
        mr     → κ    (mean-reversion speed)
        vov    → κε   (vol-of-vol, i.e. κ times epsilon)
        theta  → θ    (long-run variance)
        rho    → ρ    (correlation)
        alpha  → α    (roughness exponent; negative for rough paths)
    """

    n_ts:       int  = field(default=1000,  kw_only=True, metadata={'kind': 'numerical'})
    n_path:     int  = field(default=10000, kw_only=True, metadata={'kind': 'numerical'})
    rn_seed:    int  = field(default=None,  kw_only=True, metadata={'kind': 'numerical'})
    antithetic: bool = field(default=True,  kw_only=True, metadata={'kind': 'numerical'})

    def __post_init__(self):
        super().__post_init__()
        # Precomputed Gamma-function values used repeatedly in Eqs. (8), (17)–(20)
        self._gamma_a   = spsp.gamma(self.alpha)
        self._gamma_1ma = spsp.gamma(1 - self.alpha)   # Γ(1-α)
        self._gamma_2ma = spsp.gamma(2 - self.alpha)   # Γ(2-α) = (1-α)·Γ(1-α)
        self._gamma_1pa = spsp.gamma(1 + self.alpha)   # Γ(1+α)
        seed_seq = np.random.SeedSequence(self.rn_seed)
        self.rngs = [np.random.default_rng(s) for s in seed_seq.spawn(7)]
        self.result = {}

    @property
    def rng(self):
        return self.rngs[0]

    def configure(self, texp, n_ts=None, n_path=None, rn_seed=None, antithetic=None):
        """
        Set numerical simulation parameters and precompute frequently used constants.

        Args:
            texp: time to expiry T
            n_path: number of Monte Carlo paths (default 10 000)
            n_ts: number of time steps N; τ = T/N (default 1 000)
            rn_seed: random seed for reproducibility (default None)
            antithetic: use antithetic variates for variance reduction (default True)
        """
        if n_ts is not None:
            self.n_ts = int(n_ts)
        if n_path is not None:
            self.n_path = int(n_path)
        if rn_seed is not None:
            self.rn_seed = rn_seed
        if antithetic is not None:
            self.antithetic = antithetic

        self.__post_init__()

        self.texp = texp
        self.dt = texp / self.n_ts               # τ = T/N
        self.tgrid = np.linspace(0, texp, self.n_ts + 1)
        return self

    set_num_params = configure


@dataclass
class RoughHestonMcMaWu2022(RoughHestonMcABC):
    """
    Monte Carlo simulation of the rough Heston model using the fast algorithm of Ma & Wu (2022).

    Three variance-path simulators are provided, corresponding to Algorithms 1, 2, and 4:

    * ``ModifiedEM``       — Algorithm 1: modified Euler-Maruyama (O(N²) complexity, baseline).
    * ``Fast``             — Algorithm 2: fast simulation via sum-of-exponential kernel
                             approximation (O(N·N_exp) complexity, N_exp = O(log N)).
    * ``MultifactorApprox``— Algorithm 4: multi-factor Markovian approximation of the
                             rough Heston model (O(Ñ_exp·N) complexity).

    The kernel t^{-α} is approximated by a sum of exponentials (Eq. 12):
        t^{-α} ≈ Σ_{l=1}^{N_exp} ω_l · exp(-x_l · t)
    using a three-part Gauss quadrature scheme (Eqs. 13–15):
        · Gauss-Jacobi  on [0, 2^{-M}]   (singular near 0)
        · Gauss-Legendre on [2^j, 2^{j+1}], j = -M, …, -1  (small intervals)
        · Gauss-Legendre on [2^j, 2^{j+1}], j =  0, …,  N  (large intervals)

    Two corrections are applied to Algorithm 2 relative to the published paper:
        1. The direct stochastic term in V (third line of the update) uses the exact
           standard deviation τ^{1/2-α}/√(1-2α)/Γ(1-α); the paper omits 1/√(1-2α)
           (a typographical error — Eq. 8 is correct).
        2. The J_l stochastic increment uses the exact Itô-isometry standard deviation
           √((1-exp(-2x_l τ))/(2x_l)) rather than the paper's approximation exp(-x_l τ)·√τ.

    References:
        - Ma & Wu (2022) A fast algorithm for simulation of rough volatility models,
          Quantitative Finance, 22:3, 447–462. https://doi.org/10.1080/14697688.2021.1970213
    """

    def __init__(self, sigma, vov, rho, mr, theta, alpha, intr=0.0, divr=0.0) -> None:
        super().__init__(sigma, vov, rho, mr, theta, alpha, intr=intr, divr=divr)

    def f(self, V_s):
        """
        Drift coefficient of the rough Heston variance process: f(V_s) = κ(θ - V_s).

        This is the mean-reverting drift defined in Eq. (3) for the rough Heston
        specialisation of the general rough volatility model (Eqs. 1–2).

        Args:
            V_s: variance value V_s ≥ 0 (scalar or array)

        Returns:
            κ(θ - V_s), shape matching V_s
        """
        return self.mr * (self.theta - V_s)

    def g(self, V_s):
        """
        Diffusion coefficient of the rough Heston variance process: g(V_s) = κε√V_s.

        This is the CIR-type diffusion defined in Eq. (3) for the rough Heston
        specialisation of the general rough volatility model (Eqs. 1–2).
        Note: κε = vov is stored directly, so g(V_s) = vov · √V_s.

        Args:
            V_s: variance value V_s ≥ 0 (scalar or array)

        Returns:
            vov · √V_s (= κε√V_s), shape matching V_s
        """
        return self.vov * np.sqrt(V_s)
    
    def random_normals(self):
        """
        Generate correlated standard-normal increments for the variance and asset processes.

        The two Brownian motions B (driving variance) and W (driving asset) are correlated
        with coefficient ρ. The increments are constructed via (Eq. 26):
            Z_n  = Φ^{-1}(U^1_n)                          ~ N(0,1), drives B
            Z̃_n = ρ Z_n + √(1-ρ²) Φ^{-1}(U^2_n)         ~ N(0,1), drives W

        Returns:
            Z_t1: shape (n_ts, n_path), i.i.d. N(0,1) increments for the variance BM B
            W_t:  shape (n_ts, n_path), correlated N(0,1) increments for the asset BM W
        """
        Z_t1 = self.rngs[0].normal(size=(self.n_ts, self.n_path))
        W_t = self.rho * Z_t1 + np.sqrt(1 - self.rho**2) * self.rngs[1].normal(size=(self.n_ts, self.n_path))
        return Z_t1, W_t
    
    def term_1(self, t_n, t_k):
        """
        Kernel-increment weight for the drift sum in the modified Euler-Maruyama scheme.

        Computes (t_n - t_{k-1})^{1-α} - (t_n - t_k)^{1-α}, the weight applied to
        f(V_{t_{k-1}}) in the deterministic integral approximation (Eq. 8).  Here t_{k-1}
        = t_k - τ, so the expression simplifies to:
            (t_n - t_k + τ)^{1-α} - (t_n - t_k)^{1-α}

        Args:
            t_n: current evaluation time t_n (scalar or array)
            t_k: grid point t_k (scalar or array broadcastable with t_n)

        Returns:
            (t_n - t_k + τ)^{1-α} - (t_n - t_k)^{1-α}, same shape as inputs

        References:
            - Eq. (8) in Ma & Wu (2022): drift coefficient of the k-th term
        """
        return (t_n - t_k + self.dt) ** (1 - self.alpha) - (t_n - t_k) ** (1 - self.alpha)

    def term_2(self, t_n, t_k):
        """
        Kernel-increment weight for the diffusion sum in the modified Euler-Maruyama scheme.

        Computes the standard deviation of ∫_{t_{k-1}}^{t_k} (t_n - s)^{-α} dB_s (Eq. 7),
        which equals:
            √[ ((t_n - t_{k-1})^{1-2α} - (t_n - t_k)^{1-2α}) / (1 - 2α) ]
          = √[ ((t_n - t_k + τ)^{1-2α} - (t_n - t_k)^{1-2α}) / (1 - 2α) ]

        This weight is applied to g(V_{t_{k-1}}) · Z_k in Eq. (8).

        Args:
            t_n: current evaluation time t_n (scalar or array)
            t_k: grid point t_k (scalar or array broadcastable with t_n)

        Returns:
            √[((t_n - t_k + τ)^{1-2α} - (t_n - t_k)^{1-2α}) / (1 - 2α)], same shape as inputs

        References:
            - Eq. (7) in Ma & Wu (2022): variance of the stochastic kernel integral
            - Eq. (8) in Ma & Wu (2022): diffusion coefficient of the k-th term
        """
        return np.sqrt(((t_n - t_k + self.dt) ** (1 - 2 * self.alpha) - (t_n - t_k) ** (1 - 2 * self.alpha)) / (1 - 2 * self.alpha))
    
    def ModifiedEM(self, Z_t):
        """
        Simulate variance paths using the modified Euler-Maruyama scheme (Algorithm 1).

        Discretises the stochastic Volterra equation (Eq. 5) on the uniform grid
        t_n = nτ, τ = T/N, n = 0, …, N.  At each step (Eq. 8):

            V_{t_n} = V_0
                + 1/Γ(2-α) · Σ_{k=1}^n f(V_{t_{k-1}}) · [(t_n-t_{k-1})^{1-α} - (t_n-t_k)^{1-α}]
                + 1/Γ(1-α) · Σ_{k=1}^n g(V_{t_{k-1}}) · term_2(t_n, t_k) · Z_k

        Complexity: O(N²) multiplications per path (all past values must be retained).

        Args:
            Z_t: shape (n_ts, n_path), i.i.d. N(0,1) variance-BM increments Z_n = ΔB_n/√τ

        Returns:
            V_t: shape (n_ts+1, n_path), simulated variance paths V_{t_0}, …, V_{t_N}

        References:
            - Eq. (8) in Ma & Wu (2022): modified Euler-Maruyama discretisation
            - Algorithm 1 in Ma & Wu (2022)
        """
        assert Z_t.shape == (self.n_ts, self.n_path)

        V_t = np.zeros((self.n_ts + 1, self.n_path))
        V_t[0] = self.sigma

        for i in range(self.n_ts):
            V_t[i + 1, :] = self.sigma
            j = np.arange(1, i + 2)
            summation_f = (self.f(V_t[:i + 1, :]) * self.term_1(self.tgrid[i + 1], self.tgrid[j])[:, np.newaxis]).sum(axis=0)
            V_t[i + 1, :] += 1 / self._gamma_2ma * summation_f
            summation_g = (self.g(V_t[:i + 1, :]) * self.term_2(self.tgrid[i + 1], self.tgrid[j])[:, np.newaxis] * Z_t[:i + 1, :]).sum(axis=0)
            V_t[i + 1, :] += 1 / self._gamma_1ma * summation_g

        return V_t
    
    def get_num_nodes(self, err_tol=1e-4, scale_coef=1):
        """
        Compute the number of quadrature nodes for the sum-of-exponentials kernel approximation.

        The kernel t^{-α} is represented via the Laplace transform (Eq. 9) and approximated
        on [τ, T] by splitting the integration range into three parts (Eq. 10):
            [0, 2^{-M}]  — singular region, treated with Gauss-Jacobi
            [2^{-M}, 1]  — intermediate dyadic intervals, treated with Gauss-Legendre
            [1, ∞)       — tail region truncated at 2^{N+1}, treated with Gauss-Legendre

        Node counts follow the complexity bounds of Eq. (13):
            N_exp = O(log(1/ξ)(log log(1/ξ) + log(T/τ)) + log(1/τ) log log(1/ξ) log(1/τ))
        where ξ = err_tol and τ = dt.

        Args:
            err_tol (ξ): absolute sup-norm tolerance for the kernel approximation |t^{-α} - Σ ω_l e^{-x_l t}| ≤ ξ
            scale_coef: multiplicative scaling applied to all node counts (increase for higher accuracy)

        Returns:
            M:   lower cut-off exponent; singular integral covers [0, 2^{-M}]
            N:   upper cut-off exponent; tail integral covers [1, 2^{N+1}]
            n_o: number of Gauss-Jacobi nodes on [0, 2^{-M}]           O(log(1/ξ))
            n_s: number of Gauss-Legendre nodes per dyadic interval
                 [2^j, 2^{j+1}], j = -M, …, -1                        O(log(1/ξ))
            n_l: number of Gauss-Legendre nodes per dyadic interval
                 [2^j, 2^{j+1}], j = 0, …, N                          O(log(1/(ξτ)))

        References:
            - Eqs. (10)–(11) in Ma & Wu (2022): three-part decomposition of the kernel integral
            - Eq. (13) in Ma & Wu (2022): asymptotic node counts
        """
        M = scale_coef * np.fmax(np.ceil(np.log(self.texp)), 0) + 1
        N = scale_coef * np.ceil(np.log(np.log(1 / err_tol) / self.dt))
        n_o = scale_coef * np.ceil(np.log(1 / err_tol))
        n_s = scale_coef * np.ceil(np.log(1 / err_tol))
        n_l = scale_coef * np.ceil(np.log(1 / err_tol / self.dt))

        return M, N, n_o, n_s, n_l

    def GaussJacobiQuad(self, n, M):
        """
        Gauss-Jacobi quadrature nodes and weights for the singular interval [0, 2^{-M}].

        Approximates ∫_0^{2^{-M}} e^{-ts} s^{α-1} ds / Γ(α) ≈ Σ ω_l e^{-t s_l}
        using n-point Gauss-Jacobi quadrature with weight function (1+x)^{α-1} on [-1, 1].

        Affine map: s = 2^{-M-1}(x + 1) transforms [-1, 1] → [0, 2^{-M}].
        Jacobian factor: (2^{-M-1})^α absorbed into ω.

        Args:
            n: number of quadrature nodes (integer)
            M: cut-off exponent; interval is [0, 2^{-M}]

        Returns:
            s_o:     shape (n,), quadrature nodes in [0, 2^{-M}]
            omega_o: shape (n,), quadrature weights (already divided by Γ(α))

        References:
            - Eqs. (14)–(15) in Ma & Wu (2022): node and weight sets for the kernel approximation
        """
        s_o, w_o = spsp.roots_jacobi(n, 0, self.alpha-1)
        s_o = 2.0 ** (-M - 1) * (s_o + 1)
        w_o = 2.0 ** ((-M - 1) * self.alpha) * w_o
        omega_o = w_o / self._gamma_a

        return s_o, omega_o
    
    def GaussLegendreQuad(self, n, j):
        """
        Gauss-Legendre quadrature nodes and weights for a dyadic interval [2^j, 2^{j+1}].

        Approximates ∫_{2^j}^{2^{j+1}} e^{-ts} s^{α-1} ds / Γ(α) ≈ Σ ω_l e^{-t s_l}
        using n-point Gauss-Legendre quadrature on the standard interval [-1, 1].

        Affine map: s = 2^{j-1}(x + 3) transforms [-1, 1] → [2^j, 2^{j+1}]
        (midpoint = 3·2^{j-1}, half-width = 2^{j-1}).
        The s^{α-1} weight and the Jacobian 2^{j-1} are folded into ω.

        Used for both the intermediate intervals (j = -M, …, -1) and the large
        intervals (j = 0, …, N).  Supports broadcasting: j may be a column vector
        to compute all intervals simultaneously.

        Args:
            n: number of quadrature nodes per interval (integer or scalar)
            j: interval index or array of indices; interval is [2^j, 2^{j+1}]

        Returns:
            s_i:     quadrature nodes in [2^j, 2^{j+1}], shape (..., n)
            omega_i: quadrature weights (already divided by Γ(α)), shape (..., n)

        References:
            - Eqs. (14)–(15) in Ma & Wu (2022): node and weight sets for the kernel approximation
        """
        s_i, w_i = spsp.roots_legendre(n)
        s_i = 2.0 ** (j - 1) * (s_i + 3)
        w_i = s_i ** (self.alpha - 1) * w_i * 2.0 ** (j - 1)
        omega_i = w_i / self._gamma_a

        return s_i, omega_i
    
    def get_nodes_weights(self, err_tol, scale_coef):
        """
        Assemble all quadrature nodes and weights for the kernel approximation t^{-α} ≈ Σ ω_l e^{-x_l t}.

        Calls ``get_num_nodes`` to determine the three-part grid layout, then concatenates
        the Gauss-Jacobi nodes from [0, 2^{-M}] with the Gauss-Legendre nodes from each
        dyadic interval [2^j, 2^{j+1}], j = -M, …, N (Eqs. 10–11, 14–15).

        The total number of nodes is N_exp = n_o + M·n_s + (N+1)·n_l (Eq. 13).

        Args:
            err_tol:    absolute tolerance ξ for the kernel approximation (Eq. 12)
            scale_coef: multiplicative scaling for all node counts

        Returns:
            x_all:     shape (N_exp,), all quadrature nodes (the x_l in Eq. 12)
            omega_all: shape (N_exp,), all quadrature weights (the ω_l in Eq. 12)

        References:
            - Eq. (12) in Ma & Wu (2022): sum-of-exponentials kernel approximation
            - Eqs. (13)–(15) in Ma & Wu (2022): node counts, node sets, weight sets
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
        Advance the auxiliary drift process H_l^N by one time step (Eq. 18).

        H_l^N accumulates the weighted history of f(V) needed to approximate the
        deterministic part I_1^n of the kernel integral (Eq. 16–17):
            I_1^n ≈ τ^{1-α}/(1-α) · f(V_{t_{n-1}}) + Σ_l ω_l e^{-x_l τ} H_l^N(t_{n-1})

        Recurrence (Eq. 18):
            H_l^N(t_n) = (1 - e^{-x_l τ}) / x_l · f(V_{t_{n-1}}) + e^{-x_l τ} · H_l^N(t_{n-1})

        Initialisation: H_l^N(t_0) = 0 for all l.

        Args:
            x_all:      shape (N_exp,), quadrature nodes x_l
            H_previous: shape (n_path, N_exp), H_l^N(t_{n-1})
            V_previous: shape (n_path,),       V_{t_{n-1}}

        Returns:
            H_l^N(t_n): shape (n_path, N_exp)

        References:
            - Eq. (18) in Ma & Wu (2022): H_l recurrence
            - Eqs. (16)–(17) in Ma & Wu (2022): role of H_l in approximating I_1^n
        """
        return 1 / x_all * (1 - np.exp(-x_all * self.dt)) * self.f(V_previous)[:, np.newaxis] + np.exp(-x_all * self.dt) * H_previous
    
    def J_N(self, x_all, J_previous, V_previous, Z_t):
        """
        Advance the auxiliary stochastic process J_l^N by one time step (Eq. 20).

        J_l^N accumulates the weighted history of g(V) dB needed to approximate the
        stochastic part I_2^n of the kernel integral (Eqs. 16, 19):
            I_2^n ≈ τ^{1/2-α} · g(V_{t_{n-1}}) · Z_n + Σ_l ω_l e^{-x_l τ} J_l^N(t_{n-1})

        The paper's recurrence (Eq. 20) approximates the stochastic integral
        ∫_{t_{n-2}}^{t_{n-1}} e^{-x_l(t_{n-1}-s)} g(V) dB_s by pulling the kernel to the
        left endpoint, giving coefficient e^{-x_l τ} · √τ.  This implementation uses the
        exact Itô-isometry standard deviation instead:
            std[ ∫_{t_{n-2}}^{t_{n-1}} e^{-x_l(t_{n-1}-s)} dB_s ] = √((1 - e^{-2x_l τ}) / (2x_l))

        which reduces to e^{-x_l τ}·√τ only when x_l τ → 0.  The exact coefficient is
        more accurate for large x_l (high-frequency exponential components).

        Recurrence (corrected):
            J_l^N(t_n) = √((1 - e^{-2x_l τ})/(2x_l)) · g(V_{t_{n-1}}) · Z_n
                         + e^{-x_l τ} · J_l^N(t_{n-1})

        Initialisation: J_l^N(t_0) = 0 for all l.

        Args:
            x_all:      shape (N_exp,), quadrature nodes x_l
            J_previous: shape (n_path, N_exp), J_l^N(t_{n-1})
            V_previous: shape (n_path,),       V_{t_{n-1}}
            Z_t:        shape (n_path,),       standard-normal increment Z_n = ΔB_n/√τ

        Returns:
            J_l^N(t_n): shape (n_path, N_exp)

        References:
            - Eq. (20) in Ma & Wu (2022): J_l recurrence (paper uses approximation e^{-x_l τ}·√τ)
            - Eqs. (16), (19) in Ma & Wu (2022): role of J_l in approximating I_2^n
        """
        return np.sqrt(((1 - np.exp(-2 * x_all * self.dt)) / (2 * x_all))) * (self.g(V_previous) * Z_t)[:, np.newaxis] + np.exp(-x_all * self.dt) * J_previous
    
    def Fast(self, Z_t, err_tol=1e-4, scale_coef=1):
        """
        Simulate variance paths using the fast algorithm (Algorithm 2).

        The kernel t^{-α} is replaced by its sum-of-exponentials approximation (Eq. 12),
        reducing the O(N²) history convolution to O(N·N_exp) recurrences on the auxiliary
        processes H_l^N and J_l^N.  The variance update at each step is (Algorithm 2):

            V_{t_n} = V_0
                + τ^{1-α}/Γ(2-α)              · f(V_{t_{n-1}})                  [drift, current step]
                + 1/Γ(1-α) · Σ_l ω_l e^{-x_l τ} H_l^N(t_{n-1})                [drift, history]
                + τ^{1/2-α}/√(1-2α)/Γ(1-α)   · g(V_{t_{n-1}}) · Z_n            [diffusion, current step]
                + 1/Γ(1-α) · Σ_l ω_l e^{-x_l τ} J_l^N(t_{n-1})                [diffusion, history]

        Note: the paper's Algorithm 2 omits 1/√(1-2α) in the third term.  This is a typo:
        Eq. (8) carries the factor correctly, and it follows from the variance of the
        stochastic kernel integral (Eq. 7).

        After the V update, H_t and J_t are advanced via ``H_N`` and ``J_N`` using V_{t_n}
        and Z_n, so they are ready as H_l^N(t_n) and J_l^N(t_n) for the next step.

        Args:
            Z_t:        shape (n_ts, n_path), i.i.d. N(0,1) variance-BM increments Z_n
            err_tol:    absolute tolerance ξ for the kernel approximation (default 1e-4)
            scale_coef: node-count scaling factor (default 1; increase for higher accuracy)

        Returns:
            V_t: shape (n_ts+1, n_path), simulated variance paths V_{t_0}, …, V_{t_N}

        References:
            - Algorithm 2 in Ma & Wu (2022): fast variance-path simulation
            - Eq. (7) in Ma & Wu (2022): variance of the stochastic kernel integral
              (justifies the 1/√(1-2α) correction to the third term)
        """
        x_all, omega_all = self.get_nodes_weights(err_tol=err_tol, scale_coef=scale_coef)

        V_t = np.zeros((self.n_ts + 1, self.n_path))
        V_t[0, :] = self.sigma
        N_all = len(x_all)

        H_t = np.zeros((self.n_path, N_all))
        J_t = np.zeros((self.n_path, N_all))

        for i in range(self.n_ts):
            V_t[i + 1, :] = self.sigma + self.dt ** (1 - self.alpha) / self._gamma_2ma * self.f(V_t[i, :]) \
                            + 1 / self._gamma_1ma * (omega_all * np.exp(-x_all * self.dt) * H_t).sum(axis=1) \
                            + self.dt ** (0.5 - self.alpha) / np.sqrt(1 - 2 * self.alpha) / self._gamma_1ma * self.g(V_t[i, :]) * Z_t[i, :] \
                            + 1 / self._gamma_1ma * (omega_all * np.exp(-x_all * self.dt) * J_t).sum(axis=1)
            H_t = self.H_N(x_all, H_t, V_t[i, :])
            J_t = self.J_N(x_all, J_t, V_t[i, :], Z_t[i, :])

        return V_t
    
    def eta_j(self, N_exp):
        """
        Grid points η_j for the multi-factor kernel quadrature (Eq. 27).

        The measure μ(dγ) = γ^{α-1}/[Γ(1-α)Γ(α)] dγ is discretised on [η_0, η_{Ñexp}]
        using the uniform grid:
            η_j = j · Ñ_exp^{-1/5} / T · (√10 · α / (2+α))^{2/5},  j = 0, …, Ñ_exp

        Args:
            N_exp (Ñ_exp): number of quadrature intervals / factors

        Returns:
            eta: shape (N_exp+1,), grid points η_0, …, η_{Ñexp}

        References:
            - Eq. (27) in Ma & Wu (2022): definition of η_j
        """
        j = np.arange(N_exp + 1)
        return j * N_exp ** (-1 / 5) / self.texp * (np.sqrt(10) * self.alpha / (2 + self.alpha)) ** (2 / 5)

    def c_j_gamma_j(self, N_exp):
        """
        Multi-factor coefficients c_j and mean-reversion rates γ_j (Eq. 27).

        Each factor V^j in the multi-factor approximation (Algorithm 4) is an
        Ornstein-Uhlenbeck process with rate γ_j and loading c_j.  The c_j and γ_j
        are moments of the measure μ on each interval [η_{j-1}, η_j]:
            c_j = ∫_{η_{j-1}}^{η_j} μ(dγ) = (η_j^α - η_{j-1}^α) / [Γ(1-α) Γ(1+α)]
            γ_j = α/(1+α) · (η_j^{α+1} - η_{j-1}^{α+1}) / (η_j^α - η_{j-1}^α)

        Args:
            N_exp (Ñ_exp): number of factors

        Returns:
            cj:     shape (N_exp,), factor loadings c_j > 0
            gammaj: shape (N_exp,), mean-reversion rates γ_j > 0

        References:
            - Eq. (27) in Ma & Wu (2022): definitions of c_j and γ_j
            - Eq. (26) in Ma & Wu (2022): measure μ(dγ)
        """
        eta = self.eta_j(N_exp)
        cj = (eta[1:] ** self.alpha - eta[:-1] ** self.alpha) / self._gamma_1ma / self._gamma_1pa
        gammaj = (eta[1:] ** (self.alpha + 1) - eta[:-1] ** (self.alpha + 1)) / (eta[1:] ** self.alpha - eta[:-1] ** self.alpha) * self.alpha / (1 + self.alpha)

        return cj, gammaj
    
    def V_tJ(self, V_tj_previous, V_previous, gammaj, Z_t):
        """
        Advance the j-th factor V^j by one time step (Algorithm 4).

        Each factor satisfies the SDE (Eq. 31):
            dV_t^j = (-γ_j V_t^j - κ V_t^{Ñ,N}) dt + g(V_t^{Ñ,N}) dB_t

        Algorithm 4 discretises this with the semi-implicit Euler scheme:
            V_{t_n}^j = (V_{t_{n-1}}^j - κ V_{t_{n-1}}^{Ñ,N} τ + g(V_{t_{n-1}}^{Ñ,N}) √τ Z_n)
                        / (1 + γ_j τ)

        This implementation uses the explicit Euler scheme instead:
            V_{t_n}^j = (1 - γ_j τ) V_{t_{n-1}}^j - κ V_{t_{n-1}}^{Ñ,N} τ + g(V_{t_{n-1}}^{Ñ,N}) √τ Z_n

        The two schemes agree to first order in τ.  The semi-implicit form in Algorithm 4
        is more stable when γ_j τ is not small (large mean-reversion rate or coarse grid).

        Args:
            V_tj_previous: shape (n_path, N_exp), factor values V_{t_{n-1}}^j
            V_previous:    shape (n_path,),       aggregate variance V_{t_{n-1}}^{Ñ,N}
            gammaj:        shape (N_exp,),         mean-reversion rates γ_j
            Z_t:           shape (n_path,),        standard-normal increment Z_n

        Returns:
            V_{t_n}^j: shape (n_path, N_exp)

        References:
            - Algorithm 4 in Ma & Wu (2022): multi-factor approximation simulation
            - Eq. (31) in Ma & Wu (2022): factor SDE
        """
        return (1 - gammaj * self.dt) * V_tj_previous + (-self.mr * V_previous * self.dt + self.g(V_previous) * np.sqrt(self.dt) * Z_t)[:, np.newaxis]
    
    def MultifactorApprox(self, Z_t):
        """
        Simulate variance paths using the multi-factor Markovian approximation (Algorithm 4).

        The rough Heston model (Eq. 29) is approximated by the Ñ_exp-factor Heston model
        (Eqs. 30–31) of Abi Jaber & El Euch (2019), where the fractional kernel is replaced
        by a sum of exponentials (Eq. 26):
            t^{-α}/Γ(1-α) ≈ Σ_{j=1}^{Ñexp} c_j e^{-γ_j t}

        The aggregate variance and its factors are updated together (Algorithm 4):
            V_{t_n}^{Ñ,N} = V_0 + κθ Σ_j c_j/γ_j (1 - e^{-γ_j t_n}) + Σ_j c_j V_{t_n}^j

        where each factor V^j evolves via ``V_tJ``.  The number of factors is set to
        Ñ_exp = ceil(N^{5/4}) to balance kernel approximation and discretisation errors.

        Complexity: O(Ñ_exp · N) per path, with Ñ_exp = O(N^{1/2}) in practice (Remark 3.1).

        Args:
            Z_t: shape (n_ts, n_path), i.i.d. N(0,1) variance-BM increments Z_n

        Returns:
            V_t: shape (n_ts+1, n_path), simulated variance paths V_{t_0}, …, V_{t_N}

        References:
            - Algorithm 4 in Ma & Wu (2022): multi-factor approximation simulation
            - Eq. (30) in Ma & Wu (2022): multi-factor variance process
            - Eq. (31) in Ma & Wu (2022): factor SDE
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
    
    def price(self, spot, V_t, W_t, strike, cp=1):
        """
        Simulate log-price paths and price European options by direct Monte Carlo.

        The log-price X_t = log(S_t) follows (Eq. 24):
            dX_t = (r - V_t/2) dt + √V_t dW_t

        Discretised by the Euler-Maruyama scheme (Eq. 25):
            X_{t_n} = X_{t_{n-1}} + (r - V_{t_{n-1}}/2) τ + √(V_{t_{n-1}} τ) · Z̃_n

        The option price is the discounted expected payoff averaged over paths.

        Note: known bugs in the current implementation —
            (1) the drift uses ``self.intr`` instead of ``self.intr * self.dt``, overstating
                the interest-rate contribution by a factor of n_ts;
            (2) the loop runs ``range(n_ts - 1)``, so X_t is advanced only n_ts-1 steps and
                the terminal payoff is evaluated one step before T.

        Args:
            spot:   initial asset price S_0
            V_t:    shape (n_ts+1, n_path), variance paths from ``ModifiedEM``, ``Fast``,
                    or ``MultifactorApprox``
            W_t:    shape (n_ts, n_path), correlated N(0,1) asset-BM increments Z̃_n
                    (second return value of ``random_normals``)
            strike: strike price K (scalar or 1-D array for a strip of strikes)
            cp:     +1 for call, -1 for put (default +1)

        Returns:
            S_t:   shape (n_ts, n_path), simulated asset price paths
            price: discounted expected payoff; scalar if ``strike`` is scalar,
                   shape (len(strike),) if ``strike`` is an array

        References:
            - Eq. (24) in Ma & Wu (2022): log-price SDE
            - Eq. (25) in Ma & Wu (2022): Euler-Maruyama discretisation of log-price
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
        Compute the conditional forward and volatility for conditional Monte Carlo (CMC) pricing.

        Given a simulated variance path V and its driving BM path B, the log-price decomposes as:
            X_T = log(S_0) + (r-q)T - V_0_T/2 + ρ Y_0_T + √(1-ρ²) · ε

        where V_0_T = ∫_0^T V_t dt, Y_0_T = ∫_0^T √V_t dB_t, and ε ~ N(0, V_0_T) is
        independent of the variance path.  Conditioning on (V, B) gives:

            E[S_T | V, B] = forward · exp(ρ Y_0_T - ρ²/2 · V_0_T)
            Var[X_T | V, B] = (1-ρ²) V_0_T

        so conditional on the path, S_T has a log-normal distribution with:
            cond_forward = forward · exp(ρ Y_0_T - ρ²/2 · V_0_T)
            cond_sigma   = √((1-ρ²) V_0_T / T)      [annualised vol]

        The option price is then E[ BSM(K, cond_forward, cond_sigma) ] over paths, which
        reduces variance compared to the direct payoff average.

        Y_0_T is approximated by the left-endpoint Euler sum (Itô integral, not trapezoidal):
            Y_0_T ≈ Σ_{i=0}^{N-1} √(V_{t_i} τ) · Z_i

        Note: known bug — ``cond_forward`` currently multiplies by an extra factor
        ``exp(r·T)``, which double-counts the risk-free drift already in ``forward``.
        The correct formula is ``forward * exp(ρ Y_0_T - ρ²/2 V_0_T)``.

        Args:
            spot:        initial asset price S_0
            V_t:         shape (n_ts+1, n_path), simulated variance paths
            Z_t:         shape (n_ts, n_path), N(0,1) variance-BM increments Z_n (needed
                         to evaluate Y_0_T = ∫√V dB as an Itô integral)
            correct_fwd: if True, apply a path-wise martingale-correction (control variate)
                         that rescales cond_forward so its mean equals the true forward

        Returns:
            cond_forward: shape (n_path,), E[S_T | V, B]  (path-conditional forward)
            cond_sigma:   shape (n_path,), √((1-ρ²) V_0_T / T)  (path-conditional vol)
        """
        div_fac = np.exp(-self.texp * self.divr)
        disc_fac = np.exp(-self.texp * self.intr)
        forward = spot / disc_fac * div_fac

        V_0_T = spint.trapezoid(V_t, x=self.tgrid, axis=0)
        # Y_0_T = ∫_0^T √V_t dB_t approximated as an Itô (left-endpoint) sum;
        # the trapezoidal rule is incorrect here because the integrand is not smooth in B.
        Y_0_T = np.sum(np.sqrt(V_t[:-1] * self.dt) * Z_t, axis=0)

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
        Price European options using conditional Monte Carlo (CMC).

        For each simulated variance path, the option price under the path-conditional
        log-normal distribution is computed analytically via BSM.  The final price is
        the average of these path-conditional BSM prices.  CMC substantially reduces
        variance versus the direct payoff average because the residual randomness
        (the independent Brownian motion W) is integrated out analytically.

        Workflow:
            1. ``cond_spot_sigma`` → cond_forward, cond_sigma for each path
            2. ``base_model(vol=cond_sigma).price(strike, spot=cond_forward, texp)``
               evaluates BSM for each path
            3. Average over paths

        Args:
            spot:        initial asset price S_0
            V_t:         shape (n_ts+1, n_path), simulated variance paths
            Z_t:         shape (n_ts, n_path), N(0,1) variance-BM increments (for Y_0_T)
            strike:      strike price K; scalar or 1-D array for a strip of strikes
            correct_fwd: if True, apply the martingale-correction control variate
                         in ``cond_spot_sigma`` (default False)
            cp:          +1 for call, -1 for put (default +1)

        Returns:
            price_: discounted CMC option price; scalar if ``strike`` is scalar,
                    shape (len(strike),) if ``strike`` is an array
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
        raise NotImplementedError
        
