"""
European option pricing via the Fourier-Cosine (COS) method.

References:
    Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions.
    SIAM Journal on Scientific Computing 31(2):826-848.
    https://doi.org/10.1137/080718061
"""

import abc
import numpy as np
from . import opt_abc as opt
from .sv_fft import BsmFft, VarGammaFft, HestonFft


class CosABC(opt.OptABC, abc.ABC):
    """
    Abstract base for European option pricing via the Fourier-Cosine (COS)
    method of Fang & Oosterlee (2008).

    Mirrors FftABC: subclasses implement mgf_logprice(uu, texp) and inherit
    price() from here instead of from FftABC.

    Attributes:
        n_cos: Number of cosine terms N. Default 128.
        L: Truncation half-width multiplier. Default 12.
    """

    n_cos: int = 128
    L: float = 12.0

    def charfunc_logprice(self, u, texp):
        """Characteristic function phi(u) = MGF(i*u)."""
        return self.mgf_logprice(1j * u, texp)

    def _cumulants(self, texp):
        """
        Cumulants (c1, c2, 0, c4) of log(S_T/F) via numerical differentiation
        of log MGF. Subclasses override with analytic formulas where available.
        """
        eps = 1e-3
        lm = lambda v: float(np.log(self.mgf_logprice(v, texp)).real)
        lm0 = lm(0.0)
        lmp1, lmm1 = lm(eps), lm(-eps)
        lmp2, lmm2 = lm(2 * eps), lm(-2 * eps)
        c1 = (lmp1 - lmm1) / (2 * eps)
        c2 = (lmp1 + lmm1 - 2 * lm0) / eps**2
        c4 = (lmp2 - 4 * lmp1 + 6 * lm0 - 4 * lmm1 + lmm2) / eps**4
        return c1, c2, 0.0, c4

    def _truncation_range(self, texp):
        """Integration interval [a, b] from F&O Eq. (5.2)."""
        c1, c2, _, c4 = self._cumulants(texp)
        half = self.L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
        return c1 - half, c1 + half

    @staticmethod
    def _chi(k, u, a, c, d):
        """F&O Eq. (22): integral of exp(x)*cos(k*pi*(x-a)/(b-a)) from c to d."""
        exp_d, exp_c = np.exp(d), np.exp(c)
        cos_d = np.cos(u * (d - a))
        cos_c = np.cos(u * (c - a))
        sin_d = np.sin(u * (d - a))
        sin_c = np.sin(u * (c - a))
        return (cos_d * exp_d - cos_c * exp_c + u * (sin_d * exp_d - sin_c * exp_c)) / (1.0 + u**2)

    @staticmethod
    def _psi(k, u, a, c, d):
        """F&O Eq. (23): integral of cos(k*pi*(x-a)/(b-a)) from c to d."""
        safe_u = np.where(k == 0, 1.0, u)
        return np.where(k == 0, d - c, (np.sin(u * (d - a)) - np.sin(u * (c - a))) / safe_u)

    def price(self, strike, spot, texp, cp=1):
        """
        European call/put price via the COS method.

        Vectorised over strike and cp. Single truncation range [a, b] shared
        across all strikes (appropriate for BSM and VG; HestonCos overrides
        this with per-strike truncation).

        Args:
            strike: strike price(s) — scalar or array.
            spot:   spot (or forward when is_fwd=True) price.
            texp:   time to expiry.
            cp:     +1 call / -1 put (scalar or array).

        Returns:
            Option price(s) matching the broadcast shape of (strike, cp).
        """
        fwd, df, _ = self._fwd_factor(spot, texp)

        scalar_out = np.isscalar(strike) and np.isscalar(cp)
        kk = np.atleast_1d(np.asarray(strike / fwd, dtype=float))
        cp_a = np.broadcast_to(np.atleast_1d(np.asarray(cp, dtype=float)), kk.shape).copy()

        a, b = self._truncation_range(texp)
        ba = b - a

        k_arr = np.arange(self.n_cos)
        u_arr = k_arr * np.pi / ba

        cf = self.charfunc_logprice(u_arr, texp)
        cf_s = cf * np.exp(-1j * u_arr * a)
        cf_s[0] *= 0.5
        cf_re = cf_s.real

        log_kk = np.clip(np.log(kk), a, b)[:, None]
        u = u_arr[None, :]
        k = k_arr[None, :]
        kk_c = kk[:, None]

        W_call = (2.0 / ba) * (self._chi(k, u, a, log_kk, b) - kk_c * self._psi(k, u, a, log_kk, b))
        W_put = (2.0 / ba) * (kk_c * self._psi(k, u, a, a, log_kk) - self._chi(k, u, a, a, log_kk))

        W = np.where(cp_a[:, None] > 0, W_call, W_put)
        price_arr = df * fwd * (W @ cf_re)

        if scalar_out:
            return float(price_arr[0])
        return price_arr.reshape(np.broadcast_shapes(np.shape(strike), np.shape(cp)))


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes-Merton
# ─────────────────────────────────────────────────────────────────────────────

class BsmCos(CosABC, BsmFft):
    """
    Black-Scholes-Merton option pricing via the COS method.

    Inherits mgf_logprice from BsmFft; overrides _cumulants with exact
    analytic values (BSM is Gaussian so c4 = 0).

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmCos(sigma=0.2, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([15.71362...,  9.69251...,  5.52949...,  2.94558...,  1.48139...])
    """

    def _cumulants(self, texp):
        s2t = self.sigma**2 * texp
        return -0.5 * s2t, s2t, 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Variance Gamma
# ─────────────────────────────────────────────────────────────────────────────

class VarGammaCos(CosABC, VarGammaFft):
    """
    Variance Gamma (VG) option pricing via the COS method.

    Inherits mgf_logprice from VarGammaFft; overrides _cumulants with analytic
    VG cumulants from F&O (2008) Table 11.

    Parameter convention (matches VarGammaFft):
        sigma = σ (BM volatility)
        vov   = ν (variance rate of Gamma time-change)
        theta = θ (BM drift, controls skewness)

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.VarGammaCos(sigma=0.12, vov=0.174, theta=-0.14)
        >>> m.price(np.arange(80, 121, 10), 100, 1.0)
    """

    def _cumulants(self, texp):
        nu = self.vov
        sig2 = self.sigma**2
        w = np.log(1.0 - self.theta * nu - 0.5 * sig2 * nu) / nu
        c1 = texp * (w + self.theta)
        c2 = (sig2 + nu * self.theta**2) * texp
        c4 = 3.0 * (sig2**2 * nu + 2.0 * self.theta**4 * nu**3 + 4.0 * sig2 * self.theta**2 * nu**2) * texp
        return float(c1), float(abs(c2)), 0.0, float(abs(c4))


# ─────────────────────────────────────────────────────────────────────────────
# Heston stochastic-volatility model
# ─────────────────────────────────────────────────────────────────────────────

class HestonCos(CosABC, HestonFft):
    """
    Heston (1993) stochastic-volatility option pricing via the COS method
    of Fang & Oosterlee (2008).

    Inherits mgf_logprice from HestonFft (Lord-Kahl 2010 branch-cut-safe form).
    Overrides price() with per-strike truncation intervals centered on the
    conditional mean of log(S_T/K), using the F&O Section 5.2 half-width
    heuristic L * sqrt(theta + sigma*vov).

    References:
        - Heston SL (1993) Rev. Financial Studies 6:327-343.
        - Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31:826-848.
        - Lord R, Kahl C (2010) Mathematical Finance 20:671-694.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.array([60, 70, 100, 140])
        >>> m = pf.HestonCos(sigma=0.04, vov=1, mr=0.5, rho=-0.9)
        >>> m.price(strike, 100, 10)
        array([44.329..., 35.849..., 13.084...,  0.295...])
    """

    @staticmethod
    def _default_L(texp):
        """Adaptive L = max(10, 3T+2). Matches heston_cos_pricer._default_L."""
        return max(10.0, 3.0 * texp + 2.0)

    def _sigma_h(self):
        """F&O Section 5.2 heuristic: sqrt(theta + sigma*vov)."""
        return np.sqrt(abs(self.theta) + abs(self.sigma) * abs(self.vov))

    def _c1(self, texp):
        """Mean of log(S_T/F) = -0.5 * E[integrated variance]."""
        return -0.5 * texp * self.avgvar_mv(texp)[0]

    def _cumulants(self, texp):
        """Analytic Heston cumulants of log(S_T/F). F&O Appendix A.2."""
        kap = self.mr
        eta = self.vov
        ubar = self.theta
        v0 = self.sigma
        T = texp
        c1 = self._c1(texp)
        eT = np.exp(-kap * T)
        e2T = eT * eT
        c2 = (1.0 / (8.0 * kap**3)) * (
            eta * T * kap * eT * (v0 - ubar) * (8.0 * kap * self.rho - 4.0 * eta)
            + kap * self.rho * eta * (1.0 - eT) * (16.0 * ubar - 8.0 * v0)
            + 2.0 * ubar * kap * T * (-4.0 * kap * self.rho * eta + eta**2 + 4.0 * kap**2)
            + eta**2 * ((ubar - 2.0 * v0) * e2T + ubar * (6.0 * eT - 7.0) + 2.0 * v0)
            + 8.0 * kap**2 * (v0 - ubar) * (1.0 - eT)
        )
        return float(c1), float(abs(c2)), 0.0, 0.0

    @staticmethod
    def _payoff_coefficients(N, a_y, b_y, cp):
        """
        Payoff coefficient matrix of shape (M, N).

        Integration threshold at y = 0 (S_T = K) in y = log(S_T/K) space.
        Ported from heston_cos_pricer.payoff_coefficients.

        Args:
            N:   number of COS terms.
            a_y: lower truncation bound, shape (M, 1).
            b_y: upper truncation bound, shape (M, 1).
            cp:  +1 call, -1 put (scalar).
        """
        k = np.arange(N)[None, :]
        ba = b_y - a_y
        u = k * np.pi / ba
        inv_1_u2 = 1.0 / (1.0 + u**2)
        u_safe = np.where(k == 0, 1.0, u)

        if cp > 0:
            arg_b = u * (b_y - a_y)   # = k*pi
            arg_0 = u * (0.0 - a_y)
            exp_b = np.exp(b_y)
            chi = (np.cos(arg_b) * exp_b - np.cos(arg_0) + u * (np.sin(arg_b) * exp_b - np.sin(arg_0))) * inv_1_u2
            psi = np.where(k == 0, b_y, (np.sin(arg_b) - np.sin(arg_0)) / u_safe)
            return (2.0 / ba) * (chi - psi)

        arg_0 = u * (0.0 - a_y)
        exp_a = np.exp(a_y)
        chi = (np.cos(arg_0) - exp_a + u * np.sin(arg_0)) * inv_1_u2
        psi = np.where(k == 0, -a_y, np.sin(arg_0) / u_safe)
        return (2.0 / ba) * (-chi + psi)

    def price(self, strike, spot, texp, cp=1):
        """
        European call/put price via COS with per-strike truncation intervals.

        Each strike gets its own [a_y, b_y] centered at log(F/K) + c1, the
        conditional mean of log(S_T/K). The CF phase factor is precomputed
        once and is strike-independent.
        """
        fwd, df, _ = self._fwd_factor(spot, texp)

        scalar_out = np.isscalar(strike) and np.isscalar(cp)
        kk = np.atleast_1d(np.asarray(strike / fwd, dtype=float))
        cp_a = np.broadcast_to(np.atleast_1d(np.asarray(cp, dtype=float)), kk.shape).copy()

        N = self.n_cos
        L = self._default_L(texp)
        half = L * self._sigma_h()
        width = 2.0 * half
        c1 = self._c1(texp)

        k_arr = np.arange(N)
        u_arr = k_arr * np.pi / width

        cf = self.charfunc_logprice(u_arr, texp)
        cf_s = cf * np.exp(1j * u_arr * (half - c1))
        cf_s[0] *= 0.5
        phi_re = cf_s.real

        log_kk = np.log(kk)
        a_y = (-log_kk + c1 - half)[:, None]
        b_y = (-log_kk + c1 + half)[:, None]

        if np.all(cp_a > 0):
            W = self._payoff_coefficients(N, a_y, b_y, cp=1)
        elif np.all(cp_a < 0):
            W = self._payoff_coefficients(N, a_y, b_y, cp=-1)
        else:
            W = np.where(
                cp_a[:, None] > 0,
                self._payoff_coefficients(N, a_y, b_y, cp=1),
                self._payoff_coefficients(N, a_y, b_y, cp=-1),
            )

        price_arr = df * fwd * kk * (W @ phi_re)

        if scalar_out:
            return float(price_arr[0])
        return price_arr.reshape(np.broadcast_shapes(np.shape(strike), np.shape(cp)))
