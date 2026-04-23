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
from . import sv_abc as sv
from . import heston


class CosABC(opt.OptABC, abc.ABC):
    """
    Abstract base class for European option pricing via the Fourier-Cosine
    (COS) method of Fang & Oosterlee (2008).

    Subclasses must implement ``mgf_logprice(uu, texp)`` – the moment
    generating function of log(S_T / F) where F is the forward price.
    The same interface is used by ``FftABC`` so model-level MGF
    implementations (e.g. ``HestonFft``) are directly reusable.

    Attributes:
        n_cos (int): Number of Fourier-cosine terms N (default 128).
                     Increase for higher accuracy or extreme parameters.
        L (float): Truncation-range half-width multiplier (default 12).
    """

    n_cos: int = 128
    L: float = 12.0

    @abc.abstractmethod
    def mgf_logprice(self, uu, texp):
        """
        Moment generating function (MGF) of log(S_T / F).

        Args:
            uu: argument – scalar or array, real or complex.
            texp: time to expiry.

        Returns:
            MGF values with the same shape as *uu*.
        """
        raise NotImplementedError

    def charfunc_logprice(self, u, texp):
        """Characteristic function phi(u) = MGF(i*u)."""
        return self.mgf_logprice(1j * u, texp)

    # ------------------------------------------------------------------
    # Cumulants and truncation range
    # ------------------------------------------------------------------

    def _cumulants(self, texp):
        """
        First four cumulants of log(S_T/F) via numerical differentiation
        of log MGF at real arguments.  Subclasses override with analytic
        formulas where available.

        Returns:
            (c1, c2, c3, c4)
        """
        eps = 1e-3
        lm = lambda v: float(np.log(self.mgf_logprice(v, texp)).real)
        lm0 = lm(0.0)
        lmp1, lmm1 = lm(eps), lm(-eps)
        lmp2, lmm2 = lm(2*eps), lm(-2*eps)
        c1 = (lmp1 - lmm1) / (2*eps)
        c2 = (lmp1 + lmm1 - 2*lm0) / eps**2
        c4 = (lmp2 - 4*lmp1 + 6*lm0 - 4*lmm1 + lmm2) / eps**4
        return c1, c2, 0.0, c4

    def _truncation_range(self, texp):
        """
        Integration interval [a, b] from Eq. (5.2) of Fang & Oosterlee.

        Returns:
            (a, b) floats
        """
        c1, c2, _, c4 = self._cumulants(texp)
        half = self.L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
        return c1 - half, c1 + half

    # ------------------------------------------------------------------
    # Payoff coefficient helpers (Eqs. 22-23)
    # ------------------------------------------------------------------

    @staticmethod
    def _chi(k, u, a, c, d):
        """
        Eq. (22): integral from c to d of exp(x) * cos(k*pi*(x-a)/(b-a)) dx

        Broadcasting convention: u has shape (1, N), c and d have
        shape (M, 1) so the result has shape (M, N).
        """
        exp_d, exp_c = np.exp(d), np.exp(c)
        cos_d = np.cos(u * (d - a))
        cos_c = np.cos(u * (c - a))
        sin_d = np.sin(u * (d - a))
        sin_c = np.sin(u * (c - a))
        num = cos_d*exp_d - cos_c*exp_c + u*(sin_d*exp_d - sin_c*exp_c)
        return num / (1.0 + u**2)

    @staticmethod
    def _psi(k, u, a, c, d):
        """
        Eq. (23): integral from c to d of cos(k*pi*(x-a)/(b-a)) dx.
        k=0 handled separately to avoid division by zero.
        """
        safe_u = np.where(k == 0, 1.0, u)
        return np.where(
            k == 0,
            d - c,
            (np.sin(u * (d - a)) - np.sin(u * (c - a))) / safe_u
        )

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, strike, spot, texp, cp=1):
        """
        European call/put price via the COS method.

        Fully vectorised over *strike* and *cp*. The dominant cost is
        one (M x N) matrix-vector multiply, where M = len(strikes)
        and N = self.n_cos.

        Args:
            strike: strike price(s) - scalar or array shape (M,).
            spot:   spot (or forward if ``is_fwd=True``) price.
            texp:   time to expiry.
            cp:     +1 call / -1 put (scalar or array matching strike).

        Returns:
            Option price(s) matching the broadcast shape of (strike, cp).
        """
        fwd, df, _ = self._fwd_factor(spot, texp)

        scalar_out = np.isscalar(strike) and np.isscalar(cp)
        kk   = np.atleast_1d(np.asarray(strike / fwd, dtype=float))   # (M,)
        cp_a = np.broadcast_to(
            np.atleast_1d(np.asarray(cp, dtype=float)), kk.shape
        ).copy()

        a, b = self._truncation_range(texp)
        ba   = b - a

        k_arr = np.arange(self.n_cos)          # (N,)
        u_arr = k_arr * np.pi / ba             # (N,)

        # Characteristic function with phase shift exp(-i*u*a)
        cf   = self.charfunc_logprice(u_arr, texp)   # (N,) complex
        cf_s = cf * np.exp(-1j * u_arr * a)
        cf_s[0] *= 0.5                               # prime-sum (k=0 gets 1/2)
        cf_re = cf_s.real                            # (N,)

        # Payoff coefficients - shape (M, N)
        log_kk = np.clip(np.log(kk), a, b)[:, None] # (M, 1)
        u  = u_arr[None, :]                          # (1, N)
        k  = k_arr[None, :]                          # (1, N)
        kk_c = kk[:, None]                           # (M, 1)

        # Call:  (2/ba) * [ chi(log_kk, b) - K/F * psi(log_kk, b) ]
        W_call = (2.0 / ba) * (
            self._chi(k, u, a, log_kk, b) - kk_c * self._psi(k, u, a, log_kk, b)
        )
        # Put:   (2/ba) * [ K/F * psi(a, log_kk) - chi(a, log_kk) ]
        W_put = (2.0 / ba) * (
            kk_c * self._psi(k, u, a, a, log_kk) - self._chi(k, u, a, a, log_kk)
        )

        W = np.where(cp_a[:, None] > 0, W_call, W_put)   # (M, N)

        price_arr = df * fwd * (W @ cf_re)                 # (M,)

        if scalar_out:
            return float(price_arr[0])
        return price_arr.reshape(
            np.broadcast_shapes(np.shape(strike), np.shape(cp))
        )


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes-Merton (included for cross-validation; not exported)
# ─────────────────────────────────────────────────────────────────────────────

class BsmCos(CosABC):
    """
    Black-Scholes-Merton European option pricing via the COS method.

    Uses analytic BSM cumulants (c4 = 0), giving a tight truncation range
    and near machine-precision accuracy for N >= 64.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmCos(sigma=0.2, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([15.71361973,  9.69250803,  5.52948546,  2.94558338,  1.48139131])
    """

    def mgf_logprice(self, uu, texp):
        """BSM log-price MGF: exp(-0.5 * sigma^2 * T * u * (1 - u))."""
        return np.exp(-0.5 * self.sigma**2 * texp * uu * (1.0 - uu))

    def _cumulants(self, texp):
        """Exact BSM cumulants. c4 = 0 -> minimal truncation range."""
        s2t = self.sigma**2 * texp
        return -0.5 * s2t, s2t, 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Heston stochastic-volatility model
# ─────────────────────────────────────────────────────────────────────────────

class HestonCos(heston.HestonABC, CosABC):
    """
    Heston (1993) stochastic-volatility model: European option pricing
    via the COS method of Fang & Oosterlee (2008).

    Parameters (PyFENG ``SvABC`` convention):
        sigma  - initial variance V0
        vov    - vol-of-vol (eta)
        mr     - mean-reversion speed (kappa)
        rho    - correlation (rho)
        theta  - long-run variance V-bar (defaults to sigma)

    The CF uses the Lord-Kahl (2010) branch-cut-safe formulation,
    identical to ``HestonFft``, so both pricers can be cross-validated.
    Analytic cumulants from F&O (2008) Appendix A set the truncation
    range. For parameters violating the Feller condition (2*kappa*V-bar
    < eta^2) or maturities T > 5, increase ``n_cos`` (e.g. ``m.n_cos = 512``).

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> sigma, vov, mr, rho, texp, spot = 0.04, 0.5, 1.5, -0.7, 1.0, 100
        >>> m = pf.HestonCos(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.price(np.array([90, 95, 100, 105, 110]), spot, texp)

    References:
        - Heston SL (1993) Rev. Financial Studies 6:327-343.
        - Lord R, Kahl C (2010) Mathematical Finance 20:671-694.
        - Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31:826-848.
    """

    def mgf_logprice(self, uu, texp):
        """
        Heston log-price MGF – Lord & Kahl (2010) branch-cut-safe formulation.
        Matches HestonFft.mgf_logprice exactly (same variable names and style).

        References:
            - Lord R, Kahl C (2010) Complex Logarithms in Heston-Like Models.
              Mathematical Finance 20:671-694.
        """
        var_0 = self.sigma
        vov2 = self.vov**2

        beta = self.mr - self.vov*self.rho*uu
        dd = np.sqrt(beta**2 + vov2*uu*(1 - uu))
        gg = (beta - dd)/(beta + dd)
        exp = np.exp(-dd*texp)
        tmp1 = 1 - gg*exp

        mgf = self.mr*self.theta*((beta - dd)*texp - 2*np.log(tmp1/(1 - gg))) + var_0*(beta - dd)*(1 - exp)/tmp1
        return np.exp(mgf/vov2)

    def _cumulants(self, texp):
        """
        Analytic cumulants of log(S_T/F) for the Heston model.

        c1 uses HestonABC.avgvar_mv (Ball & Roma 1994 Appendix B).
        c2 follows Appendix A, Eq. (A.2) of Fang & Oosterlee (2008)
        and includes the rho term (avgvar_mv does not cover this).
        c4 is set to zero per F&O Section 5, keeping [a, b] well-conditioned.

        Returns:
            (c1, c2, 0.0, 0.0)
        """
        kap = self.mr
        eta = self.vov
        lam = self.theta
        v0  = self.sigma
        T   = texp

        # c1: -1/2 * E[integrated variance] — uses HestonABC helper
        c1 = -0.5 * texp * self.avgvar_mv(texp)[0]

        eT  = np.exp(-kap * T)
        e2T = np.exp(-2.0 * kap * T)

        # c2: Appendix A, Eq. (A.2) of Fang & Oosterlee (2008)
        c2 = (1.0 / (8.0 * kap**3)) * (
            eta * T * kap * eT * (v0 - lam) * (8.0 * kap * self.rho - 4.0 * eta)
          + kap * self.rho * eta * (1.0 - eT) * (16.0 * lam - 8.0 * v0)
          + 2.0 * lam * kap * T * (-4.0 * kap * self.rho * eta + eta**2 + 4.0 * kap**2)
          + eta**2 * ((lam - 2.0*v0)*e2T + lam*(6.0*eT - 7.0) + 2.0*v0)
          + 8.0 * kap**2 * (v0 - lam) * (1.0 - eT)
        )

        return float(c1), float(abs(c2)), 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Variance Gamma model
# ─────────────────────────────────────────────────────────────────────────────

class VarGammaCos(sv.SvABC, CosABC):
    """
    Variance Gamma (VG) European option pricing via the COS method.

    The VG model subordinates a drifted Brownian motion to a Gamma clock:
        log(S_T/F) = w*T + theta*G_T + sigma*W_{G_T}
    where G_T ~ Gamma(T/nu, nu) and w = (1/nu)*log(1 - theta*nu - sigma^2*nu/2)
    is the martingale drift correction.

    Parameter convention matches ``VarGammaFft`` (same SvABC names):
        sigma  - volatility of the Brownian subordinate (sigma)
        vov    - Gamma clock variance rate (nu > 0)
        theta  - drift of the Brownian subordinate (controls skewness;
                 defaults to sigma if not set)
        rho    - not used (inherited from SvABC)
        mr     - not used (inherited from SvABC)

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.VarGammaCos(sigma=0.12, vov=0.2, theta=-0.14, intr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.0)

    References:
        Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31:826-848,
        Section 5.4, Eq. (31) and Table 11.
        Madan D, Carr P, Chang E (1998) The Variance Gamma Process and Option Pricing.
        European Finance Review 2:79-105.
    """

    def mgf_logprice(self, uu, texp):
        """
        VG log-price MGF from F&O (2008) Eq. (31).
        Matches VarGammaFft.mgf_logprice formula (in-place exp rewritten for clarity).

        M(s) = exp(s*w*T) / (1 - s*theta*nu - 0.5*s^2*sigma^2*nu)^(T/nu)
        """
        nu = self.vov
        volvar = nu * self.sigma**2
        mu = np.log(1 - self.theta*nu - 0.5*volvar)    # = nu * martingale correction w
        rv = mu*uu - np.log(1 + (-self.theta*nu - 0.5*volvar*uu)*uu)
        return np.exp(texp/nu * rv)

    def _cumulants(self, texp):
        """
        Analytic VG cumulants from F&O (2008) Table 11.

        Returns:
            (c1, c2, 0.0, c4)
        """
        nu   = self.vov
        sig2 = self.sigma**2
        w    = np.log(1 - self.theta*nu - 0.5*sig2*nu) / nu
        c1   = texp * (w + self.theta)
        c2   = (sig2 + nu*self.theta**2) * texp
        c4   = 3*(sig2**2*nu + 2*self.theta**4*nu**3 + 4*sig2*self.theta**2*nu**2) * texp
        return float(c1), float(abs(c2)), 0.0, float(abs(c4))
