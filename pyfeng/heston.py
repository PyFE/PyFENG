import warnings
import numpy as np
from .opt_abc import OptABC
from . import bsm
from .util import MathFuncs
from .params import HestonParams


class CirModel:
    """
    Cox-Ingersoll-Ross (CIR) process:

        dV_t = mr * (theta - V_t) dt + sigma * sqrt(V_t) dW_t

    Parameters:
        mr    (κ > 0): mean-reversion speed
        theta (θ > 0): long-run mean
        sigma (ξ > 0): diffusion coefficient

    References:
        Cox JC, Ingersoll JE, Ross SA (1985) A theory of the term structure
        of interest rates. Econometrica 53:385–407.
        https://doi.org/10.2307/1911242

    # [Created: Claude Sonnet 4.6, 2026-05-10]
    """

    sigma: float = 0.2
    mr: float = 1.0
    theta: float = 0.04

    def __init__(self, sigma=0.2, mr=1.0, theta=0.04):
        self.sigma = sigma
        self.mr = mr
        self.theta = theta

    def log_mgf(self, uu, texp, v0):
        """
        Log-MGF of the integrated CIR process J_T = ∫₀ᵀ V_t dt:

            log E[exp(uu · J_T) | V_0 = v0] = A(T, uu) + B(T, uu) · v0

        with γ = √(κ² − 2·uu·ξ²) and Δ = 2γ + (κ+γ)(e^{γT}−1):

            B(T, uu) = 2·uu·(e^{γT}−1) / Δ
            A(T, uu) = (κθ/ξ²) · [(κ+γ)T − 2 ln(Δ / 2γ)]

        The log(Δ/2γ) = log1p((κ+γ)/(2γ) · (e^{γT}−1)) form is used for
        numerical stability when γT or uu is small.  Valid for uu < κ²/(2ξ²).

        Args:
            uu: MGF argument (uu < κ²/(2ξ²))
            texp: time horizon T (> 0)
            v0: initial value V_0 (≥ 0)

        Returns:
            log-MGF value A(T, uu) + B(T, uu) · v0
        """
        gamma = np.sqrt(self.mr**2 - 2 * uu * self.sigma**2)
        expm1_gT = np.expm1(gamma * texp)
        Delta = 2*gamma + (self.mr + gamma) * expm1_gT
        B = 2 * uu * expm1_gT / Delta
        A = self.mr * self.theta / self.sigma**2 * (
            (self.mr + gamma) * texp - 2 * np.log1p((self.mr + gamma) / (2*gamma) * expm1_gT)
        )
        return A + B * v0

    def mgf(self, uu, texp, v0):
        """
        MGF of the integrated CIR process J_T = ∫₀ᵀ V_t dt:

            E[exp(uu · J_T) | V_0 = v0]

        Valid for uu < κ²/(2ξ²).

        Args:
            uu: MGF argument
            texp: time horizon T (> 0)
            v0: initial value V_0 (≥ 0)

        Returns:
            MGF value
        """
        return np.exp(self.log_mgf(uu, texp, v0))

    def cum4_numeric(self, texp, v0):
        """
        First four cumulants of V_T given V_0 = v0 via numerical differentiation.

        V_T | V_0 follows a scaled noncentral χ²: V_T = α·W, W ~ χ²(d, λ), where

            α = ξ²(1 − e^{−κT}) / (4κ)
            d = 4κθ / ξ²
            λ = v0·e^{−κT} / α

        The MGF is M(u) = (1 − 2αu)^{−d/2} · exp(λαu / (1 − 2αu)), valid for u < 1/(2α).
        The first two cumulants match mv exactly: κ₁ = E[V_T], κ₂ = Var[V_T].

        Args:
            texp: time horizon T (> 0)
            v0: initial value V_0 (≥ 0)

        Returns:
            (κ₁, κ₂, κ₃, κ₄): first four cumulants of V_T
        """
        from .mgf2mom import Mgf2Mom
        mr_t = self.mr * texp
        alpha = self.sigma**2 * (-np.expm1(-mr_t)) / (4 * self.mr)
        d = 4 * self.mr * self.theta / self.sigma**2
        lam = v0 * np.exp(-mr_t) / alpha

        def mgf_vt(u):
            u = np.asarray(u, dtype=complex)
            return np.exp(-(d/2) * np.log1p(-2*alpha*u) + lam * alpha * u / (1 - 2*alpha*u))

        cum = Mgf2Mom(mgf_vt).cumulants(4)
        return float(cum[0]), float(cum[1]), float(cum[2]), float(cum[3])

    def mv(self, dt, v0):
        """
        Mean and variance of V(t+dt) given V(t) = v0.

        Exact CIR conditional moments for dV = κ(θ−V)dt + ξ√V dW:

            E[V_{t+dt} | V_t = v0] = θ + (v0 − θ) e^{−κdt}
            Var[V_{t+dt} | V_t = v0] = ξ²·dt·φ·(v0·e^{−κdt} + θ·κdt·φ / 2)

        where φ = avg_exp(−κdt) = (1 − e^{−κdt}) / (κdt).

        Args:
            dt: time step
            v0: initial value V(t) = v0

        Returns:
            (mean, variance)
        """
        # [Verified: Claude Sonnet 4.6, 2026-05-08]
        # Mean: E[V_{t+dt}|V_t=v0] = θ + (v0-θ)e^{-κdt}  ✓
        # Variance: with φ=avg_exp(-κdt)=(1-e)/κdt, expanding gives
        #   s2 = ξ²/κ * [v0*e*(1-e) + θ*(1-e)²/2]
        # which matches Var = v0*(ξ²/κ)*(e^{-κdt}-e^{-2κdt}) + θ*(ξ²/2κ)*(1-e^{-κdt})²  ✓
        mr_t = self.mr * dt
        e_mr = np.exp(-mr_t)
        mean = self.theta + (v0 - self.theta) * e_mr
        avg = MathFuncs.avg_exp(-mr_t)
        var = self.sigma**2 * dt * avg * (v0 * e_mr + self.theta * mr_t * avg / 2)
        return mean, var

    def chi_dim(self):
        """Noncentral chi-squared degrees of freedom: 4·θ·κ/σ²."""
        return 4 * self.theta * self.mr / self.sigma**2

    def avg_mv(self, texp, v0):
        """
        Mean, variance, and 3rd central moment of the average variance I_T = (1/T)∫v dt
        under the CIR process given V(0) = v0.

        Args:
            texp: time horizon T
            v0: initial value V(0)

        Returns:
            (mean, variance, c3) where c3 = E[(I_T − E[I_T])³]

        References:
            Ball C, Roma A (1994) Stochastic Volatility Option Pricing.
            Journal of Financial and Quantitative Analysis 29:589–607. Appendix B.
        """
        mr_t = self.mr * texp
        e_mr = np.exp(-mr_t)
        phi = MathFuncs.avg_exp(-mr_t)
        x0 = v0 - self.theta
        mean = self.theta + x0 * phi
        var = (self.theta - 2*x0*e_mr) + (v0 - 2.5*self.theta + (v0 - self.theta/2)*e_mr) * phi
        var *= (self.sigma / mr_t)**2 * texp

        fac = (self.sigma / self.mr)**4 / self.mr

        b3 = 3*fac * ((1 + 0.5*e_mr*(1 - e_mr*(2 + e_mr))) - mr_t*e_mr*(1 + mr_t + 2*e_mr))
        int_b3 = fac*texp * ((3 - 11*phi + e_mr*(9 - 3.5*phi + e_mr*(3 - 0.5*phi))) + 3*mr_t*e_mr)
        a3 = self.mr * self.theta * int_b3

        v0f = float(v0)
        c3_small = (self.sigma**2 * texp)**2 * (v0f/5 + mr_t*(self.theta/30 - 17*v0f/60) + mr_t**2*(3*v0f/14 - 17*self.theta/420))
        c3_full = (a3 + b3 * v0f) / texp**3
        c3 = np.where(mr_t < 0.1, c3_small, c3_full)

        return mean, var, c3

    def phi_exp(self, dt):
        """phi and exp(-κ·dt/2) used in NCX2/Poisson-Gamma steps."""
        exp_h = np.exp(-self.mr * dt / 2)
        phi = 4 * self.mr / self.sigma**2 / (1/exp_h - exp_h)
        return phi, exp_h

    def draw_euler(self, dt, v0, rng, milstein=False):
        """
        Euler (or Milstein) step for the CIR variance process.

        Args:
            dt: time step
            v0: initial variance (array)
            rng: numpy random Generator
            milstein: enable Milstein correction (default False)

        Returns:
            variance after dt (floored at 0)
        """
        zz = rng.standard_normal(size=np.shape(v0))
        v_t = v0 + self.mr * (self.theta - v0) * dt + np.sqrt(v0) * self.sigma * zz
        if milstein:
            v_t = v_t + 0.25 * self.sigma**2 * (zz**2 - dt)
        return np.maximum(v_t, 0.0)

    def draw_ncx2(self, dt, v0, rng):
        """
        Exact CIR step by drawing from the noncentral chi-squared distribution.

        Args:
            dt: time step
            v0: initial variance (array)
            rng: numpy random Generator

        Returns:
            variance after dt
        """
        chi_df = 4 * self.mr * self.theta / self.sigma**2
        phi, exp_h = self.phi_exp(dt)
        return (exp_h / phi) * rng.noncentral_chisquare(
            df=chi_df, nonc=v0 * exp_h * phi, size=np.shape(v0)
        )

    def draw_pois_gamma(self, dt, v0, rng_gamma, rng_pois):
        """
        Exact CIR step via Poisson-mixture Gamma (NCX2 decomposition).

        Args:
            dt: time step
            v0: initial variance (array)
            rng_gamma: numpy random Generator for Gamma draws
            rng_pois: numpy random Generator for Poisson draws

        Returns:
            (variance after dt, Poisson RV used)
        """
        chi_df = 4 * self.mr * self.theta / self.sigma**2
        phi, exp_h = self.phi_exp(dt)
        chi_nonc = v0 * exp_h * phi
        pois = rng_pois.poisson(chi_nonc / 2, size=np.shape(v0))
        v_t = (exp_h / phi) * 2 * rng_gamma.standard_gamma(
            shape=chi_df / 2 + pois, size=np.shape(v0)
        )
        return v_t, pois

class HestonABC(HestonParams, OptABC):

    @property
    def cir(self):
        """CirModel instance for the variance process (ξ = vov, κ = mr, θ = theta)."""
        return CirModel(sigma=self.vov, mr=self.mr, theta=self.theta)


    def var_mv(self, dt, var0=None):
        """
        Mean and variance of the variance V(t+dt) given V(t) = var_0.

        Delegates to CirModel.mv with vov as the CIR diffusion coefficient (ξ)
        and var0 as the initial value.

        Args:
            dt: time step
            var0: initial variance (default: self.sigma)

        Returns:
            (mean, variance)
        """
        if var0 is None:
            var0 = self.sigma
        return self.cir.mv(dt, v0=var0)

    def avgvar_mv(self, texp, var0=None):
        """
        Mean, variance, and 3rd central moment of the average variance I_T = (1/T)∫v dt.
        Delegates to CirModel.avg_mv.

        Args:
            texp: time to expiry
            var0: initial variance (default: self.sigma)

        Returns:
            (mean, variance, c3) where c3 = E[(I_T − E[I_T])³]
        """
        if var0 is None:
            var0 = self.sigma
        return self.cir.avg_mv(texp, var0)

    def strike_var_swap_analytic(self, texp, dt):
        """
        Analytic fair strike of variance swap. Eq (11) in Bernard & Cui (2014)

        Args:
            texp: time to expiry
            dt: observation time step (e.g., dt=1/12 for monthly) For continuous monitoring, set dt=0

        Returns:
            Fair strike

        References:
            - Bernard C, Cui Z (2014) Prices and Asymptotics for Discrete Variance Swaps. Applied Mathematical Finance 21:140–173. https://doi.org/10.1080/1350486X.2013.820524

        """

        var0 = self.sigma

        ### continuously monitored fair strike (same as mean of avgvar)
        mr_t = self.mr*texp
        x0 = var0 - self.theta
        strike = self.theta + x0*MathFuncs.avg_exp(-mr_t)

        if not np.all(np.isclose(dt, 0.0)):
            ### adjustment for discrete monitoring
            mr_h = self.mr * dt
            e_mr_h = np.exp(-mr_h)

            tmp = self.theta - 2*(self.intr - self.divr)
            strike += tmp*dt/4 * (tmp + 2*x0*MathFuncs.avg_exp(-mr_t))

            tmp = self.vov / self.mr
            strike += self.theta * tmp * (tmp/4 - self.rho) * (1 - MathFuncs.avg_exp(-mr_h))
            strike += x0 * tmp * (tmp/2 - self.rho) * MathFuncs.avg_exp(-mr_t) * (1 + mr_h/(1 - 1/e_mr_h))
            strike += (tmp**2*(self.theta - 2*var0) + 2*x0**2/self.mr) * MathFuncs.avg_exp(-2*mr_t)/4 * (1-e_mr_h)/(1+e_mr_h)

        return strike

    def logp_mgf(self, uu, texp):
        """
        Log price MGF under the Heston model (Lord & Kahl 2010 branch-cut-safe form).

        We use the characteristic function in Eq (2.8) of Lord & Kahl (2010) that is
        continuous in branch cut when the complex log is evaluated.

        References:
            - Heston SL (1993) A Closed-Form Solution for Options with Stochastic
              Volatility with Applications to Bond and Currency Options.
              The Review of Financial Studies 6:327–343.
              https://doi.org/10.1093/rfs/6.2.327
            - Lord R, Kahl C (2010) Complex Logarithms in Heston-Like Models.
              Mathematical Finance 20:671–694.
              https://doi.org/10.1111/j.1467-9965.2010.00416.x
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

    def logp_mv(self, texp):
        """
        Mean and variance of log(S_T/F) under the Heston model (mu = 0).

        From the conditional decomposition (with Y = vov·∫√v dW):

            ln(S_T/F) = (rho/vov)·Y + rho_c·√(T·I_T)·X − T·I_T/2

        **Mean** (E[Y] = 0 by construction):

            E[Z] = −T/2 · E[I_T]

        **Variance** (law of total variance + Itô isometry Var(Y) = vov²·T·E[I_T]):

            Var[Z] = T·μ_I + T²/4·σ²_I − (ρT/vov)·Cov(Y, I_T)

        **Cov(Y, I_T)** (solving the ODE H'(t) + κH(t) = vov·t·E[v_t] via Itô + Fubini):

            Cov(Y, I_T) = (vov²/κ)·[θ·(1−φ) + (v₀−θ)·(φ − e^{−κT})]

        where φ = (1−e^{−κT})/(κT).

        Args:
            texp: time to expiry

        Returns:
            (mean, variance) of log(S_T/F)

        References:
            Le Floc'h F (2020) arXiv:2005.13248, Appendix B, Eq. (B5)
        """
        T = float(texp)
        mu_i, var_i, _ = self.avgvar_mv(T)

        mean = -0.5 * T * mu_i

        mr_t = self.mr * T
        e_mr = np.exp(-mr_t)
        phi = MathFuncs.avg_exp(-mr_t)
        x0 = float(self.sigma) - self.theta
        cov_yi = self.vov**2 / self.mr * (self.theta * (1 - phi) + x0 * (phi - e_mr))

        var = T * mu_i + (T**2 / 4) * var_i - self.rho * T / self.vov * cov_yi

        return mean, var


class HestonUncorrBallRoma1994(HestonABC):
    """
    Ball & Roma (1994)'s approximation pricing formula for European options under uncorrelated (rho=0) Heston model.
    Up to 3rd order (order=3) is implemented.

    See Also: OusvUncorrBallRoma1994, GarchUncorrBaroneAdesi2004
    """

    order = 2

    def price(self, strike, spot, texp, cp=1):

        if self.order > 3:
            warnings.warn(f"order = {self.order} is not implemented. Calculating up to order 3.")

        if not np.isclose(self.rho, 0.0):
            warnings.warn(f"Pricing ignores rho = {self.rho}.")

        if self.order >= 3:
            avgvar, var, c3 = self.avgvar_mv(texp)
        else:
            avgvar, var, _ = self.avgvar_mv(texp)

        m_bs = bsm.Bsm(np.sqrt(avgvar), intr=self.intr, divr=self.divr)
        price = m_bs.price(strike, spot, texp, cp)

        if self.order >= 2:
            price += 0.5*var*m_bs.d2_var(strike, spot, texp, cp)
        if self.order >= 3:
            price += (1/6)*c3*m_bs.d3_var(strike, spot, texp, cp)

        return price
