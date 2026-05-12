# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:29:04 2021

@author: cy-wang15 / Zay
"""

import math
import numpy as np
import scipy.stats as spst
import mpmath as m
import sympy
from .opt_abc import OptABC
from .params import BsmParams
from .util import MathFuncs
from . import nsvh


class BsmAsianJsu(BsmParams, OptABC):
    """

    Johnson's SU distribution approximation for Asian option pricing under the BSM model.

    Note: Johnson's SU distribution is the solution of NSVh with NSVh with lambda = 1.

    References:
        Posner SE, Milevsky MA (1998) Valuing exotic options by approximating the SPD
        with higher moments. The Journal of Financial Engineering 7(2). https://ssrn.com/abstract=108539

        Choi J, Liu C, Seo BK (2019) Hyperbolic normal stochastic volatility model.
        Journal of Futures Markets 39(2):186–204. https://doi.org/10.1002/fut.21967

    """

    def price_mnc(self, texp, n=1):
        """

        The n-th raw moment of S_bar/spot = (1/T)∫(S_t/spot) dt with spot=1.
        Multiply by spot^n to recover E[S_bar^n].

        Args:
            texp: time to expiry
            n: moment order

        References:
            Geman H, Yor M (1993) Bessel processes, Asian options, and perpetuities.
            Mathematical Finance 3(4):349–375.

        Returns: the nth moment (at spot=1)

        """
        # [Verified: Claude Sonnet 4.6, 2026-05-08]
        # Implements Geman-Yor (1993) eq. on p.359 for the n-th raw moment of the time average
        #   S_bar = (1/T) * integral_0^T S_t dt,  S_t = S_0 * exp((r-q-sigma^2/2)*t + sigma*B_t).
        #
        # Notation map (code -> G-Y):  lam=sigma, v=(r-q-sigma^2/2)/sigma, beta=v/sigma=v/lam.
        # The integral is rescaled to G-Y form A_t^(nu) = int_0^t exp(lambda*(W_s+nu*s)) ds
        # with lambda=sigma, nu=v, t=T. This gives:
        #   E[(S_bar/S_0)^n] = (1/T)^n * n!/lambda^(2n) * sum_{j=0}^{n} d_j * exp(Sigma_j * T)
        # where Sigma_j = lambda^2*j^2/2 + lambda*j*v = j*(r-q) + j*(j-1)*sigma^2/2  (verified)
        # and   d_j = 2^n * prod_{i!=j} [(beta+j)^2 - (beta+i)^2]^(-1)               (verified)
        # The j=0 term (exp(0)=1) absorbs the "-1" in (exp(Sigma_k*T)-1) of the explicit form.
        # Both n=1 and n=2 cases verified algebraically; n=1 gives (exp((r-q)*T)-1)/((r-q)*T).
        #
        # KNOWN BUG: division by zero when 2*beta+i+j=0 for any 0<=i<j<=n, i.e., when
        # r-q = sigma^2*(1-i-j)/2. For n<=4 the first case is r=q (i=0,j=1). No fix implemented.
        lam = self.sigma
        v = (self.intr - self.divr - lam ** 2 / 2) / lam
        beta = v / lam

        ds = []
        for j in range(n + 1):
            item0 = 2 ** n
            for i in range(n + 1):
                if i != j:
                    item0 *= ((beta + j) ** 2 - (beta + i) ** 2) ** (-1)
            ds.append(item0)
        item1 = 0
        for i in range(n + 1):
            item1 += ds[i] * np.exp((lam ** 2 * i ** 2 / 2 + lam * i * v) * texp)
        moment = (1.0 / texp) ** n * math.factorial(n) / lam ** (2 * n) * item1

        return moment

    def price_mnc4(self, texp):
        """
        First 4 non-central (raw) moments of S_bar/spot = (1/T)∫(S_t/spot) dt at spot=1,
        as hard-coded polynomials in A = exp((r-q)T) and B = exp(sigma^2 T).
        Multiply m_n by spot^n to recover E[S_bar^n].

        Let p = 2*(r-q)/sigma^2, v = sigma^2*T.  Then:

            m1 = (A-1) / ((r-q)*T)                  [via avg_exp for r=q stability]
            m2 = 4/v^2 * N2 / (p(p+1)(p+2))
            m3 = 8/v^3 * N3 / (p(p+1)(p+2)(p+3)(p+4))
            m4 = 16/v^4 * N4 / (p(p+1)...(p+6))

        with binomial-coefficient numerators:
            N2 = (p+2) - 2(p+1)*A + p*A^2*B
            N3 = -(p+3)(p+4) + 3(p+1)(p+4)*A - 3p(p+3)*A^2*B + p(p+1)*A^3*B^3
            N4 = (p+4)(p+5)(p+6) - 4(p+1)(p+5)(p+6)*A + 6p(p+3)(p+6)*A^2*B
                 - 4p(p+1)(p+5)*A^3*B^3 + p(p+1)(p+2)*A^4*B^6

        Derived from Geman-Yor (1993) formula via Lagrange interpolation over d_j coefficients;
        replaces the O(n^2) Python loop in price_mnc() with a single vectorised expression.

        KNOWN BUG: division by zero when p = 0, i.e. r = q.  No fix implemented (same as price_mnc()).

        Args:
            texp: time to expiry

        Returns:
            (m1, m2, m3, m4) — raw moments at spot=1
        """
        # [Verified: Claude Sonnet 4.6, 2026-05-08]
        # Derived from Geman-Yor (1993) formula (see price_mnc() docstring for notation).
        # p = 2*(r-q)/sigma^2 = 2*beta + 1  (beta = (r-q-sigma^2/2)/sigma^2).
        # Exponents: Sigma_j*T = j*(r-q)*T + j*(j-1)/2 * sigma^2*T = j*u + j*(j-1)/2 * v
        #   -> e^{Sigma_j*T} = A^j * B^{j*(j-1)/2}:  e0=1, e1=A, e2=A^2*B, e3=A^3*B^3, e4=A^4*B^6
        # d_j = 2^n / prod_{i!=j}[(j-i)*(alpha+j+i)] with alpha = p-1;
        # collecting d_j*e_j over common denominator gives the numerators N2, N3, N4 above.
        # Binomial outer coefficients (1,-2,1), (-1,3,-3,1), (1,-4,6,-4,1) are verified from
        # the alternating product signs in the d_j denominators.  All polynomial factors
        # verified term-by-term from the d_j derivation.
        v = self.sigma**2 * texp
        u = (self.intr - self.divr) * texp
        A = np.exp(u)
        B = np.exp(v)
        p = 2.0 * (self.intr - self.divr) / self.sigma**2                # 2*(r-q)/sigma^2

        A2B = A * A * B
        A3B3 = A2B * A * B * B
        A4B6 = A3B3 * A * B * B * B

        m1 = MathFuncs.avg_exp(u)      # stable at u -> 0  (r = q)

        m2 = (4.0 / v**2) * (
            (p + 2) - 2*(p + 1)*A + p*A2B
        ) / (p*(p + 1)*(p + 2))

        m3 = (8.0 / v**3) * (
            -(p + 3)*(p + 4) + 3*(p + 1)*(p + 4)*A
            - 3*p*(p + 3)*A2B + p*(p + 1)*A3B3
        ) / (p*(p + 1)*(p + 2)*(p + 3)*(p + 4))

        m4 = (16.0 / v**4) * (
              (p + 4)*(p + 5)*(p + 6)
            - 4*(p + 1)*(p + 5)*(p + 6)*A
            + 6*p*(p + 3)*(p + 6)*A2B
            - 4*p*(p + 1)*(p + 5)*A3B3
            + p*(p + 1)*(p + 2)*A4B6
        ) / (p*(p + 1)*(p + 2)*(p + 3)*(p + 4)*(p + 5)*(p + 6))

        return m1, m2, m3, m4

    def price_mvsk(self, texp):
        """
        Mean, var_scaled, skewness, and raw kurtosis of S_bar at spot=1.

        var_scaled = var / mean²  = (std/mean)² = mu2/mu1² is scale-invariant (spot cancels).
        Rescale: mean *= spot; var_scaled, skew, kurt are all scale-invariant.

        For small v = sigma^2*T the raw-moment formulas have 1/v^n singularities that
        cancel analytically.  mu2 uses the aa/bb form (no blow-up); mu3/mu4 switch to
        Taylor at v < 0.01:

            mu2: aa/bb form, finite everywhere; Taylor for precision only
            mu3 = 2*v^2/5 + O(v^3)          [independent of p to leading order]
            mu4 = v^2/3 + O(v^3)             [normal-distribution limit 3*mu2^2]

        where p = 2*(r-q)/sigma^2.

        KNOWN BUG: division by zero when p = 0, i.e. r = q.

        Args:
            texp: time to expiry

        Returns:
            (mean, var_scaled, skewness, raw kurtosis) at spot=1
        """
        # [Verified: Claude Sonnet 4.6, 2026-05-08]
        # Taylor derivation (fixed p, expand in v):
        #   A = exp(pv/2), B = exp(v), A²B = exp((p+1)v), A³B³ = exp(3(p+2)v/2).
        #   N2 = p(p+1)(p+2)*v²/4 + O(v³)  → m2 = 1 + (3p+2)v/6 + O(v²)
        #   m2 - m1² = v/3 + (5p+2)v²/24 + O(v³)  [verified term by term]
        #   mu3: coefficients of v and v² in m3-3m2m1+2m1³ vanish;
        #        v² coeff = (25p²+70p+52)/80 - (15p²+14p+4)/16 + 5p²/8 = 32/80 = 2/5  ✓
        #   mu4: normal-distribution limit 3*(v/3)² = v²/3  ✓
        v = self.sigma**2 * texp
        u = (self.intr - self.divr) * texp
        A = np.exp(u)
        B = np.exp(v)
        p = 2.0 * (self.intr - self.divr) / self.sigma**2   # 2*(r-q)/sigma^2

        A2B  = A * A * B
        A3B3 = A2B  * A * B * B
        A4B6 = A3B3 * A * B * B * B

        m1 = MathFuncs.avg_exp(u)

        m2 = (4.0 / v**2) * (
            (p+2) - 2*(p+1)*A + p*A2B
        ) / (p*(p+1)*(p+2))

        m3 = (8.0 / v**3) * (
            -(p+3)*(p+4) + 3*(p+1)*(p+4)*A
            - 3*p*(p+3)*A2B + p*(p+1)*A3B3
        ) / (p*(p+1)*(p+2)*(p+3)*(p+4))

        m4 = (16.0 / v**4) * (
              (p+4)*(p+5)*(p+6)
            - 4*(p+1)*(p+5)*(p+6)*A
            + 6*p*(p+3)*(p+6)*A2B
            - 4*p*(p+1)*(p+5)*A3B3
            + p*(p+1)*(p+2)*A4B6
        ) / (p*(p+1)*(p+2)*(p+3)*(p+4)*(p+5)*(p+6))

        # mu2: aa=m1=avg_exp(u), bb=avg_exp(v) → p²v² denominator cancels algebraically.
        # (bb-m1)/v → (2-p)/4 as v→0 (finite); Taylor branch for precision only.
        bb = MathFuncs.avg_exp(v)
        diff_over_v = np.where(v < 0.01, (2-p)/4 + v*(4-p**2)/24, (bb-m1)/v)
        mu2 = (4*diff_over_v + 4*p*m1*bb + m1**2*(p**2*bb*v - 3*p - 2)) / ((p+1)*(p+2))

        mu3 = np.where(v < 0.01, 2*v**2/5,       m3 - 3*m2*m1 + 2*m1**3)
        mu4 = np.where(v < 0.01, v**2/3,  m4 - 4*m3*m1 + 6*m2*m1**2 - 3*m1**4)

        # var_scaled = var/mean²;  skew = mu3/mu2^(3/2);  raw kurt = mu4/mu2^2.
        var_scaled = mu2 / m1**2
        skew = mu3 / (mu2*np.sqrt(mu2))
        kurt = mu4 / mu2**2

        return m1, var_scaled, skew, kurt

    def price(self, strike, spot, texp, cp=1):
        """

        Asian options price.
        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
        Returns: Basket options price

        """
        df = np.exp(-texp * self.intr)

        mu, var_scaled, skew, kurt = self.price_mvsk(texp)
        mu *= spot
        price = nsvh.Nsvh1.from_vsk(
            (var_scaled * mu**2, skew, kurt - 3), texp=texp, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd
        ).price(strike, mu, texp, cp)

        return df * price


class BsmAsianLinetsky2004(BsmParams, OptABC):

    b = 1.0
    n_eig = 50

    def nu(self):
        nu = 2 * (self.intr - self.divr) / (self.sigma ** 2) - 1
        return nu

    def find_zeros_real(self):
        nu = self.nu()

        eigenval = []
        if nu <= -2:
            for n in range(2, math.floor(np.abs(nu) / 2) + 1 + 1):
                eigenvalue = np.abs(nu) - 2 * n + 2
                eigenval.append(eigenvalue)
        else:
            pass
        return np.array(eigenval)

    @staticmethod
    def positive_eigenvalue(eigval):
        eigval_p = []
        for i in range(len(eigval)):
            real_i = np.array(eigval[i], dtype=complex).real[0]
            if real_i > 0:
                eigval_p.append(real_i)
            else:
                pass
        return eigval_p

    def find_zeros_imag(self, n):
        nu = self.nu()

        p = sympy.symbols("p")
        eigenval = []
        for j in range(1, n):
            p_tilde = sympy.solve(
                p * (sympy.log(4 * p) - 1) - 2 * np.pi * (j + nu / 4 - 1 / 2), p
            )
            eigenval.append(p_tilde)
        eigenval = self.positive_eigenvalue(np.real(eigenval))
        eigenval = np.array(eigenval).transpose()
        return eigenval

    def eta_q(self, eigenval):
        nu = self.nu()

        f = lambda x: m.whitw((1 - nu) / 2, x / 2, 1 / (2 * self.b))
        return complex((f(eigenval-1e-8) - f(eigenval-1e-8))/2e-8)

    def xi_p(self, eigenval):
        nu = self.nu()
        func = lambda x: m.whitw((1 - nu) / 2, complex(0, x / 2), 1 / (2 * self.b))
        return complex((func(eigenval+1e-8) - func(eigenval-1e-8))/2e-8)

    def price_element_imag(self, tau, p_value, k):
        nu = self.nu()

        p1 = m.exp(-(nu ** 2 + p_value ** 2) * tau / 2)
        p2 = (p_value * m.gamma(complex(nu / 2, p_value / 2))) / (
            4
            * self.xi_p(eigenval=p_value)
            * m.gamma(complex(1, p_value))
        )
        const = (2 * k) ** ((nu + 3) / 2) * m.exp(-1 / (4 * k))
        w1 = m.whitw(-(nu + 3) / 2, complex(0, p_value / 2), 1 / (2 * k))
        m1 = m.whitm((1 - nu) / 2, complex(0, p_value / 2), 1 / (2 * self.b))
        return p1 * p2 * const * w1 * m1

    def price_element_real(self, tau, q_value, k):
        nu = self.nu()

        p1 = m.exp(-(nu ** 2 - q_value ** 2) * tau / 2)
        p2 = (q_value * m.gamma((nu + q_value) / 2)) / (
            4 * self.eta_q(eigenval=q_value) * m.gamma(1 + q_value)
        )
        const = (2 * k) ** ((nu + 3) / 2) * m.exp(-1 / (4 * k))
        w1 = m.whitw(-(nu + 3) / 2, q_value / 2, 1 / (2 * k))
        m1 = m.whitm((1 - nu) / 2, q_value / 2, 1 / (2 * self.b))
        return p1 * p2 * const * w1 * m1

    def price(self, strike, spot, texp, cp=1):
        tau = self.sigma ** 2 * texp / 4
        k = tau * strike / spot

        p = self.find_zeros_imag(n=self.n_eig + 1)
        q = self.find_zeros_real()

        imaginary_terms = []
        for i in range(len(p)):
            imaginary_term = self.price_element_imag(tau=tau, p_value=p[i], k=k)
            imaginary_terms.append(complex(imaginary_term))

        real_terms = []
        for i in range(len(q)):
            real_term = self.price_element_real(tau=tau, q_value=q[i], k=k)
            real_terms.append(complex(real_term))

        P = np.sum(imaginary_terms) + np.sum(real_terms)

        fwd, df, divf = self._fwd_factor(spot, texp)

        # undiscounted put price
        price = (spot / tau) * P.real
        # undiscounted call price
        if cp > 0:
            # Eq. (3)
            if np.abs((self.intr - self.divr) * texp) < 1e-8:
                fac = 1 + (self.intr - self.divr) * texp / 2
            else:
                fac = (divf/df - 1) / ((self.intr - self.divr) * texp)
            price += spot * fac - strike

        return df * price


class BsmContinuousAsianJu2002(OptABC):
    def price(self, strike, spot, texp, cp=1):

        if np.isscalar(spot) == False:
            raise ValueError("spot must be a scalar.")
        elif np.isscalar(self.divr) == False:
            raise ValueError("dividend (divr) must be a scalar.")
            return 0
        else:
            g = self.intr - self.divr
            gt = g * texp
            u1 = spot * MathFuncs.avg_exp(gt)
            g2 = 2 * g + self.sigma ** 2
            u2 = (
                2 * spot ** 2
                * (MathFuncs.avg_exp(g2 * texp) - MathFuncs.avg_exp(gt))
                / texp
                / (g + self.sigma ** 2)
            )
            z1 = -pow(self.sigma, 4) * texp ** 2 * (
                1 / 45
                + gt / 180
                - 11 * gt ** 2 / 15120
                - pow(gt, 3) / 2520
                + pow(gt, 4) / 113400
            ) - pow(self.sigma, 6) * pow(texp, 3) * (
                1 / 11340
                - 13 * gt / 30480
                - 17 * gt ** 2 / 226800
                + 23 * pow(gt, 3) / 453600
                + 59 * pow(gt, 4) / 5987520
            )
            z2 = -pow(self.sigma, 4) * texp ** 2 * (
                1 / 90
                + gt / 360
                - 11 * gt ** 2 / 30240
                - pow(gt, 3) / 5040
                + pow(gt, 4) / 226800
            ) - pow(self.sigma, 6) * pow(texp, 3) * (
                31 / 22680
                - 11 * gt / 60480
                - 37 * gt ** 2 / 151200
                - 19 * pow(gt, 3) / 302400
                + 953 * pow(gt, 4) / 59875200
            )
            z3 = (
                pow(self.sigma, 6)
                * pow(texp, 3)
                * (
                    2 / 2835
                    - gt / 60480
                    - 2 * gt ** 2 / 14175
                    - 17 * pow(gt, 3) / 907200
                    + 13 * pow(gt, 4) / 1247400
                )
            )
            m1 = 2 * np.log(u1) - 0.5 * np.log(u2)
            v1 = np.log(u2) - 2 * np.log(u1)
            sqrtv1 = np.sqrt(v1)
            y = np.log(strike)
            y1 = (m1 - y) / np.sqrt(v1) + sqrtv1
            y2 = y1 - sqrtv1
            bc = (
                u1 * np.exp(-self.intr * texp) * spst.norm._cdf(y1)
                - strike * np.exp(-self.intr * texp) * spst.norm._cdf(y2)
                + np.exp(-self.intr * texp)
                * strike
                * (
                    z1 * spst.norm._pdf((y - m1) / sqrtv1) / sqrtv1
                    + z2 * spst.norm._pdf((y - m1) / sqrtv1) / sqrtv1 * (m1 - y) / v1
                    + z3
                    * ((y - m1) * (y - m1) / v1 / v1 - 1 / v1)
                    * spst.norm._pdf((y - m1) / sqrtv1) / sqrtv1
                )
            )
        if cp == 1:
            return bc
        elif cp == -1:
            return np.exp(-self.intr * texp) * (strike - u1) + bc
        else:
            return -1