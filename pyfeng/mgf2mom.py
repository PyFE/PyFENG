# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:09:29 2019

"""

import math
import warnings
import numpy as np

class Mgf2Mom:
    """
    Choudhury and Lucantoni (1996)'s algorithm for calculating moments from moment generating function (MGF)

    References:
        - Choudhury GL, Lucantoni DM (1996) Numerical Computation of the Moments of a Probability Distribution from its Transform. Operations Research 44:368–381. https://doi.org/10.1287/opre.44.2.368
    """

    mgf = None
    l = 1

    def __init__(self, mgf, l=1, mean=None, var=None):
        """
        Args:
            mgf: moment generating function M(u) or any analytic function
            l: parameter controlling the quadrature radius (default 1)
            mean: analytic mean E[X]; if provided together with var, skips numerical pre-estimation
            var: analytic variance Var[X]; if provided together with mean, skips numerical pre-estimation
        """
        self.mgf = mgf
        self.l = l
        self.mean = mean
        self.var = var

    @staticmethod
    def radius(n, gamma=11, l=1):
        rr = np.power(10, -gamma / (2 * n * l))
        return rr

    def moment_raw(self, n, alpha, l=None):
        """
        Raw moment of order n. Eq. (33) in the reference.

        Args:
            n: order
            alpha: alpha to use
            l: l

        Returns:

        """

        if l is None:
            l = self.l

        r_n = self.radius(n, l=l)
        mgf = self.mgf(alpha*r_n*np.array([1.0, -1.0])).real

        #calc part3
        kk = np.arange(1, n*l)
        power1 = np.pi * kk * 1j / (n*l)
        power2 = -np.pi * kk * 1j / l
        #calc the value
        part3 = np.sum(np.real(self.mgf(alpha*r_n * np.exp(power1)) * np.exp(power2)))

        mu_n = math.factorial(n) / (2*n*l*(alpha*r_n)**n) * (mgf[0] + (-1)**n * mgf[1] + 2.0 * part3)
        return mu_n

    def _run(self, n, m1, m2):
        """
        Core recurrence (Algo 3 of Choudhury & Lucantoni 1996) given pre-computed
        first two values m1 and m2 (moments or cumulants depending on context).

        Args:
            n: number of values to compute
            m1: first value (used to select alpha_1)
            m2: second value (used to select alpha_1)

        Returns:
            array of length n
        """
        out = np.zeros(n)
        alpha_1 = 2.0 * m1 / m2
        out[0] = self.moment_raw(1, alpha=alpha_1)
        out[1] = self.moment_raw(2, alpha=alpha_1)
        for i in range(3, n + 1):
            alpha_i = (i - 1) * out[i - 3] / out[i - 2]
            out[i - 1] = self.moment_raw(i, alpha=alpha_i)
        return out

    @staticmethod
    def _check_replace(out, idx, analytic, label):
        """Warn if out[idx] differs from analytic value, then replace it."""
        if not np.isclose(out[idx], analytic, rtol=1e-4):
            warnings.warn(
                f"Analytic {label} ({analytic:.6g}) differs from numerical estimate ({out[idx]:.6g}).",
                UserWarning,
            )
        out[idx] = analytic

    def moments(self, n):
        """
        Raw moments E[X], E[X²], ..., E[Xⁿ].

        If mean and/or var are provided at construction, they seed the alpha selection
        and replace the corresponding computed values after the run (with a sanity check).

        Args:
            n: highest moment order (must be >= 2)

        Returns:
            array of length n with raw moments of orders 1 through n
        """
        if n < 2:
            warnings.warn("n must be >= 2. Setting n=2.", UserWarning)
            n = 2

        # Seed values for alpha selection
        if self.mean is not None:
            m1 = float(self.mean)
            m2 = float(self.var) + m1**2 if self.var is not None else self.moment_raw(2, alpha=1.0 / m1, l=1)
        else:
            m1 = self.moment_raw(1, alpha=1, l=2)
            m2 = float(self.var) + m1**2 if self.var is not None else self.moment_raw(2, alpha=1.0 / m1, l=1)

        out = self._run(n, m1, m2)

        # Sanity check and replace with analytic values
        if self.mean is not None:
            self._check_replace(out, 0, float(self.mean), "mean (μ₁)")
        if self.var is not None:
            m2_analytic = float(self.var) + out[0]**2
            self._check_replace(out, 1, m2_analytic, "second moment (μ₂)")

        return out

    def cumulants(self, n):
        """
        Cumulants κ₁, κ₂, ..., κₙ via the CGF K(u) = log M(u).

        The Choudhury-Lucantoni algorithm is applied to K(u) instead of M(u);
        its output equals the Taylor coefficients of K times n!, i.e. the cumulants.

        If mean and/or var are provided at construction, they seed the alpha selection
        and replace the corresponding computed cumulants after the run (with a sanity check).

        Args:
            n: highest cumulant order (must be >= 2)

        Returns:
            array of length n with cumulants of orders 1 through n
        """
        if n < 2:
            warnings.warn("n must be >= 2. Setting n=2.", UserWarning)
            n = 2

        cgf = Mgf2Mom(lambda u: np.log(self.mgf(u)), l=self.l)

        # Seed values for alpha selection
        if self.mean is not None:
            k1 = self.mean
            k2 = self.var if self.var is not None else cgf.moment_raw(2, alpha=1.0 / k1, l=1)
        else:
            k1 = cgf.moment_raw(1, alpha=1, l=2)
            k2 = self.var if self.var is not None else cgf.moment_raw(2, alpha=1.0 / k1, l=1)

        out = cgf._run(n, k1, k2)

        # Sanity check and replace with analytic values
        if self.mean is not None:
            self._check_replace(out, 0, float(self.mean), "mean (κ₁)")
        if self.var is not None:
            self._check_replace(out, 1, float(self.var), "variance (κ₂)")

        return out
