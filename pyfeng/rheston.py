import numpy as np
import scipy.special as spsp
from .opt_abc import OptABC
from .params import RoughHestonParams


class RoughHestonABC(RoughHestonParams, OptABC):
    """
    Rough Heston model Abstract class
    """

    '''
    Method 1: Adam's method
    '''

    def a_j_kp1(self, k, delta, j):
        """
        Weights calculation, equation 5.3 in El Euch O, Rosenbaum M (2019)
        Args:
            delta: time step length

        Returns:
            Weight
        """
        if j == 0:
            return pow(delta, self.alpha) * (pow(k, self.alpha + 1) - (k - self.alpha) * pow((k + 1), self.alpha)) / spsp.gamma(self.alpha + 2)
        elif j == k + 1:
            return pow(delta, self.alpha) / spsp.gamma(self.alpha + 2)
        else:
            return pow(delta, self.alpha) * (
                        pow(k - j + 2, self.alpha + 1) + pow(k - j, self.alpha + 1) - 2 * pow(k - j + 1, self.alpha + 1)) / spsp.gamma(
                self.alpha + 2)

    def b_j_kp1(self, k, delta, j):
        """
        Weights calculation in the approximation of h_p_hat
        Args:
            delta: time step length
        """
        return pow(delta, self.alpha) * (pow(k - j + 1, self.alpha) - pow(k - j, self.alpha)) / spsp.gamma(self.alpha + 1)

    def a_kp1(self, k, delta):
        """
        Calculation of weights array

        """
        a = np.zeros(k + 1)
        for i in range(0, k + 1):
            a[i] = self.a_j_kp1(k, delta, i)
        return a

    def b_kp1(self, k, delta):
        """
        Calculation of weights array

        """
        b = np.zeros(k + 1)
        for i in range(0, k + 1):
            b[i] = self.b_j_kp1(k, delta, i)
        return b

    def F(self, a, x):
        """
        Left hand side of fractional Riccati equation. It is transformed to get the moment-generating function.

        Args:
            a: dummy variable
            x: value of the solution function

        """
        return (1/2) * (pow(a, 2) - a) + self.mr * (a * self.rho * self.vov - 1) * x + pow(self.mr * self.vov, 2) * pow(x, 2) / 2

    def Ih(self, r, t, a, hh):
        """
        Fractional integral of order 𝑟 ∈ (0, 1] of a function, equation 4.3 in El Euch O, Rosenbaum M (2019)

        Args:
            t: time
            funcA: kernal function

        """
        grid = np.arange(0, t, self.delta)
        Ihrs = 0 + 0j
        for s in np.arange(0, t, self.delta):
            Ihrs += pow(t - s, r - 1) * hh[int(s//self.delta)] * self.delta
        return Ihrs / spsp.gamma(r)

    def logp_mgf_adam(self, uu, texp):
        """
        Log price mgf calculation under the rough Heston model, equation 4.5 in El Euch O, Rosenbaum M (2019)

        Args:
            uu: dummy variable
            texp: time to expire

        Returns:
            Log price MGF

        We use the characteristic function in Eq (4.5) of El Euch, O., Rosenbaum, M.: (2019) The characteristic function of rough Heston models

        References:
            - El Euch, O., Rosenbaum, M.: (2019) The characteristic function of rough Heston models https://doi.org/10.1111/mafi.12173
        """
        delta = 1/100
        self.delta = delta

        LL = uu
        for i in range(0, len(uu)):
            k = int(texp/delta)
            h_hat = np.zeros(int(k + 1), dtype='complex')
            h_hat_p = np.zeros(int(k + 1), dtype='complex')
            F_a_h_hat = np.zeros(int(k + 1), dtype='complex')
            F_a_h_hat[0] = self.F(uu[i], 0)
            for j in range(1, k + 1):
                F_a_h_hat[j] = self.F(uu[i], h_hat[j-1])
                h_hat_p[j] = np.dot(self.b_kp1(j-1, delta), F_a_h_hat[0:j])
                h_hat[j] = np.dot(self.a_kp1(j-1, delta), F_a_h_hat[0:j]) + self.a_j_kp1(j, delta, j + 1) * self.F(uu[i], h_hat_p[j])
            LL[i] = self.theta * self.mr * self.Ih(1, texp, uu[i], h_hat) + self.sigma * self.Ih(1 - self.alpha, texp, uu[i], h_hat)

        return np.exp(LL)

    '''
    Method 2: Fast Hybrid Method
    '''

    def a(self, r_0, lambd, mu, nu):
        '''
        'a' sequence in (19)
        '''
        a = np.zeros(r_0 + 1, dtype='complex')
        a_2 = np.zeros(r_0 + 1, dtype='complex')
        a[1] = nu / spsp.gamma(self.alpha + 1)
        a_2[1] = 0
        for i in range(2, r_0 + 1):
            tmp = spsp.gammaln(self.alpha * (i - 1) + 1) - spsp.gammaln(self.alpha * (i - 1) + self.alpha + 1)
            a[i] = (lambd * a_2[i-1] + mu * a[i-1]) * np.exp(tmp)
            a_2[i] = np.sum(a[:i] * a[i-1::-1])
        return a

    def I_1_FPS(self, t, lambd, mu, nu, r_0, aSequence):
        '''
        Power Series Expansion of fractional integral in (26)
        '''
        I_1 = 0
        for i in range(1, r_0 + 1):
            I_1 += aSequence[i] * pow(t, self.alpha * i) / (self.alpha * i + 1)
        return t * I_1

    def I_2_FPS(self, t, lambd, mu, nu, r_0, aSequence):
        '''
        Power Series Expansion of fractional integral in (27)
        '''
        I_2 = 0
        for i in range(2, r_0 + 1):
            tmp = spsp.gammaln(self.alpha * i) - spsp.gammaln(self.alpha * i - self.alpha)
            I_2 += aSequence[i] * pow(t, i * self.alpha) * np.exp(tmp) / ((1 - 1/i) * (self.alpha * i + 1 - self.alpha))
        return pow(t, 1 - self.alpha) * (nu * pow(t, self.alpha) + I_2)

    def phi_t(self, t, aSequence):
        '''
        The approximation of the solution of the fraction Riccati function using power series expansion,
        which is equation (25) in the reference paper
        '''
        phi_t = 0
        for i in range(1, self.r_0 + 1):
            phi_t += aSequence[i] * pow(t, self.alpha * i)
        return phi_t

    def phi_n(self, k, lambd, mu, nu, aSequence):
        '''
        The approximation of the solution of the fraction Riccati function using Euler discretization,
        which is equation (28) in the reference paper
        '''
        phi_n = np.zeros(k+1, dtype='complex')
        for i in range(1, int(self.k_0)+1):
            phi_n[i] = self.phi_t(self.t[i], aSequence)
        for i in range(int(self.k_0)+1, k+1):
            phi_n_j = 0
            for j in range(1, i):
                phi_n_j += self.c(i-j-1) * phi_n[j] * (lambd * phi_n[j] + mu)
            phi_n[i] = pow(self.texp/self.n, self.alpha) * (nu * i ** self.alpha + phi_n_j) / spsp.gamma(self.alpha + 1)
        return phi_n

    def c(self, l):
        return pow(l + 1, self.alpha) - pow(l, self.alpha)

    def c_alpha(self, l):
        return pow(l + 1, 1 - self.alpha) - pow(l, 1 - self.alpha)

    def I_1_PED(self, k, I_1_PED_tk0, lambd, mu, nu, phiSequence):
        '''
        The approximation of the regular derivative I(alpha)(ψ) using Euler discretization,
        which is equation (28) in the reference paper
        '''
        temp = 0
        for i in range(int(self.k_0), k):
            temp += phiSequence[i]
        return I_1_PED_tk0 + temp * self.texp / self.n + (phiSequence[k] - phiSequence[self.k_0]) * self.texp / (2 * self.n)

    def I_2_PED(self, k, lambd, mu, nu, phiSequence):
        '''
        The approximation of the fractional antiderivative I(1−alpha)(ψ) using Euler discretization,
        which is equation (28) in the reference paper
        '''
        temp = 0
        for i in range(1, k):
            temp += pow(self.c_alpha(k-i-1), 1 - self.alpha) * phiSequence[i]
        return pow(self.texp/self.n, 1 - self.alpha) * temp / spsp.gamma(2 - self.alpha)

    def logp_mgf_hybrid(self, uu, texp):
        """
        Log price MGF under the rough Heston model.
        The moment-generating function in Eq (2) in the reference paper, which is from Eq (4.5) of
        El Euch, O., Rosenbaum, M.: (2019) The characteristic function of rough Heston models was employed

        References:
            - Giorgia Callegaro, Martino Grasselli, Gilles Pagès (2020) Fast Hybrid Schemes for Fractional Riccati Equations (Rough Is Not So Tough). Mathematics of Operations Research 46(1):221-254.https://doi.org/10.1287/moor.2020.1054
            - El Euch, O., Rosenbaum, M.: (2019) The characteristic function of rough Heston models https://doi.org/10.1111/mafi.12173
        """
        mr = self.mr
        theta = self.theta
        vov = self.vov
        rho = self.rho
        alpha = self.alpha
        r_0 = 250
        n = 128
        self.r_0 = r_0
        self.n = n
        self.texp = texp

        lambd = pow((vov * mr), 2) / 2
        LL = uu
        thre = 0.9
        convRadius = 0.22
        k_0 = int((n * thre * convRadius) // texp)
        self.k_0 = k_0

        if texp <= thre * convRadius:
            '''
            if texp is smaller than a threshold times convergence radius, we use Power Series Expansion to solve the
            fractional Riccati function and obtain the moment-generating function
            '''
            for i in range(0, len(uu)):
                u = uu[i]
                nu = (u ** 2 - u) / 2
                mu = mr * (u * rho * vov - 1)
                aSequence = self.a(r_0, lambd, mu, nu)
                LL[i] = np.exp(theta * mr * self.I_1_FPS(texp, lambd, mu, nu, r_0, aSequence) + self.sigma * self.I_2_FPS(texp, lambd, mu, nu, r_0, aSequence))
        else:
            '''
            if texp is larger than a threshold times convergence radius, we use Euler discretization to solve the
            fractional Riccati function and obtain the moment-generating function
            '''
            t = np.linspace(0, texp, n + 1)
            self.t = t
            for i in range(0, len(uu)):
                u = uu[i]
                nu = (u ** 2 - u) / 2
                mu = mr * (u * rho * vov - 1)
                aSequence = self.a(r_0, lambd, mu, nu)
                phiSequence = self.phi_n(n, lambd, mu, nu, aSequence)
                I_1_PED_tk0 = 0
                for i in range(1, r_0 + 1):
                    I_1_PED_tk0 += pow(t[k_0], alpha * i + 1) * aSequence[i] / (alpha * i + 1)
                LL[i] = np.exp(theta * mr * self.I_1_PED(n, I_1_PED_tk0, lambd, mu, nu, phiSequence) + self.sigma * self.I_2_PED(n, lambd, mu, nu, phiSequence))

        return LL

    def logp_mgf(self, uu, texp):
        '''
        Choose method to solve the fractional Riccati equation.
        '''
        if self.method == 1:
            rv = self.logp_mgf_adam(uu, texp)
        elif self.method == 2:
            rv = self.logp_mgf_hybrid(uu, texp)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return rv
