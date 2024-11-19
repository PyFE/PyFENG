import numpy as np
import scipy.stats as spst
import scipy.special as spsp
import scipy.optimize as spopt
from .bsm import Bsm
from .opt_abc import OptABC
from .util import ChebInterp

class AmerOptExerBddABC(OptABC):

    bsm_model = None

    def __init__(self, sigma, *args, **kwargs):
        super().__init__(sigma, *args, **kwargs)
        self.bsm_model = Bsm(sigma, *args, **kwargs)

    def exer_bdd_t0(self, cp=-1):
        return np.fmin(1, self.intr/np.fmax(self.divr, np.finfo(float).tiny))


class AmerOptLi2010QdPlus(AmerOptExerBddABC):
    """
    Implementation of "initial guess" and QD+ of Li (2010)

    References:
        - Andersen L, Lake M, Offengenden D (2016) High-performance American option pricing. JCF 39–87. https://doi.org/10.21314/JCF.2016.312
        - Barone-Adesi G, Whaley RE (1987) Efficient Analytic Approximation of American Option Values. The Journal of Finance 42:301–320. https://doi.org/10.1111/j.1540-6261.1987.tb02569.x
        - Li M (2010) Analytical approximations for the critical stock prices of American options: a performance comparison. Rev Deriv Res 13:75–99. https://doi.org/10.1007/s11147-009-9044-3
    """

    def exer_bdd_ig(self, texp, cp=-1):
        """
        "Initial Guess" (IG) of exercise bdd in p.80, Eqs (7)-(8) of Li (2010).
        Normalized for strike = 1

        Args:
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            Exercise boundary (critical stock price)
        """
        s0 = self.exer_bdd_t0(cp)
        mm = 2*self.intr / self.sigma**2
        nn_m1 = 2*(self.intr - self.divr) / self.sigma**2 - 1
        q_inf = -0.5 * (nn_m1 + np.sqrt(nn_m1**2 + 4*mm))
        s_inf = 1 / (1 - 1/q_inf)

        # The sign in front of 2 sigma sqrt(t) should be (-)
        # Eq (8) in Li (2010) is wrong. See Eq. (33) in Barone-Adesi & Whaley (1987)
        theta = 1/(s_inf - 1) * ((self.intr - self.divr)*texp - 2*self.sigma*np.sqrt(texp))
        bdd = s_inf + (s0 - s_inf)*np.exp(-theta)
        return bdd

    def exer_bdd(self, texp, cp=-1):
        """
        QD+ method in p.85 of Li (2010) or Appendix A of Andersen et al. (2016)
        Normalized for strike = 1

        Args:
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            Exercise boundary (critical stock price)
        """
        root = spopt.newton(self.zero_func, x0=np.full_like(texp, self.exer_bdd_t0(), dtype=float), args=(texp, cp))
        return root

    def zero_func(self, spot_bdd, texp, cp=-1):
        """
        Function to solve for QD+ method.
        Eq. (34) of Li (2010) with c replaced with c0 or Eq. (66) of Andersen et al. (2016)
        Normalized for strike = 1

        Args:
            spot_bdd: boundary
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            function value
        """

        fwd, df, divf = self._fwd_factor(spot_bdd, texp)
        print(spot_bdd)
        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).tiny)
        d1 = np.log(fwd)/sigma_std
        d1 += 0.5*sigma_std

        nn_m1 = 2*(self.intr - self.divr) / self.sigma**2 - 1
        mm = 2*self.intr / self.sigma**2
        hh = 1.0 - df

        qqd_sqrt = np.sqrt(nn_m1**2 + 4*mm/hh)
        qqd_prime = mm / (hh**2 * qqd_sqrt)

        p_euro = self.bsm_model.price(1.0, spot_bdd, texp, cp=cp)
        theta = self.bsm_model.theta(1.0, spot_bdd, texp, cp=cp)

        qqd_c0 = mm/qqd_sqrt * (df / hh - theta / self.intr / (1.0 - spot_bdd - p_euro) - df * qqd_prime / qqd_sqrt)
        qqd_c0 -= 0.5*(nn_m1 + qqd_sqrt)

        zero = (1 - divf*spst.norm._cdf(-d1))*spot_bdd + qqd_c0*(1.0 - spot_bdd - p_euro)
        return zero

    def price(self, strike, spot, texp, cp=-1):
        """
        American call/put option price. Eq. (29) of Li (2010)

        Args:
            strike: strike price.
            spot: spot (or forward) price.
            texp: time to expiry.
            cp: 1/-1 for call/put option.

        Returns:
            option price
        """

        assert cp < 0
        fwd, df, divf = self._fwd_factor(spot, texp)
        spot_bdd = strike * self.exer_bdd(texp, cp=cp)

        if np.all(spot <= spot_bdd):
            return strike - spot

        p_euro = self.bsm_model.price(strike, spot_bdd, texp, cp=cp)
        theta = self.bsm_model.theta(strike, spot_bdd, texp, cp=cp)

        nn_m1 = 2*(self.intr - self.divr) / self.sigma**2 - 1
        mm = 2*self.intr / self.sigma**2
        hh = 1.0 - df

        qqd_sqrt = np.sqrt(nn_m1**2 + 4*mm/hh)
        qqd_prime = mm / (hh**2 * qqd_sqrt)
        qqd = -0.5*(nn_m1 + qqd_sqrt)
        c0 = mm/qqd_sqrt * (df/hh - theta/self.intr/(strike - spot_bdd - p_euro) - df*qqd_prime/qqd_sqrt)
        b = -df * mm * qqd_prime / (2*qqd_sqrt)

        log = np.log(spot/spot_bdd)
        p_amer = self.bsm_model.price(strike, spot, texp, cp=cp) \
                 + (strike - spot_bdd - p_euro) / (1 - log*(c0 + b*log)) * np.exp(qqd*log)
        p_amer = np.where(spot > spot_bdd, p_amer, strike - spot)

        return p_amer


class AmerOptAndersen2015Hp(AmerOptExerBddABC):
    """
    Implementation of Andersen, et al. (2016), referred to as HPAOP.

    References:
        - Andersen L, Lake M, Offengenden D (2016) High-performance American option pricing. JCF 39–87. https://doi.org/10.21314/JCF.2016.312

    """

    tmax, n_col, n_int, n_iter = 0., 0, 0, 0
    zero_func = AmerOptLi2010QdPlus.zero_func
    exer_bdd_qdp = AmerOptLi2010QdPlus.exer_bdd

    n_iter_pj = False

    ti = []
    interp = None
    interp_log = True

    def set_num_params(self, tmax=3., n_col=8, n_int=16, n_iter=5, n_iter_pj=1, interp_log=True):
        """
        Sets numberical parameters

        Args:
            tmax: max time of the exercise boundary
            n_col: # of collocation (Chebyshev interpolation) nodes
            n_int: # of integral node
            n_iter: # of fixed point iterations
            n_iter_pj: # of fixed point iterations with partial Jacobi
            interp_log: interpolate log(B(t)/B(tmax))^2

        Returns:

        """
        self.tmax = tmax
        self.n_iter = n_iter
        self.interp = ChebInterp(n_col, xlim=(0, 1))
        self.ti = tmax * (self.interp.x[1:])**2  # ti is increasing: ti[0] > 0, ti[-1] = texp
        self.bdd_t0 = self.exer_bdd_t0(cp=-1)
        self.bdd_ti = np.full_like(self.ti, self.bdd_t0)

        self.interp_log = interp_log
        self.n_iter_pj = n_iter_pj

        self.n_int = n_int
        zz, ww = spsp.roots_legendre(n_int)
        zz[:] = (1.0 + zz)/2; ww /= 2
        uu = (1 - zz**2)
        self.uu = uu; self.zz = zz; self.ww = ww

        self.u_ki = self.ti * self.uu[:, None]

    def interp_fit(self):
        if self.interp_log:
            self.interp.fit(np.concatenate([[0.0], np.log(self.bdd_ti / self.bdd_t0)**2]))
            #self.interp.fit(np.concatenate([[0.0], np.log(self.bdd_ti / self.bdd_t0)]))
        else:
            self.interp.fit(np.concatenate([[self.bdd_t0], self.bdd_ti]))

    def interp_eval(self, uu):
        if self.interp_log:
            return self.bdd_t0 * np.exp(-np.sqrt(np.fmax(0.0, self.interp.eval(np.sqrt(uu / self.tmax)))))
            #return self.bdd_t0 * np.exp(self.interp.eval(np.sqrt(uu / self.tmax)))
        else:
            return self.interp.eval(np.sqrt(uu / self.tmax))

    def integrand(self, spot, texp, u, bdd, cp=-1):
        """
        The integrand for American option in Eq. (4) of HPAO for unit strike = 1

        Args:
            spot: spot price
            texp: time to expiry
            u: integral variable
            bdd: exercise boundary values at u
            cp:

        Returns:

        """

        assert np.any(u <= texp)
        assert cp < 0

        fwd, df, divf = self._fwd_factor(spot, texp - u)
        d1, d2 = self.bsm_model.d12(bdd, spot, texp - u)
        rv = self.intr * df * spst.norm._cdf(-d2) - self.divr * spot * divf * spst.norm._cdf(-d1)
        return rv


    def FPA_K123(self, cp=-1):
        """
        The integrands of K1, K2, K3 in (18)-(20)

        Args:
            cp:

        Returns:
            K1, K2, and K3
        """

        assert cp < 0
        bdd_u = self.interp_eval(self.u_ki)

        fwd, df, divf = self._fwd_factor(1.0, self.u_ki)
        d1, d2 = self.bsm_model.d12(strike=bdd_u, spot=self.bdd_ti, texp=self.ti - self.u_ki)

        ### integrand
        k1_ = spst.norm._cdf(d1) / divf
        k2_ = spst.norm._pdf(d1) / (divf*self.sigma)
        k3_ = spst.norm._pdf(d2) / (df*self.sigma)

        k1 = 2*self.ti * np.sum(self.zz[:, None] * k1_ * self.ww[:, None], axis=0)
        k2 = 2*np.sqrt(self.ti) * np.sum(k2_ * self.ww[:, None], axis=0)
        k3 = 2*np.sqrt(self.ti) * np.sum(k3_ * self.ww[:, None], axis=0)

        return k1, k2, k3

    def FPA_ND(self, prime=False, cp=-1):
        """

        Args:
            prime: include the (partial) derivatives of N and D
            cp:

        Returns:

        """

        k1, k2, k3 = self.FPA_K123()
        d1, d2 = self.bsm_model.d12(strike=1.0, spot=self.bdd_ti, texp=self.ti)
        pdf_d2 =spst.norm._pdf(d2)
        sig = self.sigma * np.sqrt(self.ti)

        N = pdf_d2/sig + self.intr * k3
        D = spst.norm._pdf(d1)/sig + spst.norm._cdf(d1) + self.divr * (k1 + k2)

        if prime:
            Np = -d2 * pdf_d2 / sig**2 / self.bdd_ti
        else:
            Np = None

        return N, D, Np

    def fit_bdd(self, cp=-1):

        self.bdd_ti = self.exer_bdd_qdp(self.ti, cp=-1)
        self.interp_fit()

        for m in range(self.n_iter):
            N, D, Np = self.FPA_ND(prime=(m < self.n_iter_pj))

            bdd_new = np.exp(-(self.intr - self.divr)*self.ti) * N / D

            if Np is None:
                self.bdd_ti = bdd_new
            else:
                print('Using Partial Jacobi', m)
                self.bdd_ti += (bdd_new - self.bdd_ti) / (1 + bdd_new / self.bdd_ti * Np / N * (bdd_new - self.bdd_ti))

            self.interp_fit()

    def price(self, strike, spot, texp, cp=-1, n_int=None):
        """

        Args:
            strike:
            spot:
            texp:
            cp:
            n_int: number of integral nodes (`p` in HPAOP)

        Returns:

        """
        assert cp < 0
        assert texp <= self.tmax

        n_int = self.n_int if n_int is None else n_int

        zz, ww = spsp.roots_legendre(n_int)
        zz[:] = (1.0 + zz)/2; ww /= 2
        u_k = texp * (1-zz**2)
        bdd_u = self.interp_eval(u_k)

        p_euro = self.bsm_model.price(strike, spot, texp, cp=cp)
        spot_strike = spot/strike
        if np.isscalar(spot_strike):
            int = 2*texp * np.sum(zz * ww * self.integrand(spot_strike, texp, u_k, bdd_u, cp=cp))
        else:
            int = 2*texp * np.sum(zz * ww * self.integrand(spot_strike[:, None], texp, u_k, bdd_u, cp=cp), axis=1)

        return p_euro + strike * int

