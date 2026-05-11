import scipy.stats as spst
import scipy.special as spsp
import numpy as np
from .opt_abc import OptABC, OptAnalyticABC, MassZeroABC
from . import sv_abc as sv
from .params import CevParams
from .util import MathFuncs


class Cev(CevParams, OptAnalyticABC, MassZeroABC):
    """
    Constant Elasticity of Variance (CEV) model.

    Underlying price is assumed to follow CEV process:
    dS_t = (r - q) S_t dt + sigma S_t^beta dW_t, where dW_t is a standard Brownian motion.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.Cev(sigma=2, beta=0.5, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), spot=100, texp=1.2)
        array([16.0290539 ,  9.89374055,  5.53356595,  2.79581565,  1.27662838])
    """

    @staticmethod
    def price_std(sigma, k, beta=0.5, sign=1, type=1):
        """
        Standardized CEV option price / price-to-vega / vega (F₀ = 1, T = 1, no rates).

        Effective variance: V = β*²σ², centrality: z₀ = 1/V, z_K = k^{2β*}·z₀,
        degrees of freedom: 2ν = 1/β* (β < 1) or −1/β* (β > 1), β* = 1 − β.

        type = −1: vega  𝒱 = (2/|σβ*|) · f_χ²(z_K; 2+2ν, z₀)
        type =  1: price-to-vega  (default)
            sign = +1:   C / 𝒱 = (|σβ*|/2) · [Q(z_K; 2+2ν, z₀) − k·P(z₀; 2ν, z_K)] / f
            sign = −1:  (F₀−C)/𝒱 = (|σβ*|/2) · [P(z_K; 2+2ν, z₀) + k·P(z₀; 2ν, z_K)] / f
        type =  0: price  C = Q(z_K; 2+2ν, z₀) − k·P(z₀; 2ν, z_K)  (Schroder 1989)

        For β > 1 the z₀ / z_K roles swap and the NCX2 degrees of freedom change sign.

        Args:
            sigma: CEV volatility
            k: strike ratio K/F₀ (not log-strike)
            beta: elasticity parameter (default 0.5); must satisfy β ≠ 1
            sign: +1 for C/𝒱 (default); −1 for (F₀−C)/𝒱
            type: 0 for price, 1 for price-to-vega (default), −1 for vega

        Returns:
            Standardized price, price-to-vega, or vega

        References:
            Choi J, Shim S (2026) New option analytics on the CEV model. Unpublished note.
        """
        betac = 1.0 - beta
        betac_inv = 1.0 / betac

        var = betac**2 * sigma**2
        z0 = 1.0 / var
        zK = np.power(k, 2 * betac) * z0

        if beta < 1.0:
            f1 = spst.ncx2.pdf(zK, 2 + betac_inv, z0)
            if type == -1:
                return 2.0 / (sigma * betac) * f1
            if sign > 0:
                rv_num = spst.ncx2.sf(zK, 2 + betac_inv, z0) - k * spst.ncx2.cdf(z0, betac_inv, zK)
            else:
                rv_num = spst.ncx2.cdf(zK, 2 + betac_inv, z0) + k * spst.ncx2.cdf(z0, betac_inv, zK)
            coeff = sigma * betac / 2
        else:
            f1 = spst.ncx2.pdf(z0, 2 - betac_inv, zK)
            if type == -1:
                return -2.0 / (sigma * betac) * f1
            if sign > 0:
                rv_num = spst.ncx2.sf(z0, -betac_inv, zK) - k * spst.ncx2.cdf(zK, 2 - betac_inv, z0)
            else:
                rv_num = spst.ncx2.cdf(z0, -betac_inv, zK) + k * spst.ncx2.cdf(zK, 2 - betac_inv, z0)
            coeff = -sigma * betac / 2  # > 0 since betac < 0

        if type == 0:
            return rv_num
        return rv_num * coeff / f1

    def mass_zero(self, spot, texp, log=False):
        fwd = self.forward(spot, texp)

        betac = 1.0 - self.beta
        a = 0.5 / betac  # shape parameter of gamma distribution
        alpha = self.sigma / np.power(fwd, betac)
        var = (betac * alpha)**2 * texp * MathFuncs.avg_exp(2*(self.intr-self.divr)*betac*texp)
        x = 0.5 / var  # = z_0 / 2

        if log:
            log_mass = (a - 1)*np.log(x) - x - spsp.loggamma(a)
            log_mass += np.log1p((a - 1)/x*(1 + (a - 2)/x*(1 + (a - 3)/x*(1 + (a - 4)/x))))
            with np.errstate(divide="ignore"):
                log_mass = np.where(x > 100, log_mass, np.log(spst.gamma.sf(x=x, a=a)))
            return log_mass
        else:
            return spst.gamma.sf(x=x, a=a)

    def mass_zero_t0(self, spot, texp):
        """
        Limit value of -T log(M_T) as T -> 0, where M_T is the mass at zero.

        Args:
            spot: spot (or forward) price

        Returns:
            - lim_{T->0} T log(M_T)
        """
        fwd = self.forward(spot, texp)
        betac = 1.0 - self.beta
        alpha = self.sigma/np.power(fwd, betac)
        var_t0 = (betac*alpha)**2 * MathFuncs.avg_exp(2*(self.intr-self.divr)*betac*texp)
        t0 = 0.5/var_t0

        return t0

    @staticmethod
    def price_formula(strike, spot, texp, sigma=None, cp=1, beta=0.5, intr=0.0, divr=0.0, is_fwd=False):
        """
        CEV model call/put option pricing formula (static method)

        Args:
            strike: strike price
            spot: spot (or forward)
            sigma: model volatility
            texp: time to expiry
            cp: 1/-1 for call/put option
            beta: elasticity parameter
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.

        Returns:
            Vanilla option price
        """

        df = np.exp(-texp*intr)
        fwd = spot * (1.0 if is_fwd else np.exp(-texp*divr)/df)

        betac = 1.0 - beta
        betac_inv = 1.0/betac
        alpha = sigma/np.power(fwd, betac)
        var = np.maximum(alpha**2 * texp, np.finfo(float).eps) * betac**2 * MathFuncs.avg_exp(2*(intr-divr)*betac*texp)

        z0 = 1.0 / var                              # z_0
        zK = np.power(strike/fwd, 2*betac) * z0    # z_K = (K/F_0)^{2β*} z_0

        sf = spst.ncx2.sf
        cdf = spst.ncx2.cdf

        # Theorem 5 (Schroder 1989).  ν = 1/(2|β*|), so 2ν = |betac_inv|.
        # β < 1: 2ν = betac_inv,  2+2ν = 2+betac_inv;  (z_0, z_K) = (z0, zK)
        # β > 1: 2ν = -betac_inv, 2+2ν = 2-betac_inv;  roles swap: (z_0, z_K) → (zK, z0)
        if beta < 1.0:
            price = np.where(
                cp > 0,
                fwd*sf(zK, 2 + betac_inv, z0) - strike*cdf(z0, betac_inv, zK),
                strike*sf(z0, betac_inv, zK) - fwd*cdf(zK, 2 + betac_inv, z0),
            )
        else:
            price = np.where(
                cp > 0,
                fwd*sf(z0, -betac_inv, zK) - strike*cdf(zK, 2 - betac_inv, z0),
                strike*sf(zK, 2 - betac_inv, z0) - fwd*cdf(z0, -betac_inv, zK),
            )
        return df*price

    def delta(self, strike, spot, texp, cp=1):
        """
        CEV option delta (∂C/∂F_0).

        Simplified single-term expressions derived by differentiating the equivalent
        price form (Theorem 6); the exchange lemma F_0 f(z_K;2+2ν,z_0) = K f(z_0;2+2ν,z_K)
        cancels all boundary terms.

        References:
            - Choi J, Shim S (2026) New option analytics on the CEV model. Unpublished note.
        """
        fwd, df, divf = self._fwd_factor(spot, texp)
        betac = 1.0 - self.beta
        betac_inv = 1.0/betac
        alpha = self.sigma / np.power(fwd, betac)
        var = (betac * alpha)**2 * texp * MathFuncs.avg_exp(2*(self.intr-self.divr)*betac*texp)
        z0 = 1.0 / var                              # z_0  (F_0-normalised)
        zK = np.power(strike/fwd, 2*betac) * z0    # z_K = (K/F_0)^{2β*} z_0

        if self.beta < 1.0:
            delta = 0.5*(cp - 1) + spst.ncx2.sf(zK, betac_inv, z0)
        else:
            delta = 0.5*(cp - 1) + spst.ncx2.sf(z0, 2 - betac_inv, zK)

        delta *= df if self.is_fwd else divf
        return delta

    def cdf(self, strike, spot, texp, cp=1):
        fwd = self.forward(spot, texp)

        betac = 1.0 - self.beta
        betac_inv = 1.0/betac

        alpha = self.sigma/np.power(fwd, betac)
        var = (betac * alpha)**2 * texp * MathFuncs.avg_exp(2*(self.intr-self.divr)*betac*texp)
        z0 = 1.0/var
        zK = np.power(strike/fwd, 2*betac) * z0

        cdf = np.where(cp > 0, spst.ncx2.cdf(z0, betac_inv, zK), spst.ncx2.sf(z0, betac_inv, zK))
        return cdf

    def gamma(self, strike, spot, texp, cp=1):
        """
        CEV option gamma (∂²C/∂F_0²).

        Simplified single-term expression Γ = (2|β*| z_0/F_0) f_χ²(·; 2+2ν, ·),
        obtained by differentiating the single-term delta.

        References:
            - Choi J, Shim S (2026) New option analytics on the CEV model. Unpublished note.
        """
        fwd, df, divf = self._fwd_factor(spot, texp)

        betac = 1.0 - self.beta
        betac_inv = 1.0/betac
        alpha = self.sigma / np.power(fwd, betac)
        var = (betac * alpha)**2 * texp * MathFuncs.avg_exp(2*(self.intr-self.divr)*betac*texp)
        z0 = 1.0 / var                              # z_0  (F_0-normalised)
        zK = np.power(strike/fwd, 2*betac) * z0    # z_K = (K/F_0)^{2β*} z_0

        # In the F_0=1 frame: 2|β*| z0 · f_χ²; rescale back by 1/fwd.
        # (divf²/df) converts forward-gamma to spot-gamma.
        if self.beta < 1.0:
            gamma = 2*betac * z0 * spst.ncx2.pdf(zK, 2 + betac_inv, z0)
        else:
            gamma = -2*betac * z0 * spst.ncx2.pdf(z0, 2 - betac_inv, zK)

        gamma *= divf**2 / (df * fwd)   # rescale 1/fwd, then fwd→spot conversion

        if self.is_fwd:
            gamma *= (df/divf)**2

        return gamma

    def vega(self, strike, spot, texp, cp=1):
        """
        CEV option vega (∂C/∂σ) with respect to the CEV parameter σ in dF = σ F^β dW.

        Simplified single-term expression V = (2F_0/σβ*) f_χ²(·; 2+2ν, ·), derived via
        the exchange lemma (cancels the z_K bracket) and Theorem 2 (collapses the
        remaining two-term sum into one).  In the F_0=1 frame the prefactor is 2/(αβ*);
        rescaling to actual F_0 introduces the factor F_0^β (since σ = α·F_0^{β*}).

        References:
            - Choi J, Shim S (2026) New option analytics on the CEV model. Unpublished note.
        """
        fwd, df, divf = self._fwd_factor(spot, texp)

        betac = 1.0 - self.beta
        betac_inv = 1.0/betac
        alpha = self.sigma / np.power(fwd, betac)
        var = (betac * alpha)**2 * texp * MathFuncs.avg_exp(2*(self.intr-self.divr)*betac*texp)
        z0 = 1.0 / var                              # z_0  (F_0-normalised)
        zK = np.power(strike/fwd, 2*betac) * z0    # z_K = (K/F_0)^{2β*} z_0

        if self.beta < 1.0:
            vega = 2 * np.power(fwd, self.beta) / (alpha * betac) * spst.ncx2.pdf(zK, 2 + betac_inv, z0)
        else:
            vega = -2 * np.power(fwd, self.beta) / (alpha * betac) * spst.ncx2.pdf(z0, 2 - betac_inv, zK)

        return df * vega

    def theta(self, strike, spot, texp, cp=1):
        """
        CEV option theta (∂C/∂t, calendar-time convention; negative for long calls).

        Direct derivation.  The discounted call price C(F, T) depends on σ and T
        only through the effective variance V(T) = β*²σ²T·avg_exp(ρT), ρ = 2β*(r-q).
        Its time derivative is

            dV/dT = β*²σ² e^{ρT},   so   dV/dT · σ/(2V) = σ e^{ρT} / (2T·avg_exp(ρT)).

        The chain rule ∂C/∂σ = (2V/σ) ∂C/∂V therefore gives

            ∂C/∂T = (r-q)·S·Δ − r·C + V′(T)·σ/(2V)·v,

        where Δ = ∂C/∂S is the spot delta (zero for is_fwd=True) and v = ∂C/∂σ is the
        CEV vega.  Calendar theta Θ = ∂C/∂t = −∂C/∂T:

            Θ = r·C − (r-q)·S·Δ − σ e^{ρT} / (2T·avg_exp(ρT)) · v.

        References:
            - Choi J, Shim S (2026) New option analytics on the CEV model. Unpublished note.
        """
        betac = 1.0 - self.beta
        rho_T = 2 * betac * (self.intr - self.divr) * texp
        sigma_coeff = self.sigma * np.exp(rho_T) / (2 * texp * MathFuncs.avg_exp(rho_T))

        price = self.price(strike, spot, texp, cp)
        vega  = self.vega(strike, spot, texp, cp)
        drift = 0.0 if self.is_fwd else (self.intr - self.divr) * spot * self.delta(strike, spot, texp, cp)

        theta = drift + sigma_coeff * vega - self.intr * price
        return -theta


class CevMc(CevParams, OptABC):
    """
    Constant Elasticity of Variance (CEV) model with exact Monte-Carlo method (Kang, 2014)

    Underlying price is assumed to follow CEV process:
    dS_t = (r - q) S_t dt + sigma S_t^beta dW_t, where dW_t is a standard Brownian motion.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.CevMc(sigma=0.2*10, beta=0.5, intr=0.05, divr=0.1)
        >>> m.set_num_params(n_path=160000, dt=None, rn_seed=123456)
        >>> m.price(np.arange(80, 121, 10), spot=100, texp=1.2)
        array([16.03769545,  9.88906886,  5.51705317,  2.78576033,  1.27297844])

        References:
            - Kang C (2014) Simulation of the Shifted Poisson Distribution with an Application to the CEV Model. Management Science and Financial Engineering 20:27–32. https://doi.org/10.7737/MSFE.2014.20.1.027
    """

    dt = 0.05
    n_path = 10000
    rn_seed = None
    rng = np.random.default_rng(None)
    correct_fwd = False

    tobs = sv.CondMcBsmABC.tobs

    def set_num_params(self, n_path=10000, dt=0.25, rn_seed=None):
        """
        Set MC parameters

        Args:
            n_path: number of paths
            dt: time step for Euler/Milstein steps
            rn_seed: random number seed
        """
        self.n_path = int(n_path)
        self.dt = dt
        self.rn_seed = rn_seed
        self.rng = np.random.default_rng(rn_seed)

    def price_step(self, spot, dt):
        """
        Simulated asset price after dt given the current price `spot` using the Algorithm 3 of Kang (2014)

        Args:
            spot: current price
            dt: time step

        Returns:
            Simulated price after `dt`

        References:
            - Kang C (2014) Simulation of the Shifted Poisson Distribution with an Application to the CEV Model. Management Science and Financial Engineering 20:27–32. https://doi.org/10.7737/MSFE.2014.20.1.027
        """

        nz_idx = (spot > 0.0)
        s_t = np.exp((self.intr - self.divr)*dt) * spot
        betac = 1.0 - self.beta

        var = (betac * self.sigma)**2 * dt * MathFuncs.avg_exp(2*(self.intr-self.divr)*betac*dt)
        z0 = np.power(s_t[nz_idx], 2*betac) / var
        rv_gam = 2 * self.rng.standard_gamma(1/(2*betac), size=len(z0))
        pois_lam = (z0 - rv_gam) / 2

        # index for negative poisson value so that s_t becomes zero in this step
        neg_idx = (pois_lam <= 0)

        if np.any(neg_idx):
            pois_lam = pois_lam[~neg_idx]  # positive only
            tmp_idx = np.where(nz_idx)[0][neg_idx]
            nz_idx[tmp_idx] = False
            s_t[~nz_idx] = 0.0

        zt = 2 * self.rng.standard_gamma(self.rng.poisson(pois_lam) + 1)
        s_t[nz_idx] = np.power(var * zt, 1/(2*betac))
        return s_t

    def mass_zero(self, spot, texp, log=False):

        tobs = self.tobs(texp)
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)
        s_t = np.full(self.n_path, spot)

        for i in range(n_dt):
            s_t = self.price_step(s_t, dt[i])

        zm = (s_t <= 0).mean()
        if log:
            zm = np.log(zm)

        return zm

    def price(self, strike, spot, texp, cp=1):

        tobs = self.tobs(texp)
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        s_t = np.full(self.n_path, spot)

        for i in range(n_dt):
            s_t = self.price_step(s_t, dt[i])

        df = np.exp(-self.intr * texp)
        p = np.mean(np.fmax(cp*(s_t - strike[:, None]), 0), axis=1)

        return df * p
