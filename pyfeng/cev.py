import abc

import scipy.stats as spst
import scipy.special as spsp
import numpy as np
from . import opt_abc as opt
from . import opt_smile_abc as smile
from . import sv_abc as sv
from .util import avg_exp


class CevAbc(smile.OptSmileABC, abc.ABC):
    model_type = "Cev"
    beta = 0.5

    def __init__(self, sigma, beta=0.5, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility
            beta: elasticity parameter. 0.5 by default
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)
        self.beta = beta

    def params_kw(self):
        params = super().params_kw()
        extra = {"beta": self.beta}
        return {**params, **extra}  # Py 3.9, params | extra


class Cev(opt.OptAnalyticABC, CevAbc, smile.MassZeroABC):
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

    def mass_zero(self, spot, texp, log=False):
        fwd = self.forward(spot, texp)

        betac = 1.0 - self.beta
        a = 0.5 / betac  # shape parameter of gamma
        var = (betac * self.sigma)**2 * texp * avg_exp(2*(self.intr-self.divr)*betac*texp)
        x = 0.5 * np.power(fwd, 2*betac) / var

        if log:
            log_mass = (a - 1)*np.log(x) - x - np.log(spsp.gamma(a))
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
        var_t0 = (betac*alpha)**2 * avg_exp(2*(self.intr-self.divr)*betac*texp)
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
        var = np.maximum(alpha**2 * texp, np.finfo(float).eps) * betac**2 * avg_exp(2*(intr-divr)*betac*texp)

        xx = 1.0 / var
        yy = np.power(strike/fwd, 2*betac) * xx

        # Need to clean up the case beta > 0
        if beta > 1.0:
            raise ValueError("Cannot handle beta value higher than 1.0")

        ncx2_sf = spst.ncx2.sf
        ncx2_cdf = spst.ncx2.cdf

        # Computing call and put is a bit of computtion waste, but do this for vectorization.
        price = np.where(
            cp > 0,
            fwd*ncx2_sf(yy, 2 + betac_inv, xx) - strike*ncx2_cdf(xx, betac_inv, yy),
            strike*ncx2_sf(xx, betac_inv, yy) - fwd*ncx2_cdf(yy, 2 + betac_inv, xx),
        )
        return df*price

    def delta(self, strike, spot, texp, cp=1):
        fwd, df, divf = self._fwd_factor(spot, texp)
        betac = 1.0 - self.beta
        betac_inv = 1.0/betac

        var = (betac * self.sigma)**2 * texp * avg_exp(2*(self.intr-self.divr)*betac*texp)
        xx = np.power(fwd, 2*betac) / var
        yy = np.power(strike, 2*betac) / var

        if self.beta < 1.0:
            delta = 0.5*(cp - 1) + spst.ncx2.sf(yy, 2 + betac_inv, xx)\
                    + 2*xx*betac*(spst.ncx2.pdf(yy, 4 + betac_inv, xx) - (strike/fwd)*spst.ncx2.pdf(xx, betac_inv, yy))
        else:
            delta = 0.5*(cp - 1) + spst.ncx2.sf(xx, -betac_inv, yy)\
                    - 2*xx*betac*(spst.ncx2.pdf(xx, -betac_inv, yy) - (strike/fwd)*spst.ncx2.pdf(yy, 4 - betac_inv, xx))

        delta *= df if self.is_fwd else divf
        return delta

    def cdf(self, strike, spot, texp, cp=1):
        fwd = self.forward(spot, texp)

        betac = 1.0 - self.beta
        betac_inv = 1.0/betac

        alpha = self.sigma/np.power(fwd, betac)
        var = (betac * alpha)**2 * texp * avg_exp(2*(self.intr-self.divr)*betac*texp)
        xx = 1.0/var
        yy = np.power(strike/fwd, 2*betac) * xx

        cdf = np.where(cp > 0, spst.ncx2.cdf(xx, betac_inv, yy), spst.ncx2.sf(xx, betac_inv, yy))
        return cdf

    def gamma(self, strike, spot, texp, cp=1):
        fwd, df, divf = self._fwd_factor(spot, texp)

        betac = 1.0 - self.beta
        betac_inv = 1.0/betac

        var = (betac * self.sigma)**2 * texp * avg_exp(2*(self.intr-self.divr)*betac*texp)
        xx = np.power(fwd, 2*betac) / var
        yy = np.power(strike, 2*betac) / var

        if self.beta < 1.0:
            gamma = (2 + betac_inv - xx)*spst.ncx2.pdf(yy, 4 + betac_inv, xx) \
                    + xx*spst.ncx2.pdf(yy, 6 + betac_inv, xx) \
                    + strike/fwd*(xx*spst.ncx2.pdf(xx, betac_inv, yy) - yy*spst.ncx2.pdf(xx, 2 + betac_inv, yy))
        else:
            gamma = (xx*spst.ncx2.pdf(xx, -betac_inv, yy) - yy*spst.ncx2.pdf(xx, 2 - betac_inv, yy)) \
                    + strike/fwd*((2 - betac_inv - xx)*spst.ncx2.pdf(yy, 4 - betac_inv, xx)
                                  + xx*spst.ncx2.pdf(yy, 6 - betac_inv, xx))

        gamma *= 2*(divf*betac)**2/df*xx/fwd

        if self.is_fwd:
            gamma *= (df/divf)**2

        return gamma

    def vega(self, strike, spot, texp, cp=1):
        fwd, df, divf = self._fwd_factor(spot, texp)

        betac = 1.0 - self.beta
        betac_inv = 1.0/betac

        var = (betac * self.sigma)**2 * texp * avg_exp(2*(self.intr-self.divr)*betac*texp)
        xx = np.power(fwd, 2*betac) / var
        yy = np.power(strike, 2*betac) / var

        if self.beta < 1.0:
            vega = -fwd*spst.ncx2.pdf(yy, 4 + betac_inv, xx) + strike*spst.ncx2.pdf(xx, betac_inv, yy)
        else:
            vega = fwd*spst.ncx2.pdf(xx, -betac_inv, yy) - strike*spst.ncx2.pdf(yy, 4 - betac_inv, xx)

        sigma = self.sigma*np.power(spot, -betac)
        vega *= df*2*xx/sigma
        return vega

    def theta(self, strike, spot, texp, cp=1):
        ### Need to implement this
        return self.theta_numeric(strike, spot, texp, cp=cp)


class CevMc(CevAbc):
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

        var = (betac * self.sigma)**2 * dt * avg_exp(2*(self.intr-self.divr)*betac*dt)
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
