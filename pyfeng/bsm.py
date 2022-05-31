import numpy as np
import scipy.stats as spst
import warnings

from . import opt_abc as opt
from . import norm
from . import opt_smile_abc as smile


class Bsm(opt.OptAnalyticABC):
    """
    Black-Scholes-Merton (BSM) model for option pricing.

    Underlying price is assumed to follow a geometric Brownian motion.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.Bsm(sigma=0.2, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([15.71361973,  9.69250803,  5.52948546,  2.94558338,  1.48139131])
        >>> sigma = np.array([0.2, 0.3, 0.5])[:, None]
        >>> m = pf.Bsm(sigma, intr=0.05, divr=0.1) # sigma in axis=0
        >>> m.price(np.array([90, 100, 110]), 100, 1.2, cp=np.array([-1,1,1]))
        array([[ 5.75927238,  5.52948546,  2.94558338],
               [ 9.4592961 ,  9.3881245 ,  6.45745004],
               [16.812035  , 17.10541288, 14.10354768]])
    """

    @staticmethod
    def price_formula(strike, spot, sigma, texp, cp=1, intr=0.0, divr=0.0, is_fwd=False):
        """
        Black-Scholes-Merton model call/put option pricing formula (static method)

        Args:
            strike: strike price
            spot: spot (or forward)
            sigma: model volatility
            texp: time to expiry
            cp: 1/-1 for call/put option
            sigma: model volatility
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.

        Returns:
            Vanilla option price
        """
        disc_fac = np.exp(-texp*intr)
        fwd = np.array(spot)*(1.0 if is_fwd else np.exp(-texp*divr)/disc_fac)

        sigma_std = np.maximum(np.array(sigma)*np.sqrt(texp), np.finfo(float).eps)

        # don't directly compute d1 just in case sigma_std is infty
        d1 = np.log(fwd/strike)/sigma_std
        d2 = d1 - 0.5*sigma_std
        d1 += 0.5*sigma_std

        cp = np.array(cp)
        price = fwd*spst.norm.cdf(cp*d1) - strike*spst.norm.cdf(cp*d2)
        price *= cp*disc_fac
        return price

    def vega(self, strike, spot, texp, cp=1):

        fwd, df, _ = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).eps)
        d1 = np.log(fwd/strike)/sigma_std + 0.5*sigma_std

        # formula according to wikipedia
        vega = df*fwd*spst.norm.pdf(d1)*np.sqrt(texp)
        return vega

    def d2_var(self, strike, spot, texp, cp=1):
        """
        2nd derivative w.r.t. variance (=sigma^2)
        Eq. (9) in Hull & White (1987)

        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns: d^2 price / d var^2

        References:
            - Hull J, White A (1987) The Pricing of Options on Assets with Stochastic Volatilities. The Journal of Finance 42:281–300. https://doi.org/10.1111/j.1540-6261.1987.tb02568.x
        """
        fwd, df, _ = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).eps)
        d1 = np.log(fwd/strike)/sigma_std
        d2 = d1 - 0.5*sigma_std
        d1 += 0.5*sigma_std

        risk = df*spot*np.sqrt(texp)*spst.norm.pdf(d1)*(d1*d2 - 1)/(4*self.sigma**3)

        return risk

    def d3_var(self, strike, spot, texp, cp=1):
        """
        3rd derivative w.r.t. variance (=sigma^2)
        Eq. (9) in Hull & White (1987)

        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns: d^3 price / d var^3

        References:
            - Hull J, White A (1987) The Pricing of Options on Assets with Stochastic Volatilities. The Journal of Finance 42:281–300. https://doi.org/10.1111/j.1540-6261.1987.tb02568.x
        """
        fwd, df, _ = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).eps)
        d1 = np.log(fwd/strike)/sigma_std
        d2 = d1 - 0.5*sigma_std
        d1 += 0.5*sigma_std

        risk = df*spot*np.sqrt(texp)*spst.norm.pdf(d1)\
               *((d1*d2 - 1)*(d1*d2 - 3) - (d1**2 + d2**2))/(8*self.sigma**5)

        return risk

    def delta(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).eps)
        d1 = np.log(fwd/strike)/sigma_std + 0.5*sigma_std

        delta = cp*spst.norm.cdf(cp*d1)  # formula according to wikipedia
        delta *= df if self.is_fwd else divf
        return delta

    def cdf(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).eps)
        d2 = np.log(fwd/strike)/sigma_std - 0.5*sigma_std
        cdf = spst.norm.cdf(cp*d2)  # formula according to wikipedia
        return cdf

    def gamma(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), 100*np.finfo(float).eps)
        d1 = np.log(fwd/strike)/sigma_std + 0.5*sigma_std

        gamma = (
                df*spst.norm.pdf(d1)/fwd/sigma_std
        )  # formula according to wikipedia
        if not self.is_fwd:
            gamma *= (divf/df)**2
        return gamma

    def theta(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), 100*np.finfo(float).eps)
        d1 = np.log(fwd/strike)/sigma_std
        d2 = d1 - 0.5*sigma_std
        d1 += 0.5*sigma_std

        # still not perfect; need to consider the derivative w.r.t. divr and is_fwd = True
        theta = -0.5*spst.norm.pdf(d1)*fwd*self.sigma/np.sqrt(
            texp
        ) - cp*self.intr*strike*spst.norm.cdf(cp*d2)
        theta *= df
        return theta

    def _impvol_newton(self, price, strike, spot, texp, cp=1, setval=False):
        """
        BSM implied volatility with Newton's method

        Args:
            price: option price
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put option
            setval: if True, sigma is set with the solved implied volatility

        Returns:
            implied volatility
        """
        fwd, df, divf = self._fwd_factor(spot, texp)

        strike_std = strike/fwd  # strike / fwd
        price_std = price/df/fwd  # forward price / fwd

        bsm_model = Bsm(0, is_fwd=True)
        p_min = bsm_model.price(strike_std, 1.0, texp, cp)
        bsm_model.sigma = np.inf
        p_max = bsm_model.price(strike_std, 1.0, texp, cp)
        scalar_output = np.isscalar(p_min) & np.isscalar(price_std)

        # Exclude optoin price below intrinsic value or above max value (1 for call or k for put)
        # ind_solve can be scalar or array. scalar can be fine in np.abs(p_err[ind_solve])
        ind_solve = (price_std - p_min > Bsm.IMPVOL_TOL) & (
                p_max - price_std > Bsm.IMPVOL_TOL
        )

        # initial guess = inflection point in sigma (volga=0)
        _sigma = np.ones_like(ind_solve)*np.sqrt(
            2*np.abs(np.log(strike_std))/texp
        )

        bsm_model.sigma = _sigma
        p_err = bsm_model.price(strike_std, 1.0, texp, cp) - price_std
        # print(np.sign(p_err), _sigma)

        if np.any(ind_solve):
            for k in range(32):  # usually iteration ends less than 10
                vega = bsm_model.vega(strike_std, 1.0, texp, cp)
                _sigma -= p_err/vega
                bsm_model.sigma = _sigma
                p_err = bsm_model.price(strike_std, 1.0, texp, cp) - price_std
                p_err_max = np.amax(np.abs(p_err[ind_solve]))
                # print(k, p_err_max, _sigma)

                # ignore the error of the elements with ind_solve = False
                if p_err_max < Bsm.IMPVOL_TOL:
                    break
            # print(k)

            if p_err_max >= Bsm.IMPVOL_TOL:
                warn_msg = f"impvol_newton did not converged within {k} iterations: max error = {p_err_max}"
                warnings.warn(warn_msg, Warning)

        # Put Nan for the out-of-bound option prices
        _sigma = np.where(ind_solve, _sigma, np.nan)

        # Though not error is above tolerance, if the price is close to min or max, set 0 or inf
        _sigma = np.where(
            (np.abs(p_err) >= Bsm.IMPVOL_TOL)
            & (np.abs(price_std - p_min) <= Bsm.IMPVOL_TOL),
            0,
            _sigma,
        )
        _sigma = np.where(
            (np.abs(p_err) >= Bsm.IMPVOL_TOL)
            & (np.abs(price_std - p_max) <= Bsm.IMPVOL_TOL),
            np.inf,
            _sigma,
        )

        if scalar_output:
            _sigma = np.asscalar(_sigma)

        if setval:
            self.sigma = _sigma

        return _sigma

    #### Class method
    impvol = _impvol_newton

    def vol_smile(self, strike, spot, texp, cp=1, model="norm"):
        """
        Equivalent volatility smile for a given model

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
            model: {'norm' (default), 'norm-approx', 'norm-grunspan', 'bsm'}

        Returns:
            volatility smile under the specified model
        """
        if model.lower() == "bsm":
            return self.sigma*np.ones_like(strike + spot + texp + cp)
        if model.lower() == "norm":
            price = self.price(strike, spot, texp, cp=cp)
            return norm.Norm(None).impvol(price, strike, spot, texp, cp=cp)
        elif model.lower() == "norm-approx" or model.lower() == "norm-grunspan":
            fwd, _, _ = self._fwd_factor(spot, texp)
            kk = strike/fwd
            lnk = np.log(kk)
            if model.lower() == "norm-approx":
                return (
                        self.sigma
                        *fwd
                        *np.sqrt(kk)
                        *(1 + lnk**2/24)
                        /(1 + self.sigma**2*texp/24)
                )
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    term1 = np.where(np.fabs(lnk) > 1e-8, (kk - 1)/lnk, 2/(3 - kk))
                    term2 = np.where(
                        np.fabs(lnk) > 1e-8,
                        (np.log(term1) - lnk/2)/lnk**2,
                        1/24,
                    )
                return self.sigma*fwd*term1*(1 - term2*self.sigma**2*texp)
        else:
            raise ValueError(f"Unknown model: {model}")

    def _price_suboptimal(self, strike, spot, texp, cp=1, strike2=None):
        strike2 = strike if strike2 is None else strike2
        fwd, df, _ = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma*np.sqrt(texp), np.finfo(float).eps)
        d1 = np.log(fwd/strike2)/sigma_std
        d2 = d1 - 0.5*sigma_std
        d1 += 0.5*sigma_std

        price = fwd*spst.norm.cdf(cp*d1) - strike*spst.norm.cdf(cp*d2)
        price *= cp*df
        return price

    def _barrier_params(self, barrier, spot):
        """
        Parameters used for barrier option pricing

        Args:
            barrier: barrier price
            spot: spot price

        Returns:
            barrier option pricing parameters (psi, spot_mirror)
        """
        psi = np.power(
            barrier/spot, 2*(self.intr - self.divr)/self.sigma**2 - 1
        )
        spot_reflected = barrier**2/spot
        return psi, spot_reflected

    def price_barrier(self, strike, barrier, spot, texp, cp=1, io=-1):
        """
        Barrier option price under the BSM model

        Args:
            strike: strike price
            barrier: knock-in/out barrier price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
            io: +1 for knock-in, -1 for knock-out

        Returns:
            Barrier option price
        """

        psi, spot_reflected = self._barrier_params(barrier, spot)

        """
        `mirror_sign` is +1/-1 if call/put remains same/flipped in the reflection principle
        +1 if (barrier < spot AND call) or (barrier > spot AND put), -1 otherwise
        """
        mirror_sign = np.where(barrier < spot, 1, -1)*cp
        """
        This is a trick to handle a trivial case:
        Knock-out call with spot < strike < barrier is worth zero.
        Knock-out put with barrier < strike < spot is worth zero.
        Without explicit adjustment, Knock-out price is negative, Knock-in price is higher than vanilla.
        In both scenario (mirror_sign = -1), we set strike = barrier, which will do the adjustment.
        """
        barrier = np.where(
            mirror_sign > 0, barrier, cp*np.maximum(cp*strike, cp*barrier)
        )

        p_euro1 = np.where(
            mirror_sign > 0,
            0,
            self._price_suboptimal(strike, spot, texp, cp=cp, strike2=barrier),
        )

        p_euro2 = self._price_suboptimal(
            strike, spot_reflected, texp, cp=mirror_sign*cp
        )
        p_euro2 -= np.where(
            mirror_sign > 0,
            0,
            self._price_suboptimal(
                strike, spot_reflected, texp, cp=mirror_sign*cp, strike2=barrier
            ),
        )

        p = p_euro1 + psi*p_euro2  # knock-in price
        p = np.where(
            io > 0,
            p,  # knock-in type
            self._price_suboptimal(strike, spot, texp, cp=cp) - p,
        )

        return p

    def moments_vsk(self, texp=1):
        """
        Variance, skewness, and ex-kurtosis. Assume mean=1.

        Args:
            texp: time-to-expiry

        Returns:
            (variance, skewness, and ex-kurtosis)

        References:
            https://en.wikipedia.org/wiki/Log-normal_distribution
        """
        ww = np.exp(texp*self.sigma**2)
        var = ww - 1
        skew = (ww + 2)*np.sqrt(ww - 1)
        exkurt = ww**2*(ww*(ww + 2) + 3) - 6  # ww**4 + 2*ww**3 + 3*ww - 6
        return var, skew, exkurt


class BsmDisp(smile.OptSmileABC, Bsm):
    """
    Displaced Black-Scholes-Merton model for option pricing. Displace price,

        D(F_t) = beta*F_t + (1-beta)*A

    is assumed to follow a geometric Brownian motion with volatility `beta*sigma`.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmDisp(sigma=0.2, beta=0.5, pivot=100, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([15.9543935 ,  9.7886658 ,  5.4274197 ,  2.71430505,  1.22740381])
        >>> beta = np.array([0.2, 0.6, 1])[:, None]
        >>> m = pf.BsmDisp(0.2, beta=beta, pivot=100) # beta in axis=0
        >>> m.vol_smile(np.arange(80, 121, 10), 100, 1.2)
        array([[0.21915778, 0.20904587, 0.20038559, 0.19286293, 0.18625174],
               [0.20977955, 0.20461468, 0.20025691, 0.19652101, 0.19327567],
               [0.2       , 0.2       , 0.2       , 0.2       , 0.2       ]])
    """

    beta = 1  # equivalent to Black-Scholes
    pivot = 0
    sigma_disp = None

    # _m_bsm = None

    def __init__(self, sigma, beta=1, pivot=0, *args, **kwargs):
        """
        Args:
            sigma: model volatility
            beta: beta. 1 by default
            pivot: A. 0 by default
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        self.pivot = pivot
        self.beta = beta
        # self._m_bsm = Bsm(sigma=beta*sigma, *args, **kwargs)
        super().__init__(sigma, *args, **kwargs)

    @property
    def sigma(self):
        return self.sigma_disp*self.beta

    @sigma.setter
    def sigma(self, sigma):
        self.sigma_disp = sigma

    def disp_spot(self, spot):
        """
        Displaced spot

        Args:
            spot: spot (or forward) price

        Returns:
            Displaces spot
        """
        return self.beta*spot + (1 - self.beta)*self.pivot

    def disp_strike(self, strike, texp):
        """
        Displaced strike

        Args:
            strike: strike price
            texp: time to expiry

        Returns:
            Displace strike
        """
        return self.beta*strike + (1 - self.beta)*self.forward(self.pivot, texp)

    def price(self, strike, spot, texp, cp=1):
        spot = self.disp_spot(spot)
        strike = self.disp_strike(strike, texp)
        return (1/self.beta)*super().price(strike, spot, texp, cp=cp)

    def delta(self, strike, spot, texp, cp=1):
        spot = self.disp_spot(spot)
        strike = self.disp_strike(strike, texp)
        return super().delta(strike, spot, texp, cp=cp)

    def cdf(self, strike, spot, texp, cp=1):
        spot = self.disp_spot(spot)
        strike = self.disp_strike(strike, texp)
        return super().cdf(strike, spot, texp, cp=cp)

    def vega(self, strike, spot, texp, cp=1):
        spot = self.disp_spot(spot)
        strike = self.disp_strike(strike, texp)
        return super().vega(strike, spot, texp, cp=cp)

    def gamma(self, strike, spot, texp, cp=1):
        # need to mutiply beta because of (beta*sigma) appearing in the denominator of the bsm gamma
        spot = self.disp_spot(spot)
        strike = self.disp_strike(strike, texp)
        return self.beta*super().gamma(strike, spot, texp, cp=cp)

    def theta(self, strike, spot, texp, cp=1):
        spot = self.disp_spot(spot)
        strike = self.disp_strike(strike, texp)
        return (1/self.beta)*super().theta(strike, spot, texp, cp=cp)

    def impvol(self, price_in, strike, spot, texp, cp=1, setval=False):
        spot = self.disp_spot(spot)
        strike = self.disp_strike(strike, texp)
        sigma = (1/self.beta)*super().impvol(
            self.beta*price_in, strike, spot, texp, cp=cp, setval=False
        )
        if setval:
            self.sigma = sigma
        return sigma

    def vol_smile(self, strike, spot, texp, cp=1, model="bsm"):
        """
        Equivalent volatility smile for a given model

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put option
            model: {'bsm', 'norm', 'bsm-approx', 'norm-approx'}

        Returns:
            volatility smile under the specified model
        """
        if model.lower() == "norm-approx":
            fwdd = self.forward(self.disp_spot(spot), texp)
            kkd = self.disp_strike(strike, texp)/fwdd
            lnkd = np.log(kkd)
            # self.sigma actually means self.beta * self._sigma
            vol = (
                    self.sigma_disp
                    *fwdd
                    *np.sqrt(kkd)
                    *(1 + lnkd**2/24)
                    /(1 + self.sigma**2*texp/24)
            )
        elif model.lower() == "bsm-approx":
            fwd = self.forward(spot, texp)
            kk = strike/fwd
            lnk = np.log(kk)

            fwdd = self.forward(self.disp_spot(spot), texp)
            kkd = self.disp_strike(strike, texp)/fwdd
            lnkd = np.log(kkd)

            # self.sigma actually means self.beta * self.sigma_disp
            vol = self.sigma_disp*(fwdd/fwd)*np.sqrt(kkd/kk)
            vol *= (
                    (1 + lnkd**2/24)
                    /(1 + lnk**2/24)
                    *(1 + vol**2*texp/24)
                    /(1 + self.sigma**2*texp/24)
            )
        else:
            vol = super().vol_smile(strike, spot, texp, model=model, cp=cp)

        return vol

    def price_barrier(self, strike, barrier, spot, *args, **kwargs):
        return (1/self.beta)*self.bsm_model.price_barrier(
            self.disp(strike), self.disp(barrier), self.disp(spot), *args, **kwargs
        )

    def moments_vsk(self, texp=1):
        rv = super().moments_vsk(self, texp)
        rv[0] /= self.beta
        rv[1] /= self.beta**3
        rv[2] /= self.beta**4
