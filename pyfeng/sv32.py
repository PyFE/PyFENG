import numpy as np
import scipy.special as spsp
from .opt_abc import OptABC
from .params import Sv32Params


class Sv32ABC(Sv32Params, OptABC):
    """
    Abstract base class for the 3/2 stochastic-volatility model.

    Provides the closed-form log-price MGF (Lewis 2000 / Carr & Sun 2007),
    which is shared by all Fourier-based 3/2 pricers.
    """

    expo_max = np.log(np.finfo(np.float32).max)

    @staticmethod
    def hyp1f1_complex(a, b, x):
        """
        Confluent hypergeometric function 1F1 (scipy.special.hyp1f1) taking complex values of a and b

        Args:
            a: parameter (real or complex)
            b: parameter (real or complex)
            x: argument (real or complex)

        Returns:
            function value

        References:
            - https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html#scipy.special.hyp1f1
        """
        inc = a / b * x
        ret = 1 + inc

        for kk in np.arange(1, 1024):
            inc *= (a + kk) / (b + kk) / (kk + 1) * x
            ret += inc

        return ret

    def logp_mgf(self, uu, texp):
        """
        Log price MGF under the 3/2 SV model from Lewis (2000) or Carr & Sun (2007).

        In the formula in Lewis (2000, p54), ik should be replaced by -ik.

        References:
            - Eq. (73)-(75) in Carr P, Sun J (2007) A new approach for option pricing under stochastic volatility. Rev Deriv Res 10:87–150. https://doi.org/10.1007/s11147-007-9014-6
            - p 54 in Lewis AL (2000) Option valuation under stochastic volatility: With Mathematica code. Finance Press, Newport Beach, CA
        """
        vov2 = self.vov**2

        mu = 0.5 + (self.mr - uu*self.rho*self.vov)/vov2
        c_tilde = uu*(1 - uu)/vov2
        delta = np.sqrt(mu**2 + c_tilde)
        alpha = -mu + delta
        beta = 1 + 2*delta

        mr_new = self.mr * self.theta
        XX = 2*mr_new/(self.vov**2 * self.sigma)/np.expm1(mr_new * texp)

        # we use log version because of large argument of np.exp()
        expo = np.clip(spsp.loggamma(beta - alpha) - spsp.loggamma(beta) + alpha*np.log(XX), -self.expo_max, self.expo_max)
        ret = np.exp(expo) * self.hyp1f1_complex(alpha, beta, -XX)

        return ret
