import numpy as np
import scipy.integrate as scint
from . import sv_abc as sv


class GarchApproxUncor(sv.SvABC):
    """
    The implementation of Barone-Adesi et al. (2004)'s approximation pricing formula for European
    options under uncorrelated GARCH diffusion model.

    References: Barone-Adesi, G., Rasmussen, H., Ravanelli, C., 2005. An option pricing formula for the GARCH diffusion model. Computational Statistics & Data Analysis, 2nd CSDA Special Issue on Computational Econometrics 49, 287–310. https://doi.org/10.1016/j.csda.2004.05.014

    This method is only used to compare with the method GarchCondMC.
    """

class GarchCondMC(sv.SvABC, sv.CondMcBsmABC):
    """
        Garch model with conditional Monte-Carlo simulation
        """

    def vol_paths(self, tobs):
        return 0

    def cond_fwd_vol(self, texp):
        return 0

