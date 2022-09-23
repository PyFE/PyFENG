import abc
import numpy as np
import scipy.stats as spst
from . import sv_abc as sv
from . import heston
from . import heston_mc
from . import quad


class HestonMixture(heston.HestonABC, sv.SvMixtureABC):

    cond_avgvar_mv = heston_mc.HestonMcABC.cond_avgvar_mv
    x1star_avgvar_mv = heston_mc.HestonMcABC.x1star_avgvar_mv
    x2star_avgvar_mv = heston_mc.HestonMcABC.x2star_avgvar_mv

    @staticmethod
    def ncx2_quad_pois(n_quad, df, nc):
        cutoff = 1e-10
        pois_lim = spst.poisson.isf(cutoff, nc / 2)
        pois = np.arange(pois_lim + 1)

        xx_list = []
        ww_list = []
        pois_list = []

        pp = spst.poisson.pmf(pois, nc / 2)
        for (i, pi) in enumerate(pp):
            xi, wi = quad.Gamma(n_quad, shape=df / 2 + i)

            pois_list.append(np.full(n_quad, i))
            xx_list.append(2 * xi)
            ww_list.append(wi * pi)

        pois = np.concatenate(pois_list)
        xx = np.concatenate(xx_list)
        ww = np.concatenate(ww_list)

        return xx, ww, pois

    def var_t(self, texp):

        chi_df = self.chi_dim()
        phi, exp = self.phi_exp(texp)
        chi_nonc = self.sigma * exp * phi
        var, ww, pois = self.ncx2_quad_pois(9, chi_df, chi_nonc)
        var *= (exp / phi)

        return var, ww, pois

    def cond_spot_sigma(self, texp):

        var_t, ww, pois = self.var_t(texp)

        avgvar_m, avgvar_v = self.cond_avgvar_mv(texp, self.sigma, var_t, pois)
        ig_lam = avgvar_m ** 3 / avgvar_v

        var, avgvar, ww2 = [], [], []

        for i, var_i in enumerate(var_t):
            xi, wi = quad.InvGauss(7, avgvar_m[i], ig_lam[i])

            var.append(np.full_like(xi, var_i))
            avgvar.append(xi)
            ww2.append(ww[i] * wi)

        var_t = np.concatenate(var)
        avgvar = np.concatenate(avgvar)
        ww2 = np.concatenate(ww2)

        sigma_cond = np.sqrt((1-self.rho**2) * avgvar / self.sigma)
        spot_cond = ((var_t - self.sigma) + self.mr * texp * (avgvar - self.theta)) / self.vov - 0.5 * self.rho * texp * avgvar
        spot_cond *= self.rho
        np.exp(spot_cond, out=spot_cond)

        return spot_cond, sigma_cond, ww2
