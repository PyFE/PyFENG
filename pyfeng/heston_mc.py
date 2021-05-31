import numpy as np
import scipy.integrate as scint
import scipy.stats as spst
import scipy.special as spsp
import scipy.integrate as spint
from scipy.misc import derivative
from . import bsm
from . import sv_abc as sv


class HestonCondMc(sv.SvABC, sv.CondMcBsmABC):
    """
    Heston model with conditional Monte-Carlo simulation
    """

    def vol_paths(self, tobs):
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        dB_t = self._bm_incr(tobs, cum=False)  # B_t (0 <= s <= 1)
        vv_path = np.empty([n_dt+1, self.n_path])  # variance series: V0, V1,...,VT
        vv = self.sigma**2
        vv_path[0, :] = vv
        for i in range(n_dt):
            vv = vv + self.mr * (self.theta ** 2 - vv) * dt[i] + np.sqrt(vv) * self.vov * dB_t[i, :]  # Euler method
            vv = vv + 0.25 * self.vov**2 * (dB_t[i, :]**2 - dt[i])  # Milstein method
            vv[vv < 0] = 0  # variance should be larger than zero
            vv_path[i+1, :] = vv

        # return normalized sigma, e.g., sigma(0) = 1
        return np.sqrt(vv_path)/self.sigma

    def cond_fwd_vol(self, texp):

        tobs = self.tobs(texp)
        n_steps = len(tobs)
        sigma_paths = self.vol_paths(tobs)
        vv0 = self.sigma**2
        vv_ratio = sigma_paths[-1, :]
        int_var_std = scint.simps(sigma_paths**2, dx=1, axis=0) / n_steps

        int_sig_dw = ((vv_ratio - 1) * vv0 - self.mr * texp * (self.theta ** 2 - int_var_std * vv0)) / self.vov
        fwd_cond = np.exp(self.rho * int_sig_dw - 0.5*self.rho**2 * int_var_std * vv0 * texp)
        vol_cond = np.sqrt((1 - self.rho**2) * int_var_std)

        # return normalized forward and volatility
        return fwd_cond, vol_cond


class HestonCondMcQE:
    '''
    Conditional MC for Heston model based on QE discretization scheme by Andersen(2008)

    Underlying price is assumed to follow a geometric Brownian motion.
    Volatility (variance) of the price is assumed to follow a CIR process.

    Example:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = [100.0, 140.0, 70.0]
        >>> forward = 100
        >>> delta = [1, 1/2, 1/4, 1/8, 1/16, 1/32]
        >>> vov, kappa, rho, texp, theta, sigma = [1, 0.5, -0.9, 10, 0.04, 0.2]
        >>> heston_cmc_qe = pf.HestonCondMcQE(vov=vov, kappa=kappa, rho=rho, theta=theta)
        >>> price_cmc = np.zeros([len(delta), len(strike)])
        >>> for d in range(len(delta)):
        >>>     price_cmc[d, :] = heston_cmc_qe.price(strike, forward, texp, sigma=sigma, delta=delta[d], path=1e5, seed=123456)
        >>> price_cmc
        array([[14.52722285,  0.19584722, 37.20591415],
               [13.56691261,  0.26568546, 36.12295964],
               [13.22061601,  0.29003533, 35.9154245 ],
               [13.12057087,  0.29501411, 35.90207168],
               [13.1042753 ,  0.29476289, 35.89245755],
               [13.09047939,  0.29547721, 35.86410028]])
    '''

    def __init__(self, vov=1, kappa=0.5, rho=-0.9, theta=0.04):
        '''
        Initiate a Heston model

        Args:
            vov: volatility of variance, strictly positive
            kappa: speed of variance's mean-reversion, strictly positive
            rho: correlation between BMs of price and vol
            theta: long-term mean (equilibirum level) of the variance, strictly positive
        '''
        self.vov = vov
        self.kappa = kappa
        self.rho = rho
        self.theta = theta

        self.psi_points = None  # for TG scheme only
        self.rx_results = None
        self.dis = 1e-3

    def price(self, strike, spot, texp, sigma, delta, intr=0, divr=0, psi_c=1.5, path=10000, scheme='QE', seed=None):
        '''
        Conditional MC routine for Heston model
        Generate paths for vol only using QE discretization scheme.
        Compute integrated variance and get BSM prices vector for all strikes.

        Args:
            strike: strike price, in vector form
            spot: spot (or forward)
            texp: time to expiry
            sigma: initial volatility
            delta: length of each time step
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            psi_c: critical value for psi, lying in [1, 2]
            path: number of vol paths generated
            scheme: discretization scheme for vt, {'QE', 'TG', 'Euler', 'Milstein', 'KJ'}
            seed: random seed for rv generation

        Return:
            BSM price vector for all strikes
        '''
        self.sigma = sigma
        self.bsm_model = bsm.Bsm(self.sigma, intr=intr, divr=divr)
        self.delta = delta
        self.path = int(path)
        self.step = int(texp / self.delta)

        vt = self.sigma ** 2 * np.ones([self.path, self.step + 1])
        np.random.seed(seed)

        if scheme == 'QE':
            u = np.random.uniform(size=(self.path, self.step))

            expo = np.exp(-self.kappa * self.delta)
            for i in range(self.step):
                # compute m, s_square, psi given vt(i)
                m = self.theta + (vt[:, i] - self.theta) * expo
                s2 = vt[:, i] * (self.vov ** 2) * expo * (1 - expo) / self.kappa + self.theta * (self.vov ** 2) * \
                     ((1 - expo) ** 2) / (2 * self.kappa)
                psi = s2 / m ** 2

                # compute vt(i+1) given psi
                below = np.where(psi <= psi_c)[0]
                ins = 2 * psi[below] ** -1
                b2 = ins - 1 + np.sqrt(ins * (ins - 1))
                b = np.sqrt(b2)
                a = m[below] / (1 + b2)
                z = spst.norm.ppf(u[below, i])
                vt[below, i+1] = a * (b + z) ** 2

                above = np.where(psi > psi_c)[0]
                p = (psi[above] - 1) / (psi[above] + 1)
                beta = (1 - p) / m[above]
                for k in range(len(above)):
                    if u[above[k], i] > p[k]:
                        vt[above[k], i+1] = beta[k] ** -1 * np.log((1 - p[k]) / (1 - u[above[k], i]))
                    else:
                        vt[above[k], i+1] = 0

        elif scheme == 'TG':
            if np.all(self.rx_results) == None:
                self.psi_points, self.rx_results = self.prepare_rx()

            expo = np.exp(-self.kappa * self.delta)
            for i in range(self.step):
                # compute m, s_square, psi given vt(i)
                m = self.theta + (vt[:, i] - self.theta) * expo
                s2 = vt[:, i] * (self.vov ** 2) * expo * (1 - expo) / self.kappa + self.theta * (self.vov ** 2) * \
                     ((1 - expo) ** 2) / (2 * self.kappa)
                psi = s2 / m ** 2

                rx = np.array([self.find_rx(j) for j in psi])

                z = np.random.normal(size=(self.path, self.step))
                mu_v = np.zeros_like(z)
                sigma_v = np.zeros_like(z)
                mu_v[:, i] = rx * m / (spst.norm.pdf(rx) + rx * spst.norm.cdf(rx))
                sigma_v[:, i] = np.sqrt(s2) * psi ** (-0.5) / (spst.norm.pdf(rx) + rx * spst.norm.cdf(rx))

                vt[:, i+1] = np.fmax(mu_v[:, i] + sigma_v[:, i] * z[:, i], 0)

        elif scheme == 'Euler':
            z = np.random.normal(size=(self.path, self.step))
            for i in range(self.step):
                vt[:, i+1] = vt[:, i] + self.kappa * (self.theta - np.max(vt[:, i], 0)) * self.delta + \
                             self.vov * np.sqrt(np.max(vt[:, i], 0) * self.delta) * z[:, i]
            below_0 = np.where(vt < 0)
            vt[below_0] = 0

        elif scheme == 'Milstein':
            z = np.random.normal(size=(self.path, self.step))
            for i in range(self.step):
                vt[:, i+1] = vt[:, i] + self.kappa * (self.theta - np.max(vt[:, i], 0)) * self.delta + self.vov * \
                             np.sqrt(np.max(vt[:, i], 0) * self.delta) * z[:, i] + \
                             self.vov**2 * 0.25 * (z[:, i]**2 - 1) * self.delta
            below_0 = np.where(vt < 0)
            vt[below_0] = 0

        elif scheme == 'KJ':
            z = np.random.normal(size=(self.path, self.step))
            for i in range(self.step):
                vt[:, i+1] = (vt[:, i] + self.kappa * self.theta * self.delta + self.vov * \
                             np.sqrt(np.max(vt[:, i], 0) * self.delta) * z[:, i] + \
                             self.vov**2 * 0.25 * (z[:, i]**2 - 1) * self.delta) / (1 + self.kappa * self.delta)
            below_0 = np.where(vt < 0)
            vt[below_0] = 0

        # compute integral of vt, equivalent spot and vol
        vt_int = spint.simps(vt, dx=self.delta)
        spot_cmc = spot * np.exp(self.rho * (vt[:, -1] - vt[:, 0] - self.kappa * (self.theta * texp - vt_int))
                                 / self.vov - self.rho ** 2 * vt_int / 2)
        vol_cmc = np.sqrt((1 - self.rho ** 2) * vt_int / texp)

        # compute bsm price vector for the given strike vector
        price_cmc = np.zeros_like(strike)
        for j in range(len(strike)):
            price_cmc[j] = np.mean(self.bsm_model.price_formula(strike[j], spot_cmc, vol_cmc, texp, intr=intr, divr=divr))

        return price_cmc

    def prepare_rx(self):
        '''
        Pre-calculate r(x) and store the result
        for TG scheme only
        '''
        fx = lambda rx: rx * spst.norm.pdf(rx) + spst.norm.cdf(rx) * (1 + rx ** 2) / \
                        ((spst.norm.pdf(rx) + rx * spst.norm.cdf(rx)) ** 2) - 1
        rx_results = np.linspace(-2, 100, 10 ** 5)
        psi_points = fx(rx_results)

        return psi_points, rx_results

    def find_rx(self, psi):
        '''
        Return r(psi) according to the pre_calculated results
        '''

        if self.rx_results[self.psi_points >= psi].size == 0:
            print("Caution: input psi too large")
            return self.rx_results[-1]
        elif self.rx_results[self.psi_points <= psi].size == 0:
            print("Caution: input psi too small")
            return self.rx_results[0]
        else:
            return (self.rx_results[self.psi_points >= psi][0] + self.rx_results[self.psi_points <= psi][-1])/2


class HestonMcAe:
    """
    Almost exact MC for Heston model.

    Underlying price is assumed to follow a geometric Brownian motion.
    Volatility (variance) of the price is assumed to follow a CIR process.
    Example:
        >>> strike = 100
        >>> spot = 100
        >>> vov, kappa, rho, texp, theta, sigma = 0.61, 6.21, -0.7, 1, 0.019, 0.10201
        >>> heston_ae = HestonMcAe(vov, kappa, rho, theta, r)
        >>> price_ae = heston_ae.price(strike, spot, texp, sigma_0, intr=0, divr=0)
        >>> price_ae
        8.946951375550809
    """
    def __init__(self, vov=1, kappa=0.5, rho=-0.9, theta=0.04, r=0):
        """
        Initiate a Heston model

        Args:
            vov: volatility of variance, strictly positive
            kappa: speed of variance's mean-reversion, strictly positive
            rho: correlation between BMs of price and vol
            theta: long-term mean (equilibirum level) of the variance, strictly positive
            r：the drift item
        """
        self.vov = vov
        self.theta = theta
        self.kappa = kappa
        self.rho = rho
        self.r = r

    def ch_f(self, texp, sigma_0, sigma_t, chi_dim):
        """

        Args:
            texp: time to expiry
            sigma_0: initial volatility
            sigma_t: volatility at time T
            chi_dim: dimensions of chisquare distribution

        Returns:
            ch_f: characteristic function of the distribution of integral sigma_t

        """
        gamma_f = lambda a: np.sqrt(self.kappa ** 2 - 2 * self.vov**2 * a * 1j)

        temp_f = lambda a: gamma_f(a) * texp

        ch_f_part_1 = lambda a: gamma_f(a) * np.exp(-0.5 * (temp_f(a) - self.kappa * texp)) \
                                * (1 - np.exp(-self.kappa * texp)) / (self.kappa * (1 - np.exp(-temp_f(a))))

        ch_f_part_2 = lambda a: np.exp((sigma_0**2 + sigma_t**2) / self.vov ** 2 * \
                                       (self.kappa * (1 + np.exp(-self.kappa * texp)) / (1 - np.exp(-self.kappa * texp))
                                        - gamma_f(a) * (1 + np.exp(-temp_f(a))) / (1 - np.exp(-temp_f(a)))))

        ch_f_part_3 = lambda a: spsp.iv(0.5 * chi_dim - 1, np.sqrt(sigma_0**2 * sigma_t**2) * 4 * gamma_f(a) *
                                   np.exp(-0.5 * temp_f(a)) / (self.vov ** 2 * (1 - np.exp(-temp_f(a))))) / \
                                spsp.iv(0.5 * chi_dim - 1, np.sqrt(sigma_0**2 * sigma_t**2) * 4 * self.kappa *
                                   np.exp(-0.5 * self.kappa * texp) / (
                                           self.vov ** 2 * (1 - np.exp(- self.kappa * texp))))

        ch_f = lambda a: ch_f_part_1(a) * ch_f_part_2(a) * ch_f_part_3(a)
        return ch_f

    def gen_vov_t(self, chi_dim, chi_lambda, texp, n_paths):
        """

        Args:
            chi_dim: dimensions of chisquare distribution
            chi_lambda: the skewing item of chisquare distribution
            texp: time to expiry
            n_paths: number of vol paths generated

        Returns:
            sigma_t: volatility at time T

        """
        cof = self.vov ** 2 * (1 - np.exp(-self.kappa * texp)) / (4 * self.kappa)
        sigma_t = np.sqrt(cof * np.random.noncentral_chisquare(chi_dim, chi_lambda, n_paths))
        return sigma_t

    def gen_s_t(self, spot, sigma_t, sigma_0, texp, integral_sigma_t, n_paths):
        """

        Args:
            spot: spot (or forward)
            sigma_t: volatility at time T
            sigma_0: initial volatility
            texp: time to expiry
            integral_sigma_t: samples from the distribution of integral sigma_t
            n_paths: number of vol paths generated

        Returns:
            s_t: stock price at time T
        """

        integral_sqrt_sigma_t = (sigma_t**2 - sigma_0**2 - self.kappa * self.theta * texp + self.kappa * integral_sigma_t)\
                                / self.vov
        mean = np.log(spot) + (self.r * texp - 0.5 * integral_sigma_t + self.rho * integral_sqrt_sigma_t)
        sigma_2 = np.sqrt((1 - self.rho ** 2) * integral_sigma_t)
        s_t = np.exp(mean + sigma_2 * np.random.normal(size=n_paths))
        return s_t

    def price(self, strike, spot, texp, sigma_0, intr=0, divr=0, n_paths=10000, seed=None,
              dis_can="Inverse-Gaussian", call=1):
        """
        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry
            sigma_0: initial volatility
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            n_paths: number of vol paths generated
            seed: random seed for rv generation
        Returns:
            price_ae: option price
        """
        if seed:
            np.random.seed(seed)

        chi_dim = (4 * self.theta * self.kappa) / (self.vov ** 2)
        chi_lambda = (4 * self.kappa * np.exp(-self.kappa * texp)) / \
                     ((self.vov ** 2) * (1 - np.exp(-self.kappa * texp))) * sigma_0**2

        sigma_t = self.gen_vov_t(chi_dim, chi_lambda, texp, n_paths)

        ch_f = self.ch_f(texp, sigma_0, sigma_t, chi_dim)

        moment_1st = (derivative(ch_f, 0, n=1, dx=1e-5) / 1j).real
        moment_2st = (derivative(ch_f, 0, n=2, dx=1e-5) / (1j ** 2)).real

        if dis_can == "Inverse-Gaussian":
            scale_ig = moment_1st**3 / (moment_2st - moment_1st**2)
            miu_ig = moment_1st / scale_ig
            integral_sigma_t = spst.invgauss.rvs(miu_ig, scale=scale_ig)
            s_t = self.gen_s_t(spot, sigma_t, sigma_0, texp, integral_sigma_t, n_paths)

        elif dis_can == "Log-normal":
            scale_ln = np.sqrt(np.log(moment_2st) - 2 * np.log(moment_1st))
            miu_ln = np.log(moment_1st) - 0.5 * scale_ln ** 2
            integral_sigma_t = np.random.lognormal(miu_ln, scale_ln)
            s_t = self.gen_s_t(spot, sigma_t, sigma_0, texp, integral_sigma_t, n_paths)
        else:
            print("This function is not currently a candidate function!")
            return -1

        if call:
            price_ae = np.fmax(s_t - strike, 0).mean()
        else:
            price_ae = np.fmax(strike - s_t, 0).mean()

        return np.exp(- self.r * texp) * price_ae
