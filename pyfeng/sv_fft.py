import numpy as np
import abc
import scipy.fft as spfft
import scipy.special as spsp
import scipy.interpolate as spinterp
import scipy.integrate as spint
import functools
from . import opt_abc as opt
from . import sv_abc as sv
from . import opt_smile_abc as smile
from . import ousv
from . import heston
from . import rheston


class FftABC(opt.OptABC, abc.ABC):
    n_x = 2**12  # number of grid. power of 2 for FFT
    x_lim = 200  # integratin limit

    @abc.abstractmethod
    def mgf_logprice(self, uu, texp):
        """
        Moment generating function (MGF) of log price. (forward = 1)

        Args:
            uu: dummy variable
            texp: time to expiry

        Returns:
            MGF value at uu
        """
        return NotImplementedError

    def charfunc_logprice(self, x, texp):
        """
        Characteristic function of log price

        Args:
            x:
            texp:

        Returns:

        """
        return self.mgf_logprice(1j*x, texp)

    def price(self, strike, spot, texp, cp=1):
        fwd, df, divf = self._fwd_factor(spot, texp)

        kk = strike/fwd
        log_kk = np.log(kk)

        dx = self.x_lim/self.n_x
        xx = np.arange(self.n_x + 1)[:, None]*dx  # the final value x_lim is excluded
        yy = (np.exp(-log_kk*xx*1j)*self.mgf_logprice(xx*1j + 0.5, texp)).real/(xx**2 + 0.25)
        int_val = spint.simpson(yy, dx=dx, axis=0)
        if np.isscalar(kk):
            int_val = int_val[0]
        price = np.where(cp > 0, 1, kk) - np.sqrt(kk)/np.pi*int_val
        return df*fwd*price

    @functools.lru_cache(maxsize=16)
    def fft_interp(self, texp, *args, **kwargs):
        """ FFT method based on the Lewis expression

        References:
            https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/1.3%20Fourier%20transform%20methods.ipynb
        """
        dx = self.x_lim/self.n_x
        xx = np.arange(self.n_x)*dx  # the final value x_lim is excluded

        weight = np.ones(self.n_x)  # Simpson weights
        weight[1:-1:2] = 4
        weight[2:-1:2] = 2
        weight *= dx/3

        dk = 2*np.pi/self.x_lim
        b = self.n_x*dk/2
        ks = -b + dk*np.arange(self.n_x)

        integrand = np.exp(-1j*b*xx)*self.mgf_logprice(xx*1j + 0.5, texp)/(xx**2 + 0.25)*weight
        # CF: integrand = np.exp(-1j*b*xx)*self.cf(xx - 0.5j, texp)*1/(xx**2 + 0.25)*weight
        integral_value = (self.n_x/np.pi)*spfft.ifft(integrand).real

        obj = spinterp.interp1d(ks, integral_value, kind='cubic')
        return obj

    def price_fft(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        kk = strike/fwd
        log_kk = np.log(kk)

        # self.params_hash(), self.n_x, self.x_lim are only used for cache key
        spline_cub = self.fft_interp(texp, k1=self.params_hash(), k2=self.n_x, k3=self.x_lim)
        price = np.where(cp > 0, 1, kk) - np.sqrt(kk)*spline_cub(-log_kk)
        return df*fwd*price


class BsmFft(FftABC):
    """
    Option pricing under Black-Scholes-Merton (BSM) model using fast fourier transform (FFT).

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.BsmFft(sigma=0.2, intr=0.05, divr=0.1)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([15.71362027,  9.69251556,  5.52948647,  2.94558375,  1.4813909 ])
    """

    def mgf_logprice(self, uu, texp):
        val = -0.5*self.sigma**2*texp*uu*(1 - uu)
        # CF: val = -0.5 * self.sigma**2 * texp * uu * (1j + uu)
        return np.exp(val)


class VarGammaFft(sv.SvABC, FftABC):

    def mgf_logprice(self, uu, texp):
        volvar = self.vov*self.sigma**2
        mu = np.log(1 - self.theta*self.vov - 0.5*volvar)  # /self.vov
        # CF: rv = 1j*mu*uu - np.log(1 + (-1j*self.theta*self.vov + 0.5*volvar*uu)*uu)
        rv = mu*uu - np.log(1 + (-self.theta*self.vov - 0.5*volvar*uu)*uu)
        np.exp(texp/self.vov*rv, out=rv)
        return rv


class ExpNigFft(sv.SvABC, FftABC):

    def mgf_logprice(self, uu, texp):
        """

        Args:
            uu:
            texp:

        Returns:

        """
        volvar = self.vov*self.sigma**2
        mu = -1 + np.sqrt(1 - 2*self.theta*self.vov - volvar)
        rv = mu*uu + 1 - np.sqrt(1 + (-2*self.theta*self.vov - volvar*uu)*uu)
        # CF: rv = 1j*mu*uu + 1 - np.sqrt(1 + (-2j*self.theta*self.vov + volvar*uu)*uu)
        np.exp(texp/self.vov*rv, out=rv)
        return rv


class HestonFft(heston.HestonABC, FftABC):
    """
    Heston model option pricing with FFT

    References:
        - Lewis AL (2000) Option valuation under stochastic volatility: with Mathematica code. Finance Press

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> strike = np.array([60, 70, 100, 140])
        >>> sigma, vov, mr, rho, texp, spot = 0.04, 1, 0.5, -0.9, 10, 100
        >>> m = pf.HestonFft(sigma, vov=vov, mr=mr, rho=rho)
        >>> m.price(strike, spot, texp)
        >>> # true price: 44.32997507, 35.8497697, 13.08467014, 0.29577444
        array([44.32997507, 35.8497697 , 13.08467014,  0.29577444])
    """

    def mgf_logprice(self, uu, texp):
        """
        Log price MGF under the Heston model.
        We use the characteristic function in Eq (2.8) of Lord & Kahl (2010) that is
        continuous in branch cut when complex log is evaluated.

        References:
            - Heston SL (1993) A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options. The Review of Financial Studies 6:327–343. https://doi.org/10.1093/rfs/6.2.327
            - Lord R, Kahl C (2010) Complex Logarithms in Heston-Like Models. Mathematical Finance 20:671–694. https://doi.org/10.1111/j.1467-9965.2010.00416.x
        """
        var_0 = self.sigma
        vov2 = self.vov**2

        beta = self.mr - self.vov*self.rho*uu
        dd = np.sqrt(beta**2 + vov2*uu*(1 - uu))
        gg = (beta - dd)/(beta + dd)
        exp = np.exp(-dd*texp)
        tmp1 = 1 - gg*exp

        mgf = self.mr*self.theta*((beta - dd)*texp - 2*np.log(tmp1/(1 - gg))) + var_0*(beta - dd)*(1 - exp)/tmp1
        return np.exp(mgf/vov2)


class RoughHestonFft(rheston.RoughHestonABC, FftABC):
    """
    Rough Heston model option pricing with FFT 
    
    Method 1: Adam's Method is employed to solve the fractional Riccati Equation.
    References:
        - El Euch O, Rosenbaum M (2019) The characteristic function of rough Heston models. Mathematical Finance 29:3–38. https://doi.org/10.1111/mafi.12173
        
    Method 2: Fast Hybrid Approximation is employed to solve the fractional Riccati Equation.
    References:
        - Giorgia Callegaro, Martino Grasselli, Gilles Pagès (2020) Fast Hybrid Schemes for Fractional Riccati Equations (Rough Is Not So Tough). Mathematics of Operations Research 46(1):221-254.https://doi.org/10.1287/moor.2020.1054

    Examples:
        >>> import numpy as np
        >>> import pyfeng.ex as pfex
        >>> strike = np.array([60, 70, 100, 140])
        >>> sigma, vov, mr, rho, texp, spot,theta,alpha = 0.0392, 0.1, 0.3156, -0.681, 1, 100, 0.3156, 0.62
        >>> # sigma, vov, mr, rho, texp, spot,alpha = 0.04, 0.1, 0.5, -0.9, 1, 100, 0.62
        >>> m = pfex.RoughHestonFft(sigma, vov=vov, mr=mr, rho=rho,alpha = alpha)
        >>> m.price(strike, spot, texp)
            
        >>> strike = np.linspace(0.8,1.2,9)
        >>> sigma, vov, mr, rho, texp, spot,theta,alpha = 0.0392, 0.331, 0.1, -0.681, 1/12, 1, 0.3156, 0.62
        >>> m = pfex.RoughHestonFft(sigma, vov=vov, mr=mr, rho=rho,alpha = alpha, theta = theta)
        >>> m.price(strike, spot, texp)
        array([2.00004861e-01, 1.50105032e-01, 1.01136892e-01, 5.67278209e-02,2.39151266e-02, 6.81264421e-03, 1.19548373e-03, 1.19636483e-04, 6.36294510e-06])
            
    """

    method = 1
    x_lim = 200  # integratin limit

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
    
    def F(self,a,x):
        """
       Left hand side of fractional Riccati equation. It is transformed to get the moment-generating function.

        Args:
            a: dummy variable
            x: value of the solution function

        """
        return (1/2) * (pow(a,2) - a) + self.mr * (a * self.rho * self.vov - 1) * x + pow(self.mr * self.vov, 2) * pow(x,2) / 2
    
    def Ih(self,r,t,a,hh):
        """
        Fractional integral of order 𝑟 ∈ (0, 1] of a function, equation 4.3 in El Euch O, Rosenbaum M (2019)

        Args:
            t: time
            funcA: kernal function

        """
        grid = np.arange(0,t,self.delta)
        Ihrs = 0 + 0j
        for s in np.arange(0,t,self.delta):
            Ihrs += pow(t - s,r - 1) * hh[int(s//self.delta)] * self.delta
        return Ihrs / spsp.gamma(r)

    def mgf_logprice_adam(self, uu, texp):
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
        for i in range(0,len(uu)):
            k = int(texp/delta)
            h_hat = np.zeros(int(k + 1),dtype = 'complex')
            h_hat_p = np.zeros(int(k + 1),dtype = 'complex')
            F_a_h_hat = np.zeros(int(k + 1),dtype = 'complex')
            F_a_h_hat[0] = self.F(uu[i],0)
            for j in range(1,k + 1):
                F_a_h_hat[j] = self.F(uu[i],h_hat[j-1])
                h_hat_p[j] = np.dot(self.b_kp1(j-1,delta), F_a_h_hat[0:j])
                h_hat[j] = np.dot(self.a_kp1(j-1,delta),F_a_h_hat[0:j]) + self.a_j_kp1(j,delta,j + 1) * self.F(uu[i],h_hat_p[j])
            LL[i] = self.theta * self.mr * self.Ih(1,texp,uu[i],h_hat) + self.sigma * self.Ih(1 - self.alpha,texp,uu[i],h_hat)
       
        return np.exp(LL)
    
    '''
    Method 2: Fast Hybrid Method
    '''

    def a(self,r_0,lambd,mu,nu):
        '''
        'a' sequence in (19)
        '''
        a = np.zeros(r_0 + 1,dtype = 'complex')
        a_2 = np.zeros(r_0 + 1,dtype = 'complex')
        a[1] = nu / spsp.gamma(self.alpha + 1)
        a_2[1] = 0
        for i in range(2,r_0 + 1):
            tmp = spsp.gammaln(self.alpha * (i - 1) + 1) - spsp.gammaln(self.alpha * (i - 1) + self.alpha + 1)
            a[i] = (lambd * a_2[i-1] + mu * a[i-1]) * np.exp(tmp)
            a_2[i] = np.sum(a[:i] * a[i-1::-1])
        return a

    def I_1_FPS(self,t,lambd,mu,nu,r_0,aSequence):
        '''
        Power Series Expansion of fractional integral in (26)
        '''
        I_1 = 0
        for i in range(1,r_0 + 1):
            I_1 += aSequence[i] * pow(t,self.alpha * i) / (self.alpha * i + 1)
        return t * I_1

    def I_2_FPS(self,t,lambd,mu,nu,r_0,aSequence):
        '''
        Power Series Expansion of fractional integral in (27)
        '''
        I_2 = 0
        for i in range(2,r_0 + 1):
            tmp = spsp.gammaln(self.alpha * i) - spsp.gammaln(self.alpha * i - self.alpha)
            I_2 += aSequence[i] * pow(t,i * self.alpha) * np.exp(tmp) / ((1- 1/i) * (self.alpha * i + 1 - self.alpha))
        return pow(t, 1- self.alpha) * (nu * pow(t,self.alpha) + I_2)

    def phi_t(self,t,aSequence):
        '''
       the approximation of the solution of the fraction Riccati function using power series expension,which is equation (25) in the reference paper
        '''
        phi_t = 0
        for i in range(1,self.r_0 + 1):
            phi_t += aSequence[i] * pow(t,self.alpha * i)
        return phi_t

    def phi_n(self,k,lambd,mu,nu,aSequence):
        '''
       the approximation of the solution of the fraction Riccati function using Euler discretization,which is equation (28) in the reference paper
        '''
        phi_n = np.zeros(k+1,dtype = 'complex')
        for i in range(1,int(self.k_0)+1):
            phi_n[i] = self.phi_t(self.t[i],aSequence)
        for i in range(int(self.k_0)+1,k+1):
            phi_n_j = 0
            for j in range(1,i):
                phi_n_j += self.c(i-j-1) * phi_n[j] * (lambd * phi_n[j] + mu)
            phi_n[i] = pow(self.texp/self.n,self.alpha) * (nu * i ** self.alpha + phi_n_j) / spsp.gamma(self.alpha + 1)
        return phi_n

    def c(self,l):
        return pow(l + 1,self.alpha) - pow(l,self.alpha)

    def c_alpha(self,l):
        return pow(l + 1,1 - self.alpha) - pow(l,1 - self.alpha)

    def I_1_PED(self,k,I_1_PED_tk0,lambd,mu,nu,phiSequence):
        '''
       the approximation of the regular derivative I(alpha)(ψ) using Euler discretization,which is equation (28) in the reference paper
        '''
        temp = 0
        for i in range(int(self.k_0),k):
            temp += phiSequence[i]
        return I_1_PED_tk0 + temp * self.texp / self.n + (phiSequence[k] - phiSequence[self.k_0]) * self.texp/(2 * self.n)

    def I_2_PED(self,k,lambd,mu,nu,phiSequence):
        '''
       the approximation of the fractional antiderivative I(1−alpha)(ψ) using Euler discretization,which is equation (28) in the reference paper
        '''
        temp = 0
        for i in range(1,k):
            temp += pow(self.c_alpha(k-i-1),1 - self.alpha) * phiSequence[i]
        return pow(self.texp/self.n,1 - self.alpha) * temp / spsp.gamma(2 - self.alpha)

    def mgf_logprice_hybrid(self, uu, texp):
        """
        Log price MGF under the rough Heston model.
        The moment-generating function in Eq (2) in the reference paper ,which is from Eq (4.5) of El Euch, O., Rosenbaum, M.: (2019) The characteristic function of rough Heston models was employed

        References:
            - Giorgia Callegaro, Martino Grasselli, Gilles Pagès (2020) Fast Hybrid Schemes for Fractional Riccati Equations (Rough Is Not So Tough). Mathematics of Operations Research 46(1):221-254.https://doi.org/10.1287/moor.2020.1054 
            - El Euch, O., Rosenbaum, M.: (2019) The characteristic function of rough Heston models https://doi.org/10.1111/mafi.12173
        """
        sigma = self.sigma
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

        lambd = pow((vov * mr),2) /2
        LL = uu
        thre = 0.9
        convRadius = 0.22
        k_0 = int((n * thre * convRadius) // texp)
        self.k_0 = k_0
        
        if texp <= thre * convRadius:
            '''
            if texp is smaller than a threhold times convergence radius, we use Power Serise Expension to solve the fraction Riccati function and obtain the moment-generating function
            '''
            for i in range(0,len(uu)):
                u = uu[i]
                nu = (u ** 2 - u) / 2
                mu = mr * (u * rho * vov - 1)
                aSequence = self.a(r_0,lambd,mu,nu)
                LL[i] = np.exp(theta * mr * self.I_1_FPS(texp,lambd,mu,nu,r_0,aSequence) + sigma * self.I_2_FPS(texp,lambd,mu,nu,r_0,aSequence))
        else:
            '''
            if texp is larger than a threhold times convergence radius, we use Euler discretization to solve the fraction Riccati function and obtain the moment-generating function
            '''
            t = np.linspace(0,texp,n + 1)
            self.t = t
            for i in range(0,len(uu)):
                u = uu[i]
                nu = (u ** 2 - u) / 2
                mu = mr * (u * rho * vov - 1)
                aSequence = self.a(r_0,lambd,mu,nu)
                phiSequence = self.phi_n(n,lambd,mu,nu,aSequence)
                I_1_PED_tk0 = 0
                for i in range(1,r_0 + 1):
                    I_1_PED_tk0 += pow(t[k_0],alpha * i + 1) * aSequence[i] /(alpha * i + 1)
                LL[i] = np.exp(theta * mr * self.I_1_PED(n,I_1_PED_tk0,lambd,mu,nu,phiSequence) + sigma * self.I_2_PED(n,lambd,mu,nu,phiSequence))
        
        return LL
    
    def mgf_logprice(self, uu, texp):
        '''
        choose method to solve the fractional Riccati equation
        '''
        if self.method == 1:
            rv = self.mgf_logprice_adam(uu, texp)
        elif self.method == 2:
            rv = self.mgf_logprice_hybrid(uu, texp)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return rv



class OusvFft(ousv.OusvABC, FftABC):
    """
    OUSV model option pricing with FFT

    """

    def mgf_logprice(self, uu, texp):
        """
        Log price MGF under the OUSV model.
        We use the characteristic function in Eq (4.14) of Lord & Kahl (2010) that is
        continuous in branch cut when complex log is evaluated.

        Returns:
            MGF value at uu

        References:
            - Lord R, Kahl C (2010) Complex Logarithms in Heston-Like Models. Mathematical Finance 20:671–694. https://doi.org/10.1111/j.1467-9965.2010.00416.x
        """

        var_0 = self.sigma**2
        sigma_0 = self.sigma

        # equivalent Heston params when theta=0
        mr_h, vov_h, theta_h = 2*self.mr, 2*self.vov, self.vov**2/(2*self.mr)
        vov2_h = 4*self.vov**2

        beta = mr_h - self.rho*vov_h*uu
        dd = np.sqrt(beta**2 + vov2_h*uu*(1 - uu))
        gg = (beta - dd)/(beta + dd)

        exp_h = np.exp(-0.5*dd*texp)
        exp = exp_h**2
        tmp1 = 1 - gg*exp

        # Heston model part
        mgf = mr_h*theta_h*((beta - dd)*texp - 2*np.log(tmp1/(1 - gg))) + var_0*(beta - dd)*(1 - exp)/tmp1
        mgf /= vov2_h

        # Additional part for OUSV
        bb = (1 - exp_h)**2/tmp1
        aa = 0.5*self.mr*self.theta/dd**2
        aa *= beta*(dd*texp - 4) + dd*(dd*texp - 2) + 4*((dd**2 - 2*beta**2)/(beta + dd)*exp + 2*beta*exp_h)/tmp1

        mgf += 0.5*self.theta/theta_h*(beta - dd)/dd*(aa + bb*sigma_0)

        return np.exp(mgf)

    def mgf_logprice_schobelzhu1998(self, uu, texp):
        """
        MGF from Eq. (13) in Schobel & Zhu (1998).
        This form suffers discontinuity in complex log branch cut. Should not be used for pricing.

        Args:
            uu: dummy variable
            texp: time to expiry

        Returns:
            MGF value at uu

        References:
            - Schöbel R, Zhu J (1999) Stochastic Volatility With an Ornstein–Uhlenbeck Process: An Extension. Rev Financ 3:23–46. https://doi.org/10.1023/A:1009803506170

        """
        mr, theta, vov, rho = self.mr, self.theta, self.vov, self.rho

        # CF: s1 = 0.5*uu*(uu*(1 - rho**2) + 1j*(1 - 2*mr*rho/vov))
        # CF: s2 = 1j*uu*mr*theta*rho/vov
        # CF: s3 = 0.5j*uu*rho/vov
        s1 = 0.5*uu*((1 - 2*mr*rho/vov) - (1 - rho**2)*uu)
        s2 = uu*mr*theta*rho/vov
        s3 = 0.5*uu*rho/vov

        gamma1 = np.sqrt(2*vov**2*s1 + mr**2)
        gamma2 = (mr - 2*vov**2*s3)/gamma1
        gamma3 = mr**2*theta - s2*vov**2
        sinh = np.sinh(gamma1*texp)
        cosh = np.cosh(gamma1*texp)
        sincos = sinh + gamma2*cosh
        cossin = cosh + gamma2*sinh
        ktg3 = mr*theta*gamma1 - gamma2*gamma3
        s2g3 = vov**2*gamma1**3

        D = (mr - gamma1*sincos/cossin)/vov**2
        B = ((ktg3 + gamma3*sincos)/cossin - mr*theta*gamma1)/(vov**2*gamma1)
        C = (
                -0.5*np.log(cossin)
                + 0.5*mr*texp
                + ((mr*theta*gamma1)**2 - gamma3**2)
                /(2*s2g3)
                *(sinh/cossin - gamma1*texp)
                + ktg3*gamma3/s2g3*((cosh - 1)/cossin)
        )

        # CF: res = -0.5*1j*uu*rho*(self.sigma**2/vov + vov*texp)
        res = -0.5*uu*rho*(self.sigma**2/vov + vov*texp)
        res += (D/2*self.sigma + B)*self.sigma + C
        return np.exp(res)


class Sv32Fft(sv.SvABC, FftABC):
    """
    3/2 model option pricing with Fourier inversion

    References:
        - Lewis AL (2000) Option valuation under stochastic volatility: with Mathematica code. Finance Press

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> sigma, mr, theta, vov, rho = 0.06, 20.48, 0.218, 3.20, -0.99
        >>> strike, spot, texp = np.array([95, 100, 105]), 100, 0.5
        >>> m = pf.Sv32Fft(sigma, vov=vov, mr=mr, rho=rho, theta=theta)
        >>> m.price(strike, spot, texp)
        array([11.7235,  8.9978,  6.7091])
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

    def mgf_logprice(self, uu, texp):
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
        XX = 2*mr_new/(self.vov**2 * self.sigma)/(np.exp(mr_new * texp) - 1)

        #ret = spsp.gamma(beta - alpha) * spsp.rgamma(beta) * np.power(XX, alpha) * self.hyp1f1_complex(alpha, beta, -XX)
        # we use log version because of large argument of np.exp()
        expo = np.clip(spsp.loggamma(beta - alpha) - spsp.loggamma(beta) + alpha*np.log(XX), -self.expo_max, self.expo_max)
        ret = np.exp(expo) * self.hyp1f1_complex(alpha, beta, -XX)

        return ret


class CgmyFft(smile.OptSmileABC, FftABC):

    C = 1.
    G = 1.
    M = 1.
    Y = 0.

    def __init__(self, C, G, M, Y, intr=0.0, divr=0.0, is_fwd=False):
        super().__init__(C, intr=intr, divr=divr, is_fwd=is_fwd)  # self.sigma = C
        self.C, self.G, self.M, self.Y = C, G, M, Y

    def mgf_logprice(self, uu, texp):
        """
        MGF of log price of CGMY from Eq (19) from Ballotta and Kyriakou (2014)

        Args:
            uu: dummy variable
            texp: time to expiry

        Returns:
            MGF value at uu

        References:
            - Ballotta L, Kyriakou I (2014) Monte Carlo Simulation of the CGMY Process and Option Pricing. Journal of Futures Markets 34:1095–1121. https://doi.org/10.1002/fut.21647

        """
        gam_Y = spsp.gamma(-self.Y)
        M_pow_Y = np.power(self.M, self.Y)
        G_pow_Y = np.power(self.G, self.Y)

        rv = self.C * gam_Y * (
                np.power(self.M - uu, self.Y) - M_pow_Y
                + np.power(self.G + uu, self.Y) - G_pow_Y
        )
        mu = - self.C * gam_Y * (
            np.power(self.M - 1., self.Y) - M_pow_Y
            + np.power(self.G + 1., self.Y) - G_pow_Y
        )
        np.exp(texp*(mu + rv), out=rv)
        return rv
