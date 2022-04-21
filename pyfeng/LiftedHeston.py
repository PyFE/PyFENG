import numpy as np
import scipy.integrate as spint
import scipy.special as scsp


class LiftedHeston:

    def __init__(
        self,  vov=0.0, v0=0.0, rho=0.0, theta=0.0, lamda=0.0, H=0.0, n=20, rn=2.5
    ):
        """
        Args:
            vov: volatility of volatility
            v0: initial variance
            rho: correlation between price and volatility
            theta: mean_reversion term
            lamda: mean_reversion term
            H: Hurst index, measuring regularity of the sample paths of V
            n: number of factors
            rn: rn ↓ 1 and n*log(rn) → ∞ as n → ∞
            n and rn are fixed according to (26) in the paper
        """
        self.vov = vov
        self.v0=v0
        self.rho = rho
        self.theta = theta
        self.lamda = lamda
        self.H = H
        self.n = n
        self.rn = rn

    def weight(self):
    	"""
    	Generate cn from known parameters
    	according to equation (19) in the paper
    	"""
    	cn_v = []
    	for i in range(0,self.n):
    		alpha = self.H + 1/2
    		cn_i = (np.power(self.rn,1-alpha) -1) * np.power(self.rn,(alpha-1)*(1+self.n/2)) * np.power(self.rn,i*(1-alpha))/(scsp.gamma(alpha)*scsp.gamma(2-alpha))
    		cn_v += [cn_i]
    	return np.array(cn_v)

    def mean_rev_speed(self):

    	"""
    	Generate xn from known parameters
    	according to equation (19) in the paper
    	"""

    	xn_v = []
    	for i in range(0,self.n):
    		alpha = self.H + 1/2
    		xn_i = ((1-alpha) * (np.power(self.rn,(2-alpha)) - 1) * np.power(self.rn,i-1-self.n/2))/((2-alpha)* (np.power(self.rn,1-alpha)-1))
    		xn_v += [xn_i]
    	return np.array(xn_v)

    def gn(self, t):

    	"""
    	Generate gn(t) according to equation (10) in the paper
    	"""

    	mean_revs = self.mean_rev_speed()
    	weight = self.weight()
    	integral_array = []
    	for i in range(0,self.n):
    		def exp_term(s):
    			return np.exp(-mean_revs[i]*(t-s))
    		integral = spint.quad(exp_term,0,t)[0]
    		integral_array += [integral]
    	integral_array = np.array(integral_array)

    	gn = self.v0 + self.lamda * self.theta * np.sum(weight * integral_array)
    	return gn


    def price(self, strike, spot, texp, cp=1):

    	"""
    	Calculate the price
    	Firstly solve the phi function using discretization scheme in (A11) with a number of time steps N = 300
    	Then get the characteristic function
    	The call prices are computed via the cosine method (Fang and Oosterlee 2008) for the inversion of the characteristic function
    	"""

    	price=None
    	return price

class LiftedHestonMc(LiftedHeston):

	def set_mc_params(self, n_path=1000, rn_seed=None):
		"""
		Set MC parameters

        Args:
            n_path: number of paths
            rn_seed: random number seed
		"""

		self.n_path = int(n_path)
		self.rn_seed = rn_seed
		self.rng = np.random.default_rng(rn_seed)
        

	def vo_path(self, texp, N):
		"""
		Simulate one single path of Volatility process and U process of each factor
		simulated according to modified explicit–implicit scheme (A12) and (A13) in hte paper
		"""
		dt = texp/N
		tobs = dt*np.arange(0,N+1)
		zz = self.rng.standard_normal(size=N)

		u_n = np.zeros((self.n, N+1))
		v_n = np.zeros(N+1)

		for i in range(0,N):
			v_n[i] = self.gn(tobs[i]) + np.sum(self.weight()*u_n[:,i])
			u_n[:,i+1] = (1/(1+dt*self.mean_rev_speed()))* (u_n[:,i] - self.lamda * v_n[i] *dt + self.vov * np.sqrt( np.fmax(v_n[i],0) * dt)*zz[i])
		v_n[N] = self.gn(tobs[N]) + np.sum(self.weight()*u_n[:,N])
		
		return u_n, v_n

	def vo_paths(self, texp, N):
		"""
		Montecarlo Simulation of paths of Volatility process and U process of each factor
		simulated according to modified explicit–implicit scheme (A12) and (A13) in hte paper
		"""
		dt = texp/N
		tobs = dt*np.arange(0,N+1)
		zz = self.rng.standard_normal(size=(self.n_path,N))

		u_n = np.zeros((self.n_path, self.n, N+1))
		v_n = np.zeros((self.n_path, N+1))

		for i in range(0,N):
			v_n[:,i] = self.gn(tobs[i]) + np.sum(self.weight()*u_n[:,:,i],1)
			u_n[:,:,i+1] = (1/(1+dt*self.mean_rev_speed()))* (u_n[:,:,i] - self.lamda * v_n[:,i][:,None] *dt + self.vov * np.sqrt( np.fmax(v_n[:,i][:,None],0) * dt)*zz[:,i][:,None])

		v_n[:, N] = self.gn(tobs[N]) + np.sum(self.weight()*u_n[:,:,N],1)
		return (v_n, zz)    	

	def mc_price(self, strike, spot, texp, N, cp=1):

		"""
		Calculate option prices
		Parameters
		----------
		strike: strike prices
		spot: spot prices
		texp: time to expiration
		N: time steps
        ----------
		"""

		dt = texp/N
		zz = self.vo_paths(texp, N)[1]
		zz2 = np.random.default_rng(self.rn_seed-1).standard_normal(size=(self.n_path,N))
		ww = self.rho * zz + np.sqrt(1 - np.power(self.rho,2))* zz2
		st = np.ones(self.n_path)*spot

		for i in range(0,self.n):
			st = st + st * np.sqrt(np.fmax(self.vo_paths(texp, N)[0][:,i],0)*dt) * ww[:,i]



		price = np.fmax(np.mean(st)-strike,0)
		return price






