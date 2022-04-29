
import numpy as np
import scipy.integrate as spint
import scipy.special as scsp


def F(u,v,ρ,λ,ν):
	ans=0.5*(u*u-u)+(u*ρ*ν-λ)*v+0.5*ν*ν*v*v
	return ans
def χ_k(k,c,d,a,b):
	sincos_1=k*np.pi*(d-a)/(b-a)
	sincos_2=k*np.pi*(c-a)/(b-a)
	ans=np.cos(sincos_1)*np.exp(d)-np.cos(sincos_2)*np.exp(c)
	ans+=k*np.pi/(b-a)*(np.sin(sincos_1)*np.exp(d)-np.sin(sincos_2)*np.exp(c))
	ans/=(1+pow(k*np.pi/(b-a),2))
	return ans
def ψ_k(k,c,d,a,b):
	sincos_1=k*np.pi*(d-a)/(b-a)
	sincos_2=k*np.pi*(c-a)/(b-a)
	if (k==0):
		ans=d-c
	else:
		ans=(np.sin(sincos_1)-np.sin(sincos_2))*(b-a)/k/np.pi
	return ans
def U_k(k,a,b,call=1):
	if (call==1):
		ans=(χ_k(k,0,b,a,b)-ψ_k(k,0,b,a,b))*2/(b-a)
	else:
		ans=(-χ_k(k,a,0,a,b)+ψ_k(k,a,0,a,b))*2/(b-a)
	return ans


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
    	alpha = self.H + 1/2
    	gamma = scsp.gamma(alpha)*scsp.gamma(2-alpha)

    	for i in range(1,self.n+1):    		
    		cn_i = (np.power(self.rn,1-alpha) -1) * np.power(self.rn,(alpha-1)*(1+self.n/2)) * np.power(self.rn,i*(1-alpha))/gamma
    		cn_v += [cn_i]
    	return np.array(cn_v)

    def mean_rev_speed(self):

    	"""
    	Generate xn from known parameters
    	according to equation (19) in the paper
    	"""

    	xn_v = []
    	for i in range(1,self.n+1):
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
    		integral = (1-np.exp(-mean_revs[i]*t))/mean_revs[i]
    		integral_array += [integral]
    	integral_array = np.array(integral_array)

    	gn = self.v0 + self.lamda * self.theta * np.sum(weight * integral_array)
    	return gn


    def ψ(self,t,u,ρ,λ,ν,steps):
    	"""
    	Explicit-implicit discritization scheme, see (A11) in the paper
    	"""
    	Δ=t/steps
    	ψ_series=np.zeros((self.n, steps+1))
    	ψ_series=np.array(ψ_series,dtype=complex)

    	for i in range(1,steps+1):
    		weighted_ψ = np.sum(self.weight() * ψ_series[:,i-1])
    		ψ_series[:,i] = (ψ_series[:,i-1] + Δ* F(u,weighted_ψ,ρ,λ,ν))/(1+self.mean_rev_speed()*Δ)
    	return ψ_series[:,-1]


    def characteristic_func(self,u,T,ρ,λ,ν,V0,θ,steps,intervals):
    	"""
    	Given x=log(S0/K), y=log(ST/K)
    	generate a function without S0 about the characteristic function of f(y|x)from equation (16) in the paper
    	it is actually E(e^uy)/(e^ux)
    	u is a complex, u=wi
    	"""
    	ds=T/intervals
    	integral_value=0

    	for each in range(intervals):
    		left_s=ds*each
    		right_s=left_s+ds
    		ψ_right =self.ψ(right_s,u,ρ,λ,ν,steps)
    		ψ_left =self.ψ(left_s,u,ρ,λ,ν,steps)

    		integral_left=F(u,np.sum(self.weight()*ψ_left),ρ,λ,ν)*self.gn(T-left_s)
    		integral_right=F(u,np.sum(self.weight()*ψ_right),ρ,λ,ν)*self.gn(T-right_s)
    		integral_value+=(integral_left+integral_right)*ds/2
    		ψ_left=ψ_right
    	if integral_value<=0:
    		ans = np.exp(integral_value)
    	else:
    		ans = 1/np.exp(-integral_value)

    	return ans

    def price(self, strike, spot, texp, cp=1):

    	"""
    	Calculate the price
    	Firstly solve the phi function using discretization scheme in (A11) with a number of time steps 60
    	Then get the characteristic function
    	The call prices are computed via the cosine method (Fang and Oosterlee 2008) for the inversion of the characteristic function

    	"strike" needs to be array
    	"""

    	N=160
    	n_strike=strike.size
    	price_list=np.zeros(n_strike,dtype=np.float64)

    	for j in range(n_strike):
    		price=0
    		x=np.log(spot/strike[j])
    		a=x-1.5
    		b=x+1.5

    		check_end=np.zeros(5)
    		for i in range(N):
    			char_func=self.characteristic_func(complex(0,i*np.pi/(b-a)),texp,self.rho,self.lamda,self.vov,self.v0,self.theta,steps=60,intervals=70)
    			price_add=char_func*U_k(i,a,b,1)*np.exp(complex(0,i*np.pi*(x-a)/(b-a)))
    			if (i==0):
    				price+=price_add.real/2
    			else:
    				price+=price_add.real

    			flag=False
    			if (i<5):
    				check_end[i]=price*strike[j]
    			else:
    				for i_temp in range(4):
    					check_end[i_temp]=check_end[i_temp+1]
    					check_end[4]=price*strike[j]
    					check_min=check_end[0]
    					check_max=check_min
    				for i_temp in range(5):
    					check_min=min(check_min,check_end[i_temp])
    					check_max=max(check_max,check_end[i_temp])
    				if (check_max-check_min)<0.005*spot/100:
    					flag=True
    			if (flag==True):
    				break
    		price_list[j]=check_max

    	return price_list


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
		#	u_n[:,i+1] = (1-dt*self.mean_rev_speed())* u_n[:,i] - self.lamda * v_n[i] *dt + self.vov * np.sqrt( np.fmax(v_n[i],0) * dt)*zz[i]
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
			u_n[:,:,i+1] = (1/(1+dt*self.mean_rev_speed()))*(u_n[:,:,i] - self.lamda * v_n[:,i][:,None] *dt + self.vov * np.sqrt( np.fmax(v_n[:,i][:,None],0) * dt)*zz[:,i][:,None])
		#	u_n[:,:,i+1] = (1-dt*self.mean_rev_speed())* u_n[:,:,i] - self.lamda * v_n[:,i][:,None] *dt + self.vov * np.sqrt( np.fmax(v_n[:,i][:,None],0) * dt)*zz[:,i][:,None]

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
		vo = self.vo_paths(texp, N)[0]
		zz2 = np.random.default_rng(self.rn_seed-1).standard_normal(size=(self.n_path,N))
		ww = self.rho * zz + np.sqrt(1 - np.power(self.rho,2))* zz2
		st = np.ones(self.n_path)*spot

		for i in range(0,N):
			st = st + st * np.sqrt(np.fmax(vo[:,i],0)*dt) * ww[:,i]
		#print(st)

		n_strike=strike.size
		price=np.zeros(n_strike)
		for i in range(n_strike):
			price[i] = np.mean(np.fmax(st-strike[i],0))
		return price






