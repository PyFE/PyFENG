# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 07:43:01 2022

@author: Jason
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from scipy.special import gamma,roots_jacobi ,roots_legendre
from scipy.integrate import quad, trapz, simps
from functools import reduce


class RoughHeston:
    '''

    '''
    def __init__(self, texp, V0=0, alpha=0.0, kappa=0.3, theta=0.5, nu=0.01, time_steps=1_000):
        self.texp = texp
        self.V0 = V0
        self.alpha = alpha
        self.kappa = kappa
        self.theta = theta
        self.nu = nu               # need to be small to garantee V>0
        self.time_steps = time_steps
               
    def volMEM(self, texp, V0=0, alpha=0.0, kappa=0.3, theta=0.5, nu=0.01, time_steps=1_000):
        '''
        
        '''
        z = np.random.normal(size=(time_steps+1)) 
        V_tk = V0 * np.ones(time_steps+1) 
        tk=np.linspace(0, texp, num=time_steps+1)  

        for i in range(self.time_steps+1):
            fV=0
            gV=0
            for j in range(i):    # k=j+1
                fV = fV + kappa * (theta-V_tk[j]) * ((tk[i]-tk[j])**(1-alpha)-(tk[i]-tk[j+1])**(1-alpha))
            for j in range(i):
                gV = gV+ nu * np.sqrt(V_tk[j]) * np.sqrt(((tk[i]-tk[j])**(1-2*alpha)-(tk[i]-tk[j+1])**(1-2*alpha))/(1-2*alpha))*z[j+1]
            V_tk[i] = np.max((V_tk[0] + 1/gamma(2.- alpha) * fV + 1/gamma(1.- alpha) * gV),0)      
        return V_tk

        #I = simps(sigma_tk * sigma_tk, dx=texp/self.time_steps) / (self.sigma**2)  # integrate by using Simpson's rule
        
    def volFast(self, texp, n_exp=10, V0=0, alpha=0.0, kappa=0.3, theta=0.5, nu=0.01, time_steps=1_000):    
        
        '''
        '''
        tau = texp/time_steps
        xi = 0.0001 
        M = 0.5*np.log(texp)//1  #Assume O(N)=0.5(N)
        #Gauss quadrature parameters
        scale = 1
        n_quad = 2* n_exp
        #Gauss-Jacobi quadrature on the interval [0, 2^(−M)]
        x, w = roots_jacobi(n_quad, alpha=1, beta=1.5)
        x *= scale
        w /= w.sum()
        tar=np.intersect1d(np.argwhere(x>=0),np.argwhere(x<=2**(-M)))
        s0 = x[tar]
        w0 = w[tar]
        #nodes and weights for the ns-point Gauss-Legendre quadrature on the small interval [2^j,2^(j+1)], j =−M, ... , −1
        x, w = roots_legendre(n_quad)

        x *= scale
        w /= w.sum()
        tar=np.intersect1d(np.argwhere(x>2**(-M)),np.argwhere(x<=1))
        sjns = x[tar]
        wjns = w[tar]
        #nodes and weights for the nl-point Gauss-Legendre quadrature on the small interval [2^j,2^(j+1)], j = 0, ... , N
        tar=np.intersect1d(np.argwhere(x>1),np.argwhere(x<=2**(time_steps+1)))
        sjnl = x[tar] 
        wjnl = w[tar]
    
        #xl,wl
        xl = reduce(np.union1d,[s0,sjns,sjnl])
        wl = reduce(np.union1d,[w0,sjns**(alpha-1)*wjns,sjnl**(alpha-1)*wjnl])/gamma(alpha)
        N_exp = xl.size
        
       # N = (np.log(np.log(1.0/xi)) + np.log(time_steps / texp))//1           *0.5
       # N_exp = np.log(1.0/xi) * ( np.log(np.log(1.0/xi)) + np.log(time_steps)) \
        #+ np.log(time_steps / texp)*( np.log(np.log(1.0/xi)) * np.log(time_steps / texp))   *0.5

        z = np.random.normal(size=(time_steps+1)) 
        V_tk = V0 * np.ones(time_steps+1) 
        H_tn =np.zeros([time_steps+1,N_exp])
        J_tn =np.zeros([time_steps+1,N_exp])
        #tk=np.linspace(0, texp, num=time_steps+1)  

        for i in range(1,self.time_steps+1):
            #HJ matrix update by row
            for j in range(N_exp):  #perhaps no need to j loop
                if i>1:
                    H_tn[i-1,j] = kappa * (theta-V_tk[i-2])/xl[j]*(1-np.exp(-xl[j]*tau)) + np.exp(-xl[j]*tau)*H_tn[i-2,j]
                
            for j in range(N_exp):
                if i>1:
                    J_tn[i-1,j] = np.exp(-xl[j]*tau)* nu * np.sqrt(V_tk[i-2]*tau)*z[i-1] + np.exp(-xl[j]*tau)*J_tn[i-2,j]
                    
            
            V_tk[i] = np.max((V_tk[0] + tau**(1.-alpha)/gamma(2.- alpha) * kappa * (theta-V_tk[i-1])\
                      + 1/gamma(1.- alpha) * wl*np.exp(-xl*tau)@H_tn[i-1,:]      \
                      + tau**(0.5 - alpha)/gamma(1.- alpha)* nu * np.sqrt(V_tk[i-1])* z[i] \
                      + 1/gamma(1.- alpha) * wl*np.exp(-xl*tau)@J_tn[i-1,:]      ),0) 
            
        return V_tk
    
    
    def volMF(self, texp, n_exp=10, V0=0, alpha=0.0, kappa=0.3, theta=0.5, nu=0.01, time_steps=1_000):    
        '''
        '''
        tau = texp/time_steps
        eta = np.linspace(0,n_exp,n_exp+1)*n_exp**(-0.2)/texp*(np.sqrt(10)*alpha/(2+alpha))**0.4
        gammaj = np.zeros(n_exp+1) 
        c = np.zeros(n_exp+1)
        
        f = lambda x: x**(alpha-1)/(gamma(1-alpha)*gamma(alpha))
        g = lambda x: x**alpha/(gamma(1-alpha)*gamma(alpha))

        
        for j in range(1,n_exp+1):
            c[j] = quad(f,eta[j-1],eta[j])[0]
            gammaj[j] = 1/c[j]*quad(g,eta[j-1],eta[j])[0]
        
        c = np.delete(c,0,0)
        gammaj = np.delete(gammaj,0,0)
        eta = np.delete(eta,0,0)
        
        z = np.random.normal(size=(time_steps+1)) 
        V_tk = V0 * np.ones(time_steps+1) 
        V_tk_j = np.zeros([time_steps+1,n_exp])
        tk=np.linspace(0, texp, num=time_steps+1)  

        for i in range(1,self.time_steps+1):
                        
            V_tk_j[i,:] = (V_tk_j[i-1,:] - kappa *tau * V_tk[i-1] + nu * np.sqrt(V_tk[i-1] * tau) * z[i]) / (1 + tau * gammaj )
            
            V_tk[i] = np.max((V_tk[0] + kappa * theta * c/gammaj @ (1-np.exp(-tk[i]*gammaj)) + c @  V_tk_j[i,:]  ),0) 
        
        return V_tk