    # -*- coding: utf-8 -*-
"""
Created on 2021/04/27

@author: zc
"""
import math
import numpy as np
import scipy.special as spec
from sympy import *

class SpectralAsian:
    sigma, intr, divr = None, None, None
    '''
    some parameters given in the model
    '''
    def __init__(self, sigma, intr=0.0, divr=0.0):
        self.sigma=sigma
        self.intr=intr
        self.divr=divr
        
    def Whittaker_M(kappa, mu, z):
        M = np.exp(-z/2)*z**(mu+0.5)*spec.hyp1f1(mu-kappa+0.5,1+2*mu,z)
        return M
    
    def Whittaker_W(kappa, mu, z):
        W = np.exp(-z/2)*z**(mu+0.5)*spec.hyperu(mu-kappa+0.5,1+2*mu,z)
        return W
    
    def find_zeros_real(nu):
        eigenval = []
        for n in range(1,math.floor(np.abs(nu)/2)+1+1):
            eigenvalue = np.abs(nu)-2*n+2
            eigenval.append(eigenvalue)
        return np.array(eigenval)
        
    def positive_eigenvalue(eigval):
        eigval_p = [i for i in eigval if i>0]
        eigval_p = np.array(eigval_p)
        return eigval_p
    
    def find_zeros_imag(n,nu):
        p = symbols('p')
        eigenval = []
        for j in range(1,n):
            p_tilde = solve(p*(log(4*p)-1)-2*np.pi*(j+nu/4-1/2),p)
            eigenval.append(p_tilde)
        eigenval=positive_eigenvalue(np.real(np.array(eigenval,dtype=complex)))
        return eigenval
        
        
    def barrier_price(self, strike, spot, bound, n_eigenval=10, texp=None, sigma=None):
        self.tau = sigma ** 2 * texp / 4
        self.nu = 2*(intr-divr)/sigma**2 -1
        self.k = tau * strike / spot
        
        eigenval1 = find_zeros_real(nu=self.nu)
        eigenval2 = find_zeros_imag(n=n_eigenval,nu=self.nu)
        
        # 'eigenval2' gives exactly the same value of that 
        # in the parentheses of Table 2 in the paper
        
        ## To be finished
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            