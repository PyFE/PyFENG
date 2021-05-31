# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:29:04 2021

@author: cy-wang15 / Zay
"""

import math
import numpy as np
import mpmath as m
from scipy.misc import derivative
from sympy import *


class BsmAsianLinetsky2004:
    
    def __init__(self, intr, divr, vol, texp, strike, spot, b, call):
        self.intr = intr
        self.divr = divr
        self.vol = vol
        self.texp = texp
        self.strike = strike
        self.spot = spot
        self.b = b
        self.call = call
        self.nu = 2*(self.intr-self.divr)/(self.vol**2)-1
        self.tau = self.vol**2*self.texp/4
        self.k = self.tau*self.strike/self.spot
        
    def find_zeros_real(self,nu):
        eigenval = []
        if nu <= -2:
            for n in range(2,math.floor(np.abs(nu)/2)+1+1):
                eigenvalue = np.abs(nu)-2*n+2
                eigenval.append(eigenvalue)
        else:
            pass
        return np.array(eigenval)

    def positive_eigenvalue(self,eigval):
        eigval_p = []
        for i in range(len(eigval)):
            real_i = np.array(eigval[i],dtype=complex).real[0]
            if real_i > 0:
                eigval_p.append(real_i)
            else:
                pass
        return eigval_p

    def find_zeros_imag(self,n,nu):
        p = symbols('p')
        eigenval = []
        for j in range(1,n):
            p_tilde = solve(p*(log(4*p)-1)-2*np.pi*(j+nu/4-1/2),p)
            eigenval.append(p_tilde)
        eigenval = self.positive_eigenvalue(np.real(eigenval))
        eigenval = np.array(eigenval).transpose()
        return eigenval
    
    def eta_q(self,nu,eigenval,b):
        f = lambda x: m.whitw((1-nu)/2,x/2,1/(2*b))
        return complex(-derivative(f,eigenval,dx=1e-12))

    def xi_p(self,nu,eigenval,b):
        f = lambda x: m.whitw((1-nu)/2,complex(0,x/2),1/(2*b))
        return complex(derivative(f,eigenval,dx=1e-12))
        
    def price_element_imag(self,nu,tau,p_value):
        p1 = m.exp(-(nu**2+p_value**2)*tau/2)
        p2 = (p_value*m.gamma(complex(nu/2,p_value/2)))/(4*self.xi_p(nu=nu,eigenval=p_value,b=self.b)*m.gamma(complex(1,p_value)))
        const = (2*self.k)**((nu+3)/2)*m.exp(-1/(4*self.k))
        w1 = m.whitw(-(nu+3)/2,complex(0,p_value/2),1/(2*self.k))
        m1 = m.whitm((1-nu)/2,complex(0,p_value/2),1/(2*self.b))
        return p1 * p2 * const * w1 * m1
        
    def price_element_real(self,nu,tau,q_value):
        p1 = m.exp(-(nu**2-q_value**2)*tau/2)
        p2 = (q_value*m.gamma((nu+q_value)/2))/(4*self.eta_q(nu=nu,eigenval=q_value,b=self.b)*m.gamma(1+q_value))
        const = (2*self.k)**((nu+3)/2)*m.exp(-1/(4*self.k))
        w1 = m.whitw(-(nu+3)/2,q_value/2,1/(2*self.k))
        m1 = m.whitm((1-nu)/2,q_value/2,1/(2*self.b))
        return p1 * p2 * const * w1 * m1
        
    def exact_asian_price(self,n_eig):
        intr = self.intr
        divr = self.divr
        vol = self.vol
        texp = self.texp
        strike = self.strike
        spot = self.spot
        b = self.b
        call = self.call
        nu = self.nu
        tau = self.tau
        k = self.k
        p = self.find_zeros_imag(n=n_eig+1,nu=nu)
        q = self.find_zeros_real(nu=nu)
        
        imaginary_terms = []
        for i in range(len(p)):
            imaginary_term = self.price_element_imag(nu=nu,tau=tau,p_value=p[i])
            imaginary_terms.append(complex(imaginary_term))
        
        real_terms = []
        for i in range(len(q)):
            real_term = self.price_element_real(nu=nu,tau=tau,q_value=q[i])
            real_terms.append(complex(real_term))
            
        P = np.sum(imaginary_terms)+np.sum(real_terms)
        put_price = np.exp(-intr*texp)*(4*spot/(texp*vol**2))* P
        call_price = put_price + (spot*(np.exp(-divr*texp)-np.exp(-intr*texp))/((intr-divr)*texp) - strike*np.exp(-intr*texp)) 
        
        if call == False:
            return(put_price.real)
        elif call == True:
            return(call_price.real)
        else:
            return("Wrong")
         
    
    
