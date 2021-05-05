# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:29:04 2021

@author: cy-wang15 / Zay
"""

import math
import numpy as np
import mpmath as m
import scipy.special as spec
from scipy.misc import derivative
from sympy import *


class BsmAsianLinetsky2004 :
    
    def __init__(self, intr, divr, vol, texp, strike, spot, b, call):
        self.intr = intr
        self.divr = divr
        self.vol = vol
        self.texp = texp
        self.strike = strike
        self.spot = spot
        self.b = b
        self.call = call
        print(spot)
    
        
    def find_zeros_real(self,nu):
        eigenval = []
        if nu <= -2:
            for n in range(1,math.floor(np.abs(nu)/2)+1+1):
                eigenvalue = np.abs(nu)-2*n+2
                eigenval.append(eigenvalue)
        else:
            pass
        return np.array(eigenval)

    def positive_eigenvalue(self,eigval):
        eigval_p = [i for i in eigval if i>0]
        eigval_p = np.array(eigval_p)
        return eigval_p

    def find_zeros_imag(self,n,nu):
        p = symbols('p')
        eigenval = []
        for j in range(1,n):
            p_tilde = solve(p*(log(4*p)-1)-2*np.pi*(j+nu/4-1/2),p)
            eigenval.append(p_tilde)
        eigenval = self.positive_eigenvalue(np.real(eigenval))
        eigenval = eigenval.transpose()
        return np.array(eigenval[0],dtype='complex')
    
    def eta_q(self,nu,eigenval,b):
        f = lambda x: m.whitw((1-nu)/2,x/2,1/(2*b))
        return complex(-derivative(f,eigenval,dx=1e-12))

    def xi_p(self,nu,eigenval,b):
        f = lambda x: m.whitw((1-nu)/2,complex(0,x/2),1/(2*b))
        return complex(derivative(f,eigenval,dx=1e-12))
        
    def exact_asian_price(self,strike,spot,vol,texp,intr,divr,b,call):
        nu = 2*(intr-divr)/(vol**2)-1
        tau = vol**2*texp/4
        k = tau*strike/spot
        p = self.find_zeros_imag(n=51,nu=nu)
        q = self.find_zeros_real(nu=nu)
        
        
        imaginary_terms = []
        for i in range(len(p)):
            p1 = m.exp(-(nu**2+p[i]**2)*tau/2)
            p2 = (p[i]*m.gamma(complex(nu/2,p[i]/2)))/(4*self.xi_p(nu=nu,eigenval=p[i],b=b)*m.gamma(complex(1,p[i])))
            const = (2*k)**((nu+3)/2)*m.exp(-1/(4*k))
            w1 = m.whitw(-(nu+3)/2,complex(0,p[i]/2),1/(2*k))
            m1 = m.whitm((1-nu)/2,complex(0,p[i]/2),1/(2*b))
            imaginary_term = p1 * p2 * const * w1 * m1
            imaginary_terms.append(complex(imaginary_term))
        
        real_terms = []
        for i in range(len(q)):
            p1 = m.exp(-(nu**2-q[i]**2)*tau/2)
            p2 = (q[i]*m.gamma((nu+q[i])/2))/(4*self.eta_q(nu=nu,eigenval=q[i],b=b)*m.gamma(1+q[i]))
            const = (2*k)**((nu+3)/2)*m.exp(-1/(4*k))
            w1 = m.whitw(-(nu+3)/2,q[i]/2,1/(2*k))
            m1 = m.whitm((1-nu)/2,q[i]/2,1/(2*b))
            real_term = p1 * p2 * const * w1 * m1
            real_terms.append(complex(real_term))
            
        P = np.sum(imaginary_terms)+np.sum(real_terms)
        put_price = np.exp(-intr*texp)*(4*spot/(texp*vol**2))* P
        call_price = put_price + (spot*(np.exp(-divr*texp)-np.exp(-intr*texp))/(intr*texp) - strike*np.exp(-intr*texp)) 
        
        if call == False:
            return(put_price)
        elif call == True:
            return(call_price)
        else:
            return("Wrong")
         
    
    
