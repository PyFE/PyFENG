# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:08:29 2021

@author: Yuzhe, Lantian
"""
from . import multiasset
import numpy as np

class Ju2002_Basket_Asian(multiasset.NormBasket): 
    """
        Args:
            sigma: model volatilities of `n_asset` assets. (n_asset, ) array
            rho: correlation. If matrix, used as it is. (n_asset, n_asset)
                If scalar, correlation matrix is constructed with all same off-diagonal values.
            weight: asset weights, If None, equally weighted as 1/n_asset
                If scalar, equal weights of the value
                If 1-D array, uses as it is. (n_asset, )
            intr: interest rate (domestic interest rate)
            divr: vector of dividend/convenience yield (foreign interest rate) 0-D or (n_asset, ) array
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
    def average_s(self, spot, texp):
        #cal the forward price of asset num in the basket
        av_s = np.zeros(len(self.weight))
        for num in range(len(self.weight)):
            av_s[num] = (self.weight[num]*spot[num]*np.exp((self.intr-self.divr[num])*texp))
        self.av_s = av_s
    
    def average_rho(self, texp):
        #cal the rho between asset i and j
        if (np.isscalar(self.rho)):
            self.rho = np.full((len(self.weight),len(self.weight)),self.rho)
        av_rho = np.zeros((len(self.weight),len(self.weight)))
        for i in range(len(self.weight)):
            for j in range(len(self.weight)):
                av_rho[i,j] = self.rho[i,j]*self.sigma[i]*self.sigma[j]*texp
        self.av_rho = av_rho
    
    def u1(self, spot, texp):
        #the first momentum of log normal distribution#
        u1_value = self.weight @ (spot * np.exp((self.intr-self.divr)*texp))
        return u1_value
    
    def u2(self, z):
        #the second momentum of log normal distribution#
        u2_value = 0
        for i in range(len(self.weight)):
            for j in range(len(self.weight)):
              u2_value += self.av_s[i] * self.av_s[j] *np.exp(z*z*self.av_rho[i,j])
        return u2_value
    
    def u2_1st_der(self):
        u2_1st_value = 0
        for i in range(len(self.weight)):
            for j in range(len(self.weight)):
              u2_1st_value += self.av_s[i] * self.av_s[j] *self.av_rho[i,j]
        return u2_1st_value
    
    def u2_2nd_der(self):
        u2_2nd_value = 0
        for i in range(len(self.weight)):
            for j in range(len(self.weight)):
              u2_2nd_value += self.av_s[i] * self.av_s[j] *pow(self.av_rho[i,j],2)
        return u2_2nd_value
    
    def u2_3rd_der(self):
        u2_3rd_value = 0
        for i in range(len(self.weight)):
            for j in range(len(self.weight)):
              u2_3rd_value += self.av_s[i] * self.av_s[j] *pow(self.av_rho[i,j],3)
        return u2_3rd_value
    
    def ak_bar(self):
        #calculate the average a, save to self#
        av_a = self.av_rho @ self.av_s
        self.av_a = av_a
    
    def e_a12_a2(self):
        return 2 * self.av_s @ pow(self.av_a,2)
    
    def e_a12_a22(self):
        value = 0
        for i in range(len(self.weight)):
            for j in range(len(self.weight)):
                value += self.av_a[i]*self.av_s[i]*self.av_rho[i,j]*self.av_a[j]*self.av_s[j]
        value *=8
        value += 2*self.u2_1st_der()*self.u2_2nd_der()
        return value
    
    def e_a13_a3(self):
        return 6* self.av_s @ pow(self.av_a,3)
    
    def e_a1_a2_a3(self):
        value = 0
        for i in range(len(self.weight)):
            for j in range(len(self.weight)):
                value += self.av_s[i]*pow(self.av_rho[i,j],2)*self.av_a[j]*self.av_s[j]
        value *= 6
        return value
    
    def e_a23(self):
        value = 0
        temp = np.zeros((len(self.weight),len(self.weight)))
        for i in range(len(self.weight)):
            for j in range(len(self.weight)):
                temp[i,j] = pow(self.av_s[i],0.5)*self.av_rho[i,j]*pow(self.av_s[j],0.5)
        for i in range(len(self.weight)):
            for j in range(len(self.weight)):
                for k in range(len(self.weight)):
                    value += temp[i,j]*temp[j,k]*temp[k,i]
        value *= 8
        return value
    
    def func_a1(self, z):
        return -pow(z,2)*self.u2_1st_der()/2/self.u2(0)
    
    def func_a2(self, z):
        return 2*pow(self.func_a1(z),2)-pow(z,4)*self.u2_2nd_der()/2/self.u2(0)
    
    def func_a3(self, z):
        return 6*self.func_a1(z)*self.func_a2(z)-4*pow(self.func_a1(z),3)-pow(z,6)*self.u2_3rd_der()/2/self.u2(0)
    
    def func_b1(self, spot, texp, z):
        return pow(z,4)*self.e_a12_a2()/4/pow(self.u1(spot,texp),3)
        
    def func_b2(self, z):
        return pow(self.func_a1(z),2)-self.func_a2(z)/2
    
    def func_c1(self, z):
        return -self.func_a1(z)*self.func_b1(z)
    
    def func_c2(self, spot, texp, z):
        return pow(z,6)*(9*self.e_a12_a22()+4*self.e_a13_a3())/144/pow(self.u1(spot,texp),4)
    
    def func_c3(self, spot, texp, z):
        return pow(z,6)*(4*self.e_a1_a2_a3()+self.e_a23())/48/pow(self.u1(spot,texp),3)
    
    def func_c4(self, z):
        return self.func_a1(z)*self.func_a2(z)-2*pow(self.func_a1(z),3)/3-self.func_a3(z)/6
    
    def func_d1(self, z):
        return 0.5*(6*pow(self.func_a1(z),2)+self.func_a2(z)-4*self.func_b1(z)+2*self.func_b2(z))-1/6*(120*pow(self.func_a1(z),3)-self.func_a3(z)+6*(24*self.func_c1(z)-6*self.func_c2(z)+2*self.func_c3(z)-self.func_c4(z)))
    
    def func_d2(self, z):
        return 0.5*(10*pow(self.func_a1(z),2)+self.func_a2(z)-6*self.func_b1(z)+2*self.func_b2(z))-(128*pow(self.func_a1(z),3)/3-self.func_a3(z)/6+2*self.func_a1(z)*self.func_b1(z)-self.func_a1(z)*self.func_b2(z)+50*self.func_c1(z)-11*self.func_c2(z)+3*self.func_c3(z)-self.func_c4(z))
    
    def func_d3(self, z):
        return 2*pow(self.func_a1(z),2)-self.func_b1(z)-1/3*(88*pow(self.func_a1(z),3)+3*self.func_a1(z)*(5*self.func_b1(z)-2*self.func_b2(z))+3*(35*self.func_c1(z)-6*self.func_c2(z)+self.func_c3(z)))
                     
    def func_d4(self, z):
        return -20*pow(self.func_a1(z),3)/3+self.func_a1(z)*(-4*self.func_b1(z)+self.func_b2(z))-10*self.func_c1(z)+self.func_c2(z)
    
    def price(self, strike, spot, texp, cp=1):
        self.average_s(spot, texp)
        self.averge_rho(texp)
        # to be continue
        return 0