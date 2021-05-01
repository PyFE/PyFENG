# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:08:29 2021

@author: Yuzhe, Lantian
"""
from multiasset import NormBasket
import numpy as np

class Ju2002_Basket_Asian(NormBasket):
    """
        Args:
            sigma: model volatilities of `n_asset` assets. (n_asset, ) array
            cor: correlation. If matrix, used as it is. (n_asset, n_asset)
                If scalar, correlation matrix is constructed with all same off-diagonal values.
            weight: asset weights, If None, equally weighted as 1/n_asset
                If scalar, equal weights of the value
                If 1-D array, uses as it is. (n_asset, )
            intr: interest rate (domestic interest rate)
            divr: vector of dividend/convenience yield (foreign interest rate) 0-D or (n_asset, ) array
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
    def average_s(self, spot, texp):
        a_s = self.weight @ self.spot @ np.exp(self.divr*texp)
        return a_s
    
    def averge_rho(self):
        pass
    
    def u1(self):
        #the first momentum of log normal distribution#
        pass
    
    def u2():
        #the second momentum of log normal distribution#
        pass
    
    def u2_1st_der():
        ##
        pass
    
    def u2_2nd_der():
        
        pass
    
    def u2_3rd_der():
        pass
    
    def ak_bar():
        #calculate the average a, save to self#
        pass
    
    def e_a12_a2():
        pass
    
    def e_a12_a22():
        pass
    
    def e_a13_a3():
        pass
    
    def e_a1_a2_a3():
        pass
    
    def func_a():
        #the function #
        pass
    
    def func_b():
        pass
    
    def func_c():
        pass
    
    def func_d():
        pass
    
    def price(self, strike, spot, texp, cp=1):
        pass