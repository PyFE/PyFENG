# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:17:09 2021

@author: Lantian
"""

import pyfeng as pf
import numpy as np

sigma = np.full(5, 0.05)
cor = np.full((5,5),0.2)
for i in range(5):
    cor[i,i] = 1
weight = [0.05,0.15,0.2, 0.25, 0.35]
intr = 0
divr = np.zeros(5)
spot = [100,100,100,100,100]
texp = 1
z = 1

a =pf.Ju2002_Basket_Asian(sigma, cor, weight, intr, divr)
a.average_s(spot, texp)
a.average_rho(texp)
a.ak_bar()
a.func_a1(z)
a.func_a2(z)
a.func_a3(z)
a.func_b1(spot, texp, z)
a.func_b2(z)
a.func_c1(spot,texp,z)
a.func_c2(spot, texp, z)
a.func_c3(spot, texp, z)
# a.func_d1(1)
