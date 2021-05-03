# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:17:09 2021

@author: Lantian
"""

import pyfeng as pf
import numpy as np

sigma = np.full(5, 0.2)
cor = np.full((5,5),0)
for i in range(5):
    cor[i,i] = 1
weight = [0.05,0.15,0.2, 0.25, 0.35]
intr = 0.1
divr = 0
spot = [100,100,100,100,100]
texp = 3
z = 1

a =pf.Ju2002_Basket_Asian(sigma, 0, weight, intr, divr)
print(a.price(100,100,3,-1))
