# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:17:09 2021

@author: Yuze, Lantian
"""

import pyfeng as pf
import numpy as np

sigma = np.full(5, 0.2)
# cor = np.full((5,5),0)
# for i in range(5):
#     cor[i,i] = 1
weight = [0.05, 0.15, 0.2, 0.25 , 0.35]
# weight = [1/156] *156
intr = 0.05
divr = 0
spot = 100
texp = 1
z = 1

#a =pf.BsmBasketAsianJu2002(sigma, 0.5, weight, intr, divr)
#print(a.price(100,100,texp,1, True))
b=pf.BsmContinuousAsianJu2002(0.05, 0.09, 0)
print(b.price(95,100,texp,1))
print(b.price(100,100,texp,1))
print(b.price(105,100,texp,1))

