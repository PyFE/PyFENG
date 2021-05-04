# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:17:09 2021

@author: Yuze, Lantian
"""

import pyfeng as pf
import numpy as np

sigma = np.full(156, 0.05)
# cor = np.full((5,5),0)
# for i in range(5):
#     cor[i,i] = 1
weight = [1/156] *156
intr = 0.09
divr = 0
spot = 100
texp = 3
z = 1

a =pf.Ju2002_Basket_Asian(sigma, 0, weight, intr, divr)
print(a.price(95,100,texp,1, False))
