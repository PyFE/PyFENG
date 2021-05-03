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

a =pf.Ju2002_Basket_Asian(sigma, 0.2, weight, intr, divr)
a.average_s([100,100,100,100,100], 1)
a.average_rho(1)
a.func_d1(1)
