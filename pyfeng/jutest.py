# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:17:09 2021

@author: Lantian
"""
from . import Ju2002_Basket_Asian as ju
import numpy as np

sigma = np.zeros(5)
cor = np.full((5,5),0.2)
for i in range(5):
    cor[i,i] = 1
weight = [0.05,0.15,0.2, 0.25, 0.35]
intr = np.zeros(5)
divr = np.zeros(5)

a = ju.Ju2002_Basket_Asian(sigma, cor, weight, intr, divr)