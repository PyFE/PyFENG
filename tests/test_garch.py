# -*- coding: utf-8 -*-
"""
Created on 2022/4/19 22:24
@author: jhyu
"""

import unittest
import copy
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf


class TestGarch(unittest.TestCase):

    def test_BaroneAdesi2004(self):
        raise NotImplementedError

    def test_Capriotti2018(self):
        num = 50
        p = np.zeros(num)
        garchmodel = pf.garch_2.GarchCapriotti2018()
        for k in range(num):
            p[k] = garchmodel.price(strike=110, spot=100, texp=1.0)
        var_sims = p.var()
        return p, var_sims


# test = TestGarch()
# a = test.test_Capriotti2018()
