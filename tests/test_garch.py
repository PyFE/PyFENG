# -*- coding: utf-8 -*-
"""
Created on 2022/4/19 22:24
@author: jhyu
"""

import sys
import os

sys.path.insert(0, os.getcwd())
import pyfeng as pf
import numpy as np


def run():
    (a, b, vov, y0, rho) = (0.1, 0.04, 0.6, 0.06, 0.5)
    model = pf.garch_2.GarchCapriotti2018(
        mr=0.1, theta=0.04, vov=0.6, sigma_0=np.sqrt(y0), rho=rho)
    model.n_path = 100
    # cond = model.cond_states(var_0=y0, texp=1.0)
    # print(cond)
    p = model.price(strike=110, spot=100, texp=1.0)
    print(p)


if __name__ == '__main__':
    run()
