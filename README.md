# PyFENG: [Py]thon [F]inancial [ENG]ineering
[![PyPI version](https://badge.fury.io/py/pyfeng.svg)](https://pypi.org/project/pyfeng/)
[![Documentation Status](https://readthedocs.org/projects/pyfeng/badge/?version=latest)](https://pyfeng.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/pyfeng)](https://pepy.tech/project/pyfeng)

PyFENG provides an implementation of the standard financial engineering models for 
derivative pricing.

## Implemented Models 
  * Black-Scholes-Merton (BSM) and displaced BSM models:
    * Analytic option price, Greeks, and implied volatility.
  * Bachelier (Normal) model
    * Analytic option price, Greeks, and implied volatility.
  * Constant-elasticity-of-variance (CEV) model
    * Analytic option price, Greeks, and implied volatility.
  * Stochastic-alpha-beta-rho (SABR) model
    * Hagan's BSM vol approximation. 
    * Choi & Wu's CEV vol approximation.
    * Analytic integral for the normal SABR.
    * Closed-form MC simulation for the normal SABR.
  * Hyperbolic normal stochastic volatility (NSVh) model
    * Analytic option pricing.
  * Heston model
    * FFT option pricing.
    * Almost exact MC simulation by Glasserman & Kim and Choi & Kwok.
  * Schobel-Zhu (OUSV) model
    * FFT option pricing.
    * Almost exact MC simulation by Choi

## About the Package
* Uses `numpy` arrays as basic datatype so computations are naturally vectorized.
* Purely Python without C/C++ extensisons. 
* Implemented with Python class.
* Intended for academic use. By providing reference models, it saves researchers' time. 
  See [PyFENG for Papers](https://github.com/PyFE/PyfengForPapers) in [Related Projects](#related-projects) below.

## Installation
```sh
pip install pyfeng
```
For upgrade,
```sh
pip install pyfeng --upgrade
```

## Code Snippets
`In [1]:`
```python
import numpy as np
import pyfeng as pf
m = pf.Bsm(sigma=0.2, intr=0.05, divr=0.1)
m.price(strike=np.arange(80, 121, 10), spot=100, texp=1.2)
```
`Out [1]:`
```
array([15.71361973,  9.69250803,  5.52948546,  2.94558338,  1.48139131])
```

`In [2]:`
```python
sigma = np.array([[0.2], [0.5]])
m = pf.Bsm(sigma, intr=0.05, divr=0.1) # sigma in axis=0
m.price(strike=[90, 95, 100], spot=100, texp=1.2, cp=[-1,1,1])
```
`Out [2]:`
```
array([[ 5.75927238,  7.38869609,  5.52948546],
       [16.812035  , 18.83878533, 17.10541288]])
```

## Author
* Prof. [Jaehyuk Choi](https://jaehyukchoi.net/phbs_en) (Peking University HSBC Business School). Email: pyfe@eml.cc

## Related Projects
* Commercial versions (implemented and optimized in C/C++) for some models are available. Email the author at pyfe@eml.cc.
* [PyFENG for Papers](https://github.com/PyFE/PyfengForPapers) is a collection of Jupyter notebooks that reproduce the 
  results of financial engineering research papers using [PyFENG](https://github.com/PyFE/PyFENG).
* [FER: Financial Engineering in R](https://cran.r-project.org/package=FER) developed by the same author.
Not all models in `PyFENG` are implemented in `FER`. `FER` is a subset of `PyFENG`. 
