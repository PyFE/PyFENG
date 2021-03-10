# PyFENG: Python Financial ENGineering
[![PyPI version](https://badge.fury.io/py/pyfeng.svg)](https://pypi.org/project/pyfeng/)

PyFENG is the python implemention of the standard option pricing models in financial engineering.
  * Black-Scholes-Merton (and displaced diffusion)
  * Bachelier (Normal)
  * Constant-elasticity-of-variance (CEV)
  * Stochastic-alpha-beta-rho (SABR)
  * Hyperbolic normal stochastic volatility model (NSVh)

## About the package
* It assumes variables are `numpy` arrays. So the computations are naturally vectorized.
* It is purely in Python (i.e., no C, C++, cython). 
* It is implemented with python class.
* It is intended for, but not limited to, academic use. By providing reference models, it saves researchers' time. 

## Installation
```sh
pip install pyfeng
```
For upgrade,
```sh
pip install --upgrade pyfeng
```

## Code Snippets
`In [1]:`
```python
import numpy as np
import pyfeng as pf
m = pf.Bsm(sigma=0.2, intr=0.05, divr=0.1)
m.price(np.arange(80, 121, 10), 100, 1.2)
```
`Out [1]:`
```
array([15.71361973,  9.69250803,  5.52948546,  2.94558338,  1.48139131])
```

`In [2]:`
```python
sigma = np.array([0.2, 0.3, 0.5])[:, None]
m = pf.Bsm(sigma, intr=0.05, divr=0.1) # sigma in axis=0
m.price(np.array([90, 100, 110]), 100, 1.2, cp=np.array([-1,1,1]))
```
`Out [2]:`
```
array([[ 5.75927238,  5.52948546,  2.94558338],
       [ 9.4592961 ,  9.3881245 ,  6.45745004],
       [16.812035  , 17.10541288, 14.10354768]])
```

## Author
* Prof. [Jaehyuk Choi](https://jaehyukchoi.net/phbs_en) (Peking University HSBC Business School). Email: pyfe@eml.cc

## Others
* See also [FER: Financial Engineering in R](https://cran.r-project.org/package=FER) developed by the same author.
Not all models in `PyFENG` is implemented in `FER`. `FER` is a subset of `PyFENG`. 
