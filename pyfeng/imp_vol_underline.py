import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
import pyfeng as pf
import pandas as pd
import datetime
import time

#import exchange option price
ex_opt_prc = np.array(pd.read_excel(r'D:\百度云同步盘\22年10月-ASP项目\历史走势.xlsx'))
ex_opt_prc0919 = ex_opt_prc[2]

ex_opt_prc0919 = np.delete(ex_opt_prc0919,[0])
print(ex_opt_prc0919)

######################################################################
#import spot
# undline_spot = np.array(pd.read_excel(r'D:\百度云同步盘\22年10月-ASP项目\500ETF_close_price.xlsx'))
# undline_spot_idx = undline_spot[:,0]
#
# dates=[]
# for i in range(len(undline_spot_idx)):
#      print(undline_spot_idx[i].strftime('%Y-%m-%d'))
#      dates.append(undline_spot_idx[i].strftime('%Y-%m-%d'))
#
# dates=np.array(dates)
#
# StartIdx = np.array(np.where(dates=='2022-09-19'))
#
# Start = StartIdx[0][0]
#
# SpotCut = undline_spot[Start:-1,1]
spot = np.ones(11)*5.919
print(spot)
##################################################################


# impvol = pf.Bsm.impvol(price_in, strike, spot, texp, cp=1)

m0 = pf.HestonFft(sigma=0.04, vov=0.8, rho=-0.7, mr=0.5, theta=0.04, intr=0.03)#用这个




# input exchange option price

# calculate different strike for each option
StrikeRatio = [1, 0.975, 0.95, 0.9, 0.8, 0.6, 1.025, 1.05, 1.1, 1.2, 1.3]


# strike = StrikeRatio * opt_price
strike = ex_opt_prc0919 * StrikeRatio

print(strike)


vol = m0.vol_smile(strike, spot, texp=1)
plt.plot(strike, vol)
plt.grid()