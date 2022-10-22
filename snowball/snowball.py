"""
# File       : snowball.py
# Time       ：2022/10/21 10:36
# Author     ：WuHao
# version    ：python 3.7
# Description：Pricing Snow Ball Option
"""

import numpy as np
import pandas as pd
import scipy.stats as st
from datetime import datetime
import matplotlib.pyplot as plt

class BSM_model:

    miu = 0.00
    sigma = 0.1976

    def stock_price(self,S_t:np.array,dt):
        """
        in BSM model, we assume stock price is geometry Brownian motion
        dS_t/S_t = miu dt + sigma dB_t
        S_t+1 = S_t * exp((miu-0.5*sigma^2)dt + sigma * dB_t)

        :param S_t: stock price in t
        :param dt: discrete time interval
        :return: S_t+1
        """
        Z_t = st.norm.rvs(loc=0,scale=1,size=(len(S_t))) # normal random number array
        S_tp1 = S_t*np.exp((self.miu-0.5*self.sigma**2)*dt+self.sigma*np.sqrt(dt)*Z_t) # S_t+1

        return S_tp1

class Heston_model:
    kar = 0.1
    theta = 0.1
    vov = 0.05
    miu = 0
    sigma0 = 0.2


    def stock_price(self,S_t:np.array,sigma_t,dt):
        """
        dv(t) = kar(theta-v(t))dt + vov*np.sqrt(v(t))dZ_t
        dS_t/S_t = miu dt + sigma dB_t
        :param S_t: stock price in t
        :param dt: discrete time interval
        :return: S_t+1
        """
        Z_t = st.norm.rvs(loc=0, scale=1, size=(len(sigma_t)))  # normal random number array
        sigma_tp1 = sigma_t + (4*self.kar*(self.theta-sigma_t**2) - self.vov**2) / (8*sigma_t**2) * dt + self.vov/2 * np.sqrt(dt) * Z_t
        S_tp1 = S_t*np.exp((self.miu-0.5*sigma_t**2)*dt+sigma_t*np.sqrt(dt)*Z_t) # S_t+1

        return S_tp1, sigma_tp1


class SnowBallOption:
    texp = 2
    nominal_amount = 1000000
    coupon_rate = 0.157
    intr = 0.02746  # China 10y Government Bond annual yield
    bound = [0.75, 1.0]
    model = BSM_model
    n_path = 30000
    n_time = texp * 365
    dt = 1/365

    def __init__(self, texp, nominal_amount, coupon_rate, bound, model, n_path, n_time, start_date, check_knockout_date):
        """
        :param texp: maturity
        :param coupon_rate: risk-free rate
        :param nominal_amount: size of the snow ball option
        :param bound: knock in and out level
        :param model: BSM & Heston
        :param n_path: simulate stock prices process paths
        :param n_time: discrete time number
        :param start_date: start date of the snow ball option
        :param check_knockout_date: check knock out dates
        """


        self.texp = texp
        self.model = model
        self.nominal_amount = nominal_amount
        self.coupon_rate = coupon_rate
        self.knock_out = bound[1]
        self.knock_in = bound[0]
        self.n_path = n_path
        self.n_time = n_time
        self.start_date = start_date
        self.check_knockout_date = check_knockout_date

    def MC_res(self, spot_price):

        df_res = pd.DataFrame(index=range(self.n_path),columns=['knock_out','knock_in','expire date','stock price','discounted payoff'])

        S_0 = np.ones(self.n_path) * spot_price # start price

        S_path = np.zeros(shape=(self.n_path,self.n_time)) # record every path's stock price at each time point
        S_path[:,0] = S_0

        if self.model == BSM_model:
            for i in range(self.n_time-1):
                S_path[:,i+1] = self.model().stock_price(S_path[:,i], self.dt)
        elif self.model == Heston_model:
            sigma_t = np.ones(self.n_path) * Heston_model.sigma0
            for i in range(self.n_time-1):
                S_path[:,i+1], sigma_tp1 = self.model().stock_price(S_path[:,i], sigma_t, self.dt)
                sigma_t =  sigma_tp1

        # if knock out
        check_knockout_id = [(i-self.start_date).days for i in self.check_knockout_date] # id of check knock out dates
        check_knockout_price = S_path[:,check_knockout_id] # underlying price at observed knockout date
        bool_knockout = (check_knockout_price>self.knock_out*spot_price) # whether it knock out at each observed date
        whether_knockout = np.sum(bool_knockout,axis=1) # whether the path has knocked out

        df_res.loc[whether_knockout==0,'knock_out'] = False # path has not knocked out
        df_res.loc[whether_knockout!=0,'knock_out'] = True # path has knocked out

        for i in df_res.loc[:,'knock_out'][whether_knockout!=0].index:
            first_knockout_date = check_knockout_id[np.where(bool_knockout[i,:]==True)[0][0]] # the first date of knock out
            df_res.loc[i,'expire date'] = first_knockout_date # expire date of the path
            df_res.loc[i,'stock price'] = S_path[i,first_knockout_date]
        net_return = df_res.loc[whether_knockout!=0,'expire date']/365 * self.coupon_rate
        df_res.loc[whether_knockout != 0, 'discounted payoff'] = net_return * self.nominal_amount * np.exp((-self.intr * df_res.loc[whether_knockout!=0,'expire date']/365).astype(float))

        # if not knock out and not knock in
        bool_knockin = (S_path<self.knock_in*spot_price) # whether the path knock in each day
        whether_knockin = np.sum(bool_knockin,axis=1) # whether the path has knocked in
        df_res.loc[whether_knockin==0,'knock_in'] = False
        df_res.loc[whether_knockin!=0,'knock_in'] = True
        df_res.loc[(df_res.knock_in==False) & (df_res.knock_out==False),'expire date'] = check_knockout_id[-1] # not knock in and knock out
        df_res.loc[(df_res.knock_in==False) & (df_res.knock_out==False),'stock price'] = S_path[(whether_knockin==0) & (whether_knockout==0),-1]
        df_res.loc[(df_res.knock_in == False) & (df_res.knock_out == False), 'discounted payoff'] = check_knockout_id[-1]/365 * self.coupon_rate  * self.nominal_amount * np.exp(-self.intr * check_knockout_id[-1]/365)

        # if knock in and final price is between spot price and knock level

        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:,-1]>spot_price),'expire date'] = check_knockout_id[-1]
        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1]>spot_price), 'stock price'] = S_path[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1]<spot_price), -1]
        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] > spot_price), 'discounted payoff'] = self.nominal_amount * (np.exp(-self.intr * check_knockout_id[-1] / 365) - 1)

        # if knock in and final price is smaller than spot price
        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] < spot_price), 'expire date'] = check_knockout_id[-1]
        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] < spot_price), 'stock price'] = S_path[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] < spot_price), -1]
        end_price = S_path[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] < spot_price), -1]
        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] < spot_price), 'discounted payoff'] = \
            (end_price/spot_price-1) * self.nominal_amount * np.exp(-self.intr * check_knockout_id[-1] / 365)

        return df_res

    def price(self,spot_price):
        mc_res = self.MC_res(spot_price)
        price = mc_res['discounted payoff'].mean()
        return price

def analysis_sigma(snowball):
    """
    analysis the influence of implied volatility on snow ball price
    :param snowball:
    :return:
    """
    sigma_list = np.arange(0.0,0.5,0.025)
    price_list = []
    for sigma in sigma_list:
        BSM_model.sigma = sigma
        price_list.append(snowball.price(spot_price=1000))
    plt.scatter(x=sigma_list, y=price_list)
    plt.savefig('./IV_analysis.png')
    plt.show()

def analysis_miu(snowball):
    """
    analysis the influence of underlying assets return on snow ball price
    :param snowball:
    :return:
    """
    df = pd.DataFrame(columns=['log return','knock out rate','knock in not out rate','both None rate','win rate','price'])
    miu_list = np.arange(0,0.45,0.05)
    for i,miu in enumerate(miu_list):
        BSM_model.miu = miu
        mc_res = snowball.MC_res(spot_price=1000)
        df.loc[i,'log return'] = miu
        df.loc[i,'knock out rate'] = (mc_res['knock_out'] == True).sum() / len(mc_res)
        df.loc[i,'knock in not out rate'] = ((mc_res['knock_out'] == False) & (mc_res['knock_in'] == True)).sum() / len(mc_res)
        df.loc[i, 'both None rate'] = ((mc_res['knock_out'] == False) & (mc_res['knock_in'] == False)).sum() / len(mc_res)
        df.loc[i,'win rate'] = df.loc[i,'knock out rate'] + df.loc[i, 'both None rate']
        df.loc[i,'price'] = mc_res['discounted payoff'].mean()
    df = df.astype(float)
    df = df.round(4)
    return df






if __name__ == '__main__':
    texp = 2
    coupon_rate = 0.157
    nominal_amount = 1000000
    bound = [0.75, 1.0]
    model = Heston_model
    n_path = 30000
    n_time = texp * 365
    dt = 1/365
    start_date = datetime(2021,8,26)
    check_knockout_date = [datetime(2021,11,26),datetime(2021,12,24),datetime(2022,1,26),
                           datetime(2022,2,25),datetime(2022,3,25),datetime(2022,4,26),
                           datetime(2022,5,26),datetime(2022,6,24),datetime(2022,7,26),
                           datetime(2022,8,26),datetime(2022,9,26),datetime(2022,10,26),
                           datetime(2022,11,25),datetime(2022,12,26),datetime(2023,1,30),
                           datetime(2023,2,24),datetime(2023,3,24),datetime(2023,4,26),
                           datetime(2023,5,26),datetime(2023,6,26),datetime(2023,7,26),datetime(2023,8,25)]
    snowball = SnowBallOption(texp,nominal_amount, coupon_rate, bound, model, n_path, n_time, start_date, check_knockout_date)
    analysis_miu(snowball)
    analysis_sigma(snowball)

    print('finish')