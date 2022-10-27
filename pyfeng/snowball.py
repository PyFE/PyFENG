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
import pyfeng as pf

class BSM_model:

    miu = 0.0273
    sigma = 0.3347

    def __init__(self,miu,sigma):
        self.miu = miu
        self.sigma = sigma

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
    sigma = 0.06
    vov = 0.017
    rho = 0.0
    mr = 0.15
    theta = 0.01
    intr = 0.02

    def __init__(self,sigma,vov,rho,mr,theta,intr):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.mr = mr
        self.theta = theta
        self.intr = intr
    def stock_price(self, m1, S_t: np.array, sigma_t, dt):
        """
        @param m1: pf.HestonMcAndersen2008()
        @param S_t: stock price in t
        @param sigma_t: sigma in t
        @param dt: discrete time interval
        @return:
        """
        sigma_tp1, avgvar, *_ = m1.cond_states_step(dt, sigma_t)
        log_rt = m1.draw_log_return(dt, sigma_t, sigma_tp1, avgvar)
        S_tp1 = S_t * np.exp(log_rt)

        return S_tp1, sigma_tp1


    # def stock_price(self,S_t:np.array,sigma_t,dt):
    #     """
    #     dv(t) = kar(theta-v(t))dt + vov*np.sqrt(v(t))dZ_t
    #     dS_t/S_t = miu dt + sigma dB_t
    #     :param S_t: stock price in t
    #     :param dt: discrete time interval
    #     :return: S_t+1
    #     """
    #     Z_t = st.norm.rvs(loc=0, scale=1, size=(len(sigma_t)))  # normal random number array
    #     sigma_tp1 = sigma_t + (4*self.kar*(self.theta-sigma_t**2) - self.vov**2) / (8*sigma_t**2) * dt + self.vov/2 * np.sqrt(dt) * Z_t
    #     S_tp1 = S_t*np.exp((self.miu-0.5*sigma_t**2)*dt+sigma_t*np.sqrt(dt)*Z_t) # S_t+1
    #
    #     return S_tp1, sigma_tp1



class SnowBallOption:
    texp = 2
    nominal_amount = 1000000
    coupon_rate = 0.152
    intr = 0.0273  # China 10y Government Bond annual yield
    bound = [0.75, 1.0]
    model = BSM_model
    n_path = 50000
    n_time = texp * 365
    dt = 1/365

    def __init__(self, texp, nominal_amount, coupon_rate, bound, model, n_path):
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
        self.n_time = texp * 365
        # self.start_date = start_date
        # self.check_knockout_date = check_knockout_date

    def set_model_params(self,**kwargs):
        self.kwargs = kwargs
    def MC_res(self, spot_price):

        df_res = pd.DataFrame(index=range(self.n_path),columns=['knock_out','knock_in','expire date','stock price','discounted payoff'])

        S_0 = np.ones(self.n_path) * spot_price # start price

        S_path = np.zeros(shape=(self.n_path,self.n_time)) # record every path's stock price at each time point
        S_path[:,0] = S_0

        if self.model == BSM_model:
            for i in range(self.n_time-1):
                S_path[:,i+1] = self.model(miu=self.kwargs['miu'], sigma=self.kwargs['sigma']).stock_price(S_path[:,i], self.dt)

        elif self.model == Heston_model:
            m1 = pf.HestonMcAndersen2008(sigma=self.kwargs['sigma'], vov=self.kwargs['vov'], rho=self.kwargs['rho'], mr=self.kwargs['mr'], theta=self.kwargs['theta'], intr=self.kwargs['intr'])
            m1.set_num_params(n_path=self.n_path, dt=self.dt, rn_seed=12345)
            sigma_t = np.ones(self.n_path) * Heston_model.sigma
            for i in range(self.n_time-1):

                S_path[:,i+1], sigma_tp1 = self.model(sigma=self.kwargs['sigma'], vov=self.kwargs['vov'], rho=self.kwargs['rho'], mr=self.kwargs['mr'], theta=self.kwargs['theta'], intr=self.kwargs['intr']).stock_price(m1, S_path[:,i], sigma_t, self.dt)
                sigma_t =  sigma_tp1
        self.stock_path = S_path
        # if knock out
        check_knockout_id = [i for i in range(90,self.texp*365,30)]
        # check_knockout_id = [(i-self.start_date).days for i in self.check_knockout_date] # id of check knock out dates
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
        df_res.loc[whether_knockout != 0, 'discounted payoff'] = (net_return+1) * self.nominal_amount * np.exp((-self.intr * df_res.loc[whether_knockout!=0,'expire date']/365).astype(float))

        # if not knock out and not knock in
        bool_knockin = (S_path<self.knock_in*spot_price) # whether the path knock in each day
        whether_knockin = np.sum(bool_knockin,axis=1) # whether the path has knocked in
        df_res.loc[whether_knockin==0,'knock_in'] = False
        df_res.loc[whether_knockin!=0,'knock_in'] = True
        df_res.loc[(df_res.knock_in==False) & (df_res.knock_out==False),'expire date'] = check_knockout_id[-1] # not knock in and knock out
        df_res.loc[(df_res.knock_in==False) & (df_res.knock_out==False),'stock price'] = S_path[(whether_knockin==0) & (whether_knockout==0),-1]
        df_res.loc[(df_res.knock_in == False) & (df_res.knock_out == False), 'discounted payoff'] = (check_knockout_id[-1]/365 * self.coupon_rate+1) * self.nominal_amount * np.exp(-self.intr * check_knockout_id[-1]/365)

        # if knock in and final price is between spot price and knock level

        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:,-1] > spot_price),'expire date'] = check_knockout_id[-1]
        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] > spot_price), 'stock price'] = S_path[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1]>spot_price), -1]
        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] > spot_price), 'discounted payoff'] = self.nominal_amount * (np.exp(-self.intr * check_knockout_id[-1] / 365))

        # if knock in and final price is smaller than spot price
        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] < spot_price), 'expire date'] = check_knockout_id[-1]
        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] < spot_price), 'stock price'] = S_path[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] < spot_price), -1]
        end_price = S_path[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] < spot_price), -1]
        df_res.loc[(df_res.knock_in == True) & (df_res.knock_out == False) & (S_path[:, -1] < spot_price), 'discounted payoff'] = \
            (end_price/spot_price) * self.nominal_amount * np.exp(-self.intr * check_knockout_id[-1] / 365)

        return df_res

    def price(self,spot_price):
        mc_res = self.MC_res(spot_price)
        price = mc_res['discounted payoff'].mean()
        return price

    # def vol_smile(self, strike, spot_price):
    #     ''''
    #     From the price from self.price() compute the implied vol
    #     Use self.bsm_model.impvol() method
    #     '''
    #     price = self.price(spot_price)
    #     if self.model == BSM_model:
    #        iv = pf.Bsm(sigma=self.model.sigma, intr=self.intr).impvol(price, strike, spot_price, self.texp, cp=1)
    #
    #     elif self.model == Heston_model:
    #         iv = pf.HestonFft(sigma=self.model.sigma,vov=self.model.vov,mr=self.model.mr,theta=self.model.theta,intr=self.model.intr).impvol_brentq(price, strike, spot_price, self.texp, cp=1)
    #
    #     else:
    #         raise ValueError('No support this model')
    #     return iv

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
    coupon_rate = 0.152
    nominal_amount = 100
    bound = [0.75, 1.0]
    model = BSM_model
    n_path = 30000
    dt = 1/365
    strike = [600, 800, 900, 950, 975, 1000, 1025, 1050, 1100, 1200, 1300]
    snowball = SnowBallOption(texp,nominal_amount, coupon_rate, bound, model, n_path)
    snowball.set_model_params(miu = 0.0273, sigma = 0.3347)
    # iv = snowball.vol_smile(strike,spot_price=1000)
    price = snowball.price(spot_price=1000)
    analysis_miu(snowball)
    analysis_sigma(snowball)

    print('finish')