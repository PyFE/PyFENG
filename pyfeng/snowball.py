import numpy as np
from abc import abstractmethod
import pyfeng as pf

class BaseModel:
    def __init__(self):
        self.intr = 0
        self.divr = 0
        self.sigma = 0.02
        self.texp = 3

    @abstractmethod
    def stock_price(self, spot, dt, n_path, n_days):
        pass

    @abstractmethod
    def option_price(self, strike, spot, dt, n_path, n_days, cp):
        pass

class BSMC(BaseModel):
    def __init__(self):
        super().__init__()

    def stock_price(self, spot, dt, n_path, n_days):
        st_path = np.zeros((n_path, n_days))
        st_path[:,0] = spot
        Z1 = np.random.normal(size=(n_path, n_days))
        for i in range(n_days-1):
            st_path[:, i+1] = st_path[:, i] * np.exp((self.intr - 0.5 * self.sigma**2) * dt - Z1[:,i] * np.sqrt(dt) * self.sigma)
        return st_path

    # def st_increments(self, n_path, dt):
    #     Z1 = np.random.normal(n_path)
    #     st_increments = np.exp((self.intr - 0.5 * self.sigma ** 2) * dt - Z1 * np.sqrt(dt) * self.sigma)
    #     return st_increments

    def option_price(self, strike, spot, dt, n_path, n_days, cp):
        st_path = self.stock_price(spot, dt, n_path, n_days)
        if cp == 1:
            option_price = np.mean(np.fmax(st_path [:,-1] - strike, 0)) * np.exp(-self.intr * self.texp)
        elif cp == 0:
            option_price = np.mean(np.fmax(strike - st_path [:,-1], 0)) * np.exp(-self.intr * self.texp)
        else:
            raise ValueError('cp should be 1 (call) or 0 (put)')
        return option_price

    def implied_vol(self, strike, spot, dt, n_path, n_days, cp):
        option_price = self.option_price(strike, spot, dt, n_path, n_days, cp)
        bs_model = pf.Bsm(self.sigma, self.intr, self.divr)
        impvol = bs_model.impvol(option_price, strike, spot, self.texp, cp)
        return impvol

class Snowball:
    texp = 3
    coupon_rate = 0.03
    bound = [0.75, 1.0]
    model = BSMC
    n_path = 30000
    dt = 1 / 365
    n_days = texp * 365

    def __init__(self, model, texp, coupon_rate, bound, n_path=n_path, ko_ovserv_dates=None, ki_ovserv_dates=None):
        self.texp = texp
        self.coupon_rate = coupon_rate
        self.knock_out_bound = bound[1]
        self.knock_in_bound = bound[0]
        self.model = model
        self.n_path = n_path
        self.model.texp = self.texp
        if ko_ovserv_dates != None:
            self.ko_observ_dates = ko_ovserv_dates   # knock out observation dates ... should be a list of integers <= texp * 365
        else:
            self.ko_observ_dates = np.linspace(90, self.n_days, 30, dtype='int')   # roughly approximate
        if ki_ovserv_dates != None:
            self.ki_observ_dates = ki_ovserv_dates   # knock in observation dates ... should be a list of integers <= texp * 365
        else:
            self.ki_observ_dates = np.linspace(90, self.n_days, 1, dtype='int')    # roughly approximate

        # def price(self, strike, spot, dt, cp):
        #     s_t = np.ones(n_path) * spot_price
        #     payout = np.zeros(n_path)
        #     payout_t = np.zeros(n_path)
        #
        #     for i in range(n_days):
        #         model.st_increments(n_path, dt)

    def price(self, spot):
        if self.model == BSMC:
            st_path = self.model().stock_price(spot, self.dt, self.n_path, self.n_days)
        else:
            raise ValueError('Only support BSMC currently...')

        payout = np.zeros(self.n_path)
        payout_t = np.zeros(self.n_path)
        for i in range(self.n_path):
            ki = 0      # knock in
            ko = 0      # knock out
            last_ki_t = 0
            last_ki_t = 0
            for j in range(1, self.n_days):
                # knock out
                if (j in self.ko_observ_dates) & (st_path[i,j] > self.knock_out_bound):
                    ko += 1
                    payout[i] = self.coupon_rate / 365 * j
                    payout_t[i] = j
                    last_ko_t = j
                    break
                # knock in
                elif (st_path[i,j] < self.knock_in_bound):
                    ki += 1
                    last_ki_t = j
            if ki > 0:
                # the case in but not out discussed here
                if st_path[i,-1] >= spot:
                    payout[i] = 0
                    payout_t[i] = self.n_days
                elif st_path[i,-1] < spot:
                    payout[i] = (st_path[i,-1]/spot -1) * self.n_days / 365
                    payout_t[i] = self.n_days
            if (ki == 0) & (ko == 0):
                payout[i] = self.coupon_rate * self.n_days / 365
                payout_t[i] = self.n_days

        # knock_in_flag = np.zeros((n_path, len(self.ko_observ_dates)))
        # knock_out_flag = np.zeros((n_path, len(self.ki_observ_dates)))
        # st_path = self.model.stock_price(strike, spot, dt, cp)
        # = =||did't figure out a elegant way..maybe try it later, first just use for.. for...
        # knock_out_flag[st_path[:, self.ko_observ_dates[0]:] > self.knock_out_bound] = 1
        # knock_in_flag[st_path[:, self.ki_observ_dates[0]:] < self.knock_in_bound] =1
        # # record the first ko time id of each st path, if never reached ko barrier then the return -1 (ko: knock out)
        # ko_mask = knock_out_flag != 0
        # ko_date_id = np.where(ko_mask.any(axis=1), knock_out_flag.argmax(axis=1), -1*np.ones(np.shape(knock_out_flag.argmax(axis=1))))
        # # record the first ki time of each st path

        price = np.mean(payout * np.exp(-payout_t * self.model().intr))
        return price


if __name__ == '__main__':
    texp = 3
    coupon_rate = 0.3
    bound = [0.75, 1]
    # stock_price = model.stock_price(spot=6482, dt = 1/250)
    # option_price = model.option_price(strike=6482, spot=6482, dt=1/250, cp=1)
    # print(option_price)
    # print(stock_price)
    # implied_vol = model.implied_vol(strike=6482, spot=6482, dt=1/250, cp=1)
    # option_price_pf = pf.Bsm(sigma=0.22, intr=0, divr=0).price(spot=6482,strike=6482, texp=3, cp=0)
    # print(option_price_pf)
    snowball = Snowball(BSMC, texp, coupon_rate, bound, n_path=10000)
    price = snowball.price(100)
    print(price)