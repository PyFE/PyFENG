import numpy as np
from abc import abstractmethod
import pyfeng as pf

class BaseModel:
    def __init__(self, intr, sigma, texp, n_path):
        self.reset_params()
        self.set_params(intr, sigma, texp, n_path)

    def reset_params(self):
        self.intr = None
        self.sigma = None
        self.texp = None
        self.n_path = 3000

    def set_params(self, intr, sigma, texp, n_path):
        self.intr = intr
        self.sigma = sigma
        self.texp = texp
        self.n_path = n_path

    @abstractmethod
    def stock_price(self):
        pass

    @abstractmethod
    def price(self):
        pass

class BSMC(BaseModel):
    def __init__(self):
        super().__init__()

    def stock_price(self, spot,dt):
        n_days = self.texp / dt
        st_path = np.zeros((self.n_path, n_days))
        st_path[:,0] = spot
        Z1 = np.random.normal(self.n_path, n_days-1)
        for i in range(n_days)-1:
            st_path[:,i+1] = st_path[:, i ] * np.exp((self.intr - 0.5 * self.sigma**2)*self.texp - Z1 * np.sqrt(self.texp) * self.sigma)
        return st_path

    def price(self, strike, spot, dt, cp):
        st_path = self.stock_price(spot, dt)
        if cp == 1:
            price = np.mean(np.fmax(st_path [:,-1] - strike, 0) * np.exp(-self.intr * self.texp))
        elif cp == 0:
            price = np.mean(np.fmax(strike - st_path [:,-1], 0) * np.exp(-self.intr * self.texp))
        return price

    def implied_vol(self, strike, spot, dt, cp):
        price = self.price(strike, spot, dt, cp)
        impvol = pf.Bsm.impvol(price, strike, spot, self.texp,cp)
        return impvol


class Snowball:
    texp = 2
    intr = 0
    coupon_rate = 0.03
    bound = [0.75, 1.0]
    model = None
    n_path = 30000
    dt = 1/250
    n_days = texp / dt

    def __init__(self, texp, coupon_rate, bound, model):
        self.texp = texp
        self.coupon_rate = coupon_rate
        self.knock_out_bound = bound[1]
        self.knock_in_bound = bound[0]
        self.model = model

    def price(self, spot_price):
        s_t = np.ones(n_path) * spot_price
        payout = np.zeros(n_path)
        payout_t = np.zeros(n_path)

        for i in range(n_days):
            s_t = self.model.stock_price(s_t, self.dt)
            ### calculate payout

        price = np.mean(payout * np.exp(-payout_t * self.model.intr))

        return price


if __name__ == '__main__':
    model = BSMC(intr=0, sigma=0.22, texp=3, n_path=1000)