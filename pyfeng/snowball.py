import numpy as np
from abc import abstractmethod
import pyfeng as pf

class BaseModel:
    def __init__(self):
        self.intr = 0.0273
        self.divr = 0
        self.sigma = 0.22
        self.texp = 2

    @abstractmethod
    def stock_price(self, spot, dt, n_path, n_days):
        pass

    def option_price(self, strike, spot, dt, n_path, n_days, cp):
        st_path = self.stock_price(spot, dt, n_path, n_days)
        if cp == 1:
            price = np.mean(np.fmax(st_path[:,-1] - strike, 0) * np.exp(-self.intr * self.texp))
        elif cp == 0:
            price = np.mean(np.fmax(strike - st_path[:,-1], 0) * np.exp(-self.intr * self.texp))
        else:
            raise ValueError('cp should be 1 (call) or 0 (put)')
        return price

    def implied_vol(self, strike, spot, dt, n_path, n_days, cp):
        option_price = self.option_price(strike, spot, dt, n_path, n_days, cp)
        bs_model = pf.Bsm(self.sigma, self.intr, self.divr)
        impvol = bs_model.impvol(option_price, strike, spot, self.texp, cp)
        return impvol

class BSMC(BaseModel):
    def __init__(self):
        super().__init__()

    def stock_price(self, spot, dt, n_path, n_days):
        st_path = np.zeros((n_path, n_days))
        st_path[:, 0] = spot
        Z1 = np.random.normal(size=(n_path, n_days))
        for i in range(n_days-1):
            st_path[:, i+1] = st_path[:, i] * np.exp((self.intr - 0.5 * self.sigma**2) * dt + Z1[:, i] * np.sqrt(dt) * self.sigma)
        return st_path

    # def st_increments(self, n_path, dt):
    #     Z1 = np.random.normal(n_path)
    #     st_increments = np.exp((self.intr - 0.5 * self.sigma ** 2) * dt - Z1 * np.sqrt(dt) * self.sigma)
    #     return st_increments

    def implied_vol(self, strike, spot, dt, n_path, n_days, cp):
        option_price = self.option_price(strike, spot, dt, n_path, n_days, cp)
        bs_model = pf.Bsm(self.sigma, self.intr, self.divr)
        impvol = bs_model.impvol(option_price, strike, spot, self.texp, cp)
        return impvol

class HestonMC(BaseModel):
    vov = 0.09
    rho = 0.1
    kappa = 0.2
    theta = 0.1

    def __init__(self):
        super().__init__()

    def stock_price(self, spot, dt, n_path, n_days):
        """
        St = S0 * exp(mu * - 0.5* sigma**2) * dt + sigma(t) * W1 * sqrt(dt)
        dv(t) =kappa * v(t) * (theta - v(t)) * dt + vov * v(t)^(0.5) * Z1 * sqrt(dt)
        v(t) = sigma(t)**2
        dW(t)dZ(t) = rho * dt
        or, dWt = rho * dZt + (1-rho**2)**0.5 * dXt, where Zt and Xt are independent
        """

        # get RV with relation rho, the number of RV is n_path * n_days
        X1 = np.random.normal(size=(n_path, n_days))
        Z1 = np.random.normal(size=(n_path, n_days))
        W1 = np.zeros((n_path, n_days))
        for i in range(n_path):
            W1[i, :] = self.rho * Z1[i, :] + np.sqrt(1-self.rho**2) * X1[i, :]

        #get sigma_square path and st_path
        v = np.zeros((n_path, n_days))
        v[:, 0] = self.sigma**2
        st_path = np.zeros((n_path, n_days))
        st_path[:, 0] = spot
        for i in range(n_days-1):
            v[:, i+1] = v[:, i] + self.kappa * (self.theta - v[:, i]) * dt + self.vov * np.sqrt(v[:, i] * dt) * Z1[:, i]
            st_path[:, i+1] = st_path[:, i] * np.exp((self.intr - 0.5 * v[:, i])**2 * dt + W1[:, i] * np.sqrt(dt * v[:, i]))
        return st_path

class Snowball:
    """
    texp is the time to maturity (in years)
    coupon_rate is the annual coupon of the snowball product
    bound is a list, should have two parameters, the first is the knock-in bound, and the second is knock-out bond
    model is the model to generate the paths of underlying assets
    """
    texp = 2
    coupon_rate = 0.152
    bound = [0.75, 1.0]
    model = BSMC
    n_path = 30000
    dt = 1 / 365
    n_days = texp * 365
    notional = 10000
    close_peroid = 90

    def __init__(self, model, texp, coupon_rate, bound, n_path=n_path, notional=notional, ko_ovserv_dates=None, ki_ovserv_dates=None):
        self.texp = texp
        self.coupon_rate = coupon_rate
        self.knock_out_bound = bound[1]
        self.knock_in_bound = bound[0]
        self.model = model
        self.n_path = n_path
        self.model.texp = self.texp
        self.notional = notional
        if ko_ovserv_dates != None:
            self.ko_observ_dates = ko_ovserv_dates   # knock out observation dates ... should be a list of integers <= texp * 365
        else:
            self.ko_observ_dates = np.linspace(90, self.n_days, 30, dtype='int')   # roughly approximate
        if ki_ovserv_dates != None:
            self.ki_observ_dates = ki_ovserv_dates   # knock in observation dates ... should be a list of integers <= texp * 365
        else:
            self.ki_observ_dates = np.linspace(90, self.n_days, 1, dtype='int')    # roughly approximate

    def price(self, spot):
        """
        calculate the price of snowball option
        :spot: is the S0 of the underlying asset
        """
        knock_out_price = spot * self.knock_out_bound
        knock_in_price = spot * self.knock_in_bound
        if self.model == BSMC:
            st_path = self.model().stock_price(spot, self.dt, self.n_path, self.n_days)
        elif self.model == HestonMC:
            st_path = self.model().stock_price(spot, self.dt, self.n_path, self.n_days)
        else:
            raise ValueError('Only support BSMC & HestonMC currently...')

        payout = np.zeros(self.n_path)
        payout_t = np.zeros(self.n_path)
        last_ko_t = np.zeros(self.n_path)
        last_ki_t = np.zeros(self.n_path)
        for i in range(self.n_path):
            ki = 0      # knock in
            ko = 0      # knock out
            # last_ki_t = 0
            # last_ki_t = 0
            for j in range(1, self.n_days):
                # knock out
                if (j in self.ko_observ_dates) & (st_path[i, j] > knock_out_price):
                    ko += 1
                    payout[i] = self.coupon_rate / 365 * j
                    payout_t[i] = j
                    last_ki_t[i] = j
                    break
                # knock in
                elif (st_path[i,j] < knock_in_price):
                    ki += 1
                    last_ki_t[i] = j
            if ki > 0:
                # the case in but not out discussed here
                if st_path[i, -1] >= spot:
                    payout[i] = 0
                    payout_t[i] = self.n_days
                elif st_path[i,-1] < spot:
                    payout[i] = (st_path[i, -1]/spot -1) * self.n_days / 365
                    payout_t[i] = self.n_days
            if (ki == 0) & (ko == 0):
                payout[i] = self.coupon_rate * self.n_days / 365
                payout_t[i] = self.n_days
            # price_list[i] = self.notional * (payout + 1) * np.exp(-payout_t / 365 * self.model().intr)

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
        price = np.mean(self.notional * (payout + 1) * np.exp(-payout_t / 365 * self.model().intr))
        return price

if __name__ == '__main__':
    texp = 3
    coupon_rate = 0.3
    bound = [0.75, 1]
    notional = 10000
    snowball = Snowball(HestonMC, texp, coupon_rate, bound, n_path=50000)
    price = snowball.price(5955.52)
    print(price)