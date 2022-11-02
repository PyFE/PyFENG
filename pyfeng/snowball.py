import numpy as np
from abc import abstractmethod
import pyfeng as pf

class BaseModel:
    def __init__(self, sigma, texp, intr, divr):
        self.intr = intr
        self.divr = divr
        self.sigma = sigma
        self.texp = texp

    @abstractmethod
    def stock_price(self, spot, dt, n_path, n_days):
        pass

    def option_price(self, strike, spot, dt, n_path, n_days, cp):
        st_path = self.stock_price(spot, dt, n_path, n_days)
        if cp == 1:
            price = np.mean(np.fmax(st_path[:, -1] - strike, 0) * np.exp(-self.intr * self.texp))
        elif cp == 0:
            price = np.mean(np.fmax(strike - st_path[:, -1], 0) * np.exp(-self.intr * self.texp))
        else:
            raise ValueError('cp should be 1 (call) or 0 (put)')
        return price

    def implied_vol(self, strike, spot, dt, n_path, n_days, cp):
        option_price = self.option_price(strike, spot, dt, n_path, n_days, cp)
        bs_model = pf.Bsm(self.sigma, self.intr, self.divr)
        impvol = bs_model.impvol(option_price, strike, spot, self.texp, cp)
        return impvol

class BSMC(BaseModel):
    def __init__(self, sigma, texp, intr, divr):
        super().__init__(sigma, texp, intr, divr)

    def stock_price(self, spot, dt, n_path, n_days):
        st_path = np.zeros((n_path, n_days))
        st_path[:, 0] = spot
        np.random.seed(0)
        Z1 = np.random.normal(size=(n_path, n_days))
        for i in range(n_days-1):
            st_path[:, i+1] = st_path[:, i] * np.exp((self.intr - 0.5 * self.sigma**2) * dt + Z1[:, i] * np.sqrt(dt) * self.sigma)
        return st_path

    # def st_increments(self, n_path, dt):
    #     Z1 = np.random.normal(n_path)
    #     st_increments = np.exp((self.intr - 0.5 * self.sigma ** 2) * dt - Z1 * np.sqrt(dt) * self.sigma)
    #     return st_increments

class HestonMC(BaseModel):
    def __init__(self, sigma, vov, rho, kappa, theta, texp, intr, divr):
        super().__init__(sigma, texp, intr, divr)
        self.vov = vov
        self.rho = rho
        self.kappa = kappa
        self.theta = theta

    def stock_price(self, spot, dt, n_path, n_days):
        """
        St = S0 * exp(mu * - 0.5* sigma**2) * dt + sigma(t) * W1 * sqrt(dt)
        dv(t) =kappa * v(t) * (theta - v(t)) * dt + vov * v(t)^(0.5) * Z1 * sqrt(dt)
        v(t) = sigma(t)**2
        dW(t)dZ(t) = rho * dt
        or, dWt = rho * dZt + (1-rho**2)**0.5 * dXt, where Zt and Xt are independent
        """
        # get RV with relation rho, the number of RV is n_path * n_days
        np.random.seed(0)
        X1 = np.random.normal(size=(n_path, n_days))
        np.random.seed(99)
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

    def stock_price_Andersen(self, spot, dt, n_path, n_days):
        model = pf.HestonMcAndersen2008(sigma=self.sigma, vov=self.vov, rho=self.rho, mr=self.kappa, theta=self.theta, intr=self.intr)
        model.set_num_params(n_path=n_path, dt=dt, rn_seed=12345)
        var_t1 = np.full(n_path, self.sigma)
        st_path = np.zeros((n_path, n_days))
        st_path[:, 0] = spot
        for i in range(n_days-1):
            var_t2, avgvar, *_ = model.cond_states_step(dt, var_t1)
            log_rt = model.draw_log_return(dt, var_t1, var_t2, avgvar)
            st_path[:, i+1] = st_path[:, i] * np.exp(log_rt)
            var_t1 = var_t2
        return st_path

class Snowball:
    """
    texp is the time to maturity (in years)
    coupon_rate is the annual coupon of the snowball product
    bound is a list, should have two parameters, the first is the knock-in bound, and the second is knock-out bond
    model is the model to generate the paths of underlying assets
    n_path is the number of paths generated to calculate the price
    n_days is the days to maturity of the product
    notional is nominal price
    dt is the time interval to simulate st
    closure_dates means, at the beginning closer_dates num. of days, knock out and knock will not be observed.
    knock_out_gap is the gap between two knock out observation days
    ko_observ_dates is knock out observation dates
    ki_observ_dates is knock in observation dates
    """

    def __init__(self, model, texp, coupon_rate, bound, n_path=30000, notional=10000, dt=1/365, closure_dates=90, knock_out_gap=30, ko_observ_dates=None, ki_observ_dates=None):
        self.texp = texp
        self.dt = dt
        self.n_days = texp * 365
        self.coupon_rate = coupon_rate
        self.knock_out_bound = bound[1]
        self.knock_in_bound = bound[0]
        self.model = model
        self.n_path = n_path
        self.notional = notional
        self.knock_out_gap = knock_out_gap
        self.closure_dates = closure_dates
        if ko_observ_dates != None:
            self.ko_observ_dates = ko_observ_dates   # knock out observation dates ... should be a list of integers <= texp * 365
        else:
            self.ko_observ_dates = np.arange(start=self.closure_dates, stop=self.n_days, step=30, dtype='int')   # roughly approximate
        if ki_observ_dates != None:
            self.ki_observ_dates = ki_observ_dates   # knock in observation dates ... should be a list of integers <= texp * 365
        else:
            self.ki_observ_dates = np.arange(start=self.closure_dates, stop=self.n_days, step=1, dtype='int')    # roughly approximate

    def set_model_param(self, sigma=0.04, vov=0.5, rho=-0.3, mr=0.5, theta=0.1, intr=0.019155):
        """
        set the parameter for Heston model / BSM model
        """
        if self.model == HestonMC:
            self.model_init = self.model(sigma, vov, rho, mr, theta, self.texp, intr, divr=0)
        elif self.model == BSMC:
            self.model_init = self.model(sigma, self.texp, intr, divr=0)
        else:
            raise ValueError("self.model should be BSMC or HestonMC! something's wrong...")

    def price(self, spot, Andersen=False):
        """
        calculate the price of snowball option
        :spot: is the S0 of the underlying asset
        """
        knock_out_price = spot * self.knock_out_bound
        knock_in_price = spot * self.knock_in_bound
        if self.model == BSMC:
            st_path = self.model_init.stock_price(spot, self.dt, self.n_path, self.n_days)
        elif (self.model == HestonMC) & (Andersen==False):
            st_path = self.model_init.stock_price(spot, self.dt, self.n_path, self.n_days)
        elif (self.model == HestonMC) & (Andersen==True):
            st_path = self.model_init.stock_price_Andersen(spot, self.dt, self.n_path, self.n_days)
        else:
            raise ValueError('Only support BSMC & HestonMC currently...')

        payout = np.zeros(self.n_path)
        payout_t = np.zeros(self.n_path)
        last_ko_t = np.zeros(self.n_path)
        last_ki_t = np.zeros(self.n_path)
        for i in range(self.n_path):
            ki = 0      # knock in
            ko = 0      # knock out
            for j in range(1, self.ko_observ_dates[-1]):
                # knock out
                if (j in self.ko_observ_dates) & (st_path[i, j] > knock_out_price):
                    ko += 1
                    payout[i] = (self.coupon_rate / 365 * j + 1) * self.notional
                    payout_t[i] = j
                    last_ko_t[i] = j
                    break
                # knock in
                elif (st_path[i, j] < knock_in_price):
                    ki += 1
                    last_ki_t[i] = j
            if (ki > 0) & (ko == 0):
                # the case in but not out discussed here
                if st_path[i, self.ko_observ_dates[-1]] >= spot:
                    payout[i] = self.notional
                    payout_t[i] = self.ko_observ_dates[-1]
                elif st_path[i, self.ko_observ_dates[-1]] < spot:
                    payout[i] = st_path[i, self.ko_observ_dates[-1]] / spot * self.notional
                    payout_t[i] = self.ko_observ_dates[-1]
            if (ki == 0) & (ko == 0):
                payout_t[i] = self.ko_observ_dates[-1]
                payout[i] = (self.coupon_rate * payout_t[i] / 365 + 1) * self.notional
        price = np.mean(payout * np.exp(-payout_t / 365 * self.model_init.intr))
        return price

    def price_Method2(self, spot, Andersen=False):
        """
        Add a faster method to calculate price. Avoid the for ... for... loop
        """
        knock_out_price = spot * self.knock_out_bound
        knock_in_price = spot * self.knock_in_bound
        if self.model == BSMC:
            st_path = self.model_init.stock_price(spot, self.dt, self.n_path, self.n_days)
        elif (self.model == HestonMC) & (Andersen == False):
            st_path = self.model_init.stock_price(spot, self.dt, self.n_path, self.n_days)
        elif (self.model == HestonMC) & (Andersen == True):
            st_path = self.model_init.stock_price_Andersen(spot, self.dt, self.n_path, self.n_days)
        else:
            raise ValueError('Only support BSMC & HestonMC currently...')

        payout = np.zeros(self.n_path)
        payout_t = np.zeros(self.n_path)
        for i in range(self.n_path):
            knock_out_id = np.argwhere(st_path[i, :] > knock_out_price)
            knock_in_id = np.argwhere(st_path[i, :] < knock_in_price)
            # if knock out at knock out observation date
            if len(np.intersect1d(knock_out_id, self.ko_observ_dates)) > 0:
                payout_t[i] = np.intersect1d(knock_out_id, self.ko_observ_dates)[0]
                payout[i] = (self.coupon_rate / 365 * payout_t[i] + 1) * self.notional
            # if knock in and never knock out
            elif (len(knock_in_id) > 0) & (len(np.intersect1d(knock_out_id, self.ko_observ_dates)) == 0):
                # if end price is above initial price
                if st_path[i, self.ko_observ_dates[-1]] >= spot:
                    payout_t[i] = self.ko_observ_dates[-1]
                    payout[i] = self.notional
                # if end price is lower than initial price, suffer a loss
                elif st_path[i, -self.ko_observ_dates[-1]] < spot:
                    payout_t[i] = self.ko_observ_dates[-1]
                    payout[i] = st_path[i, self.ko_observ_dates[-1]] / spot * self.notional
            elif (len(knock_in_id) == 0) & (len(np.intersect1d(knock_out_id, self.ko_observ_dates)) == 0):
                payout_t[i] = self.ko_observ_dates[-1]
                payout[i] = (1 + self.coupon_rate) * payout_t[i] / 365 * self.notional
        price = np.mean(payout * np.exp(-payout_t / 365 * self.model_init.intr))
        return price

if __name__ == '__main__':
    texp = 2
    coupon_rate = 0.152
    bound = [0.75, 1]
    notional = 10000
    # snowball = Snowball(BSMC, texp, coupon_rate, bound, n_path=10000)
    # snowball.set_model_param(sigma=0.2225, intr=0.0273)
    # price = snowball.price(5955.52)
    snowball = Snowball(HestonMC, texp, coupon_rate, bound, n_path=10000)
    snowball.set_model_param(sigma=0.05, vov=0.56, rho=-0.2, mr=0.5, theta=0.1, intr=0.019155)
    price = snowball.price(5955.52, Andersen=True)
    print(price)