import numpy as np
from abc import abstractmethod
import pyfeng as pf

class BaseModel:
    def __init__(self):
        self.intr = 0.019155   # 1y CDB bond rate. we don't use the
        self.divr = 0
        self.sigma = 0.04
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
    vov = 0.2
    rho = -0.2
    kappa = 0.5
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

    def stock_price_Andersen(self, spot, dt, n_path, n_days):
        model = pf.HestonMcAndersen2008(sigma=self.sigma, vov=self.vov, rho=self.rho, mr=self.kappa, theta=self.theta, intr=self.intr)
        model.set_num_params(n_path=n_path, dt=dt, rn_seed=12345)
        var_t1 = np.full(n_path, self.sigma)
        st_path = np.zeros((n_path, n_days))
        st_path[:, 0] = spot
        for i in range(n_days-1):
            var_t2, avgvar, *_ = model.cond_states_step(dt, var_t1)
            log_rt = model.draw_log_return(dt, var_t1, var_t2, avgvar)
            st_path[:, i+1] = st_path[:,i] * np.exp(log_rt)
            var_t1 = var_t2
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
    closure_dates = 90
    knock_out_gap = 30

    def __init__(self, model, texp, coupon_rate, bound, n_path=n_path, notional=notional, closure_dates=closure_dates, knock_out_gap=knock_out_gap, ko_ovserv_dates=None, ki_ovserv_dates=None):
        self.texp = texp
        self.coupon_rate = coupon_rate
        self.knock_out_bound = bound[1]
        self.knock_in_bound = bound[0]
        self.model = model
        self.n_path = n_path
        self.model.texp = self.texp
        self.notional = notional
        self.knock_out_gap = knock_out_gap
        self.closure_dates = closure_dates
        if ko_ovserv_dates != None:
            self.ko_observ_dates = ko_ovserv_dates   # knock out observation dates ... should be a list of integers <= texp * 365
        else:
            self.ko_observ_dates = np.linspace(self.closure_dates, self.n_days, 30, dtype='int')   # roughly approximate
        if ki_ovserv_dates != None:
            self.ki_observ_dates = ki_ovserv_dates   # knock in observation dates ... should be a list of integers <= texp * 365
        else:
            self.ki_observ_dates = np.linspace(self.closure_dates, self.n_days, 1, dtype='int')    # roughly approximate

    def set_model_param(self, sigma=0.04, vov=0.5, rho=-0.3, mr=0.5, theta=0.1, intr=0.019155):
        """
        set the parameter for Heston model / BSM model
        """
        self.model.intr = intr
        self.model.sigma = sigma
        if self.model == HestonMC:
            self.model.vov = vov
            self.model.rho = rho
            self.model.kappa = mr
            self.model.theta = theta

    def price(self, spot, Andersen=False):
        """
        calculate the price of snowball option
        :spot: is the S0 of the underlying asset
        """
        knock_out_price = spot * self.knock_out_bound
        knock_in_price = spot * self.knock_in_bound
        if self.model == BSMC:
            st_path = self.model().stock_price(spot, self.dt, self.n_path, self.n_days)
        elif (self.model == HestonMC) & (Andersen==False):
            st_path = self.model().stock_price(spot, self.dt, self.n_path, self.n_days)
        elif (self.model == HestonMC) & (Andersen==True):
            st_path = self.model().stock_price_Andersen(spot, self.dt, self.n_path, self.n_days)
        else:
            raise ValueError('Only support BSMC & HestonMC currently...')

        payout = np.zeros(self.n_path)
        payout_t = np.zeros(self.n_path)
        last_ko_t = np.zeros(self.n_path)
        last_ki_t = np.zeros(self.n_path)
        for i in range(self.n_path):
            ki = 0      # knock in
            ko = 0      # knock out
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
        price = np.mean(self.notional * (payout + 1) * np.exp(-payout_t / 365 * self.model().intr))
        return price

if __name__ == '__main__':
    texp = 3
    coupon_rate = 0.3
    bound = [0.75, 1]
    notional = 10000
    snowball = Snowball(HestonMC, texp, coupon_rate, bound, n_path=10000)
    price = snowball.price(5955.52, Andersen=False)
    print(price)