import numpy as np

class Snowball:
    texp = 2
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