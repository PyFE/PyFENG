class Heston(BaseModel):
    def __init__(self):
        super().__init__()
        self.vov = vov
        self.rho = rho
        self.kappa = kappa
        self.theta = theta


    def stock_price(self, spot,dt):
        n_days = self.texp / dt

        # get  RV with relation rho, the number of RV is n_path * n_days
        f_xy = np.random.multivariate_normal(mean=[0,0],cov = [[1,self.rho],[self.rho,1]],size=n_path*n_days)
        x = [i[0] for i in f_xy]
        y = [i[1] for i in f_xy]

        #get sigma_sqrt path
        v = np.zeros(n_days+1)
        v[0] = self.sigma**2

        st_path = np.zeros((self.n_path, n_days))
        st_path[:,0] = spot

        for i in range(n_path):
            for k in range(n_days):
                v[k+1] = v[k] + self.kappa*(self.theta-v[k])*dt + self.vov*np.sqrt(v[k]*dt) * x[i*n_days+k]
                st_path[i, k+1] = st_path[i,k] * np.exp( sqrt(v[k]) * (self.rho * sqrt(dt) * x[i*n_days+k] + sqrt((1 - self.rho ** 2) * dt) * y[i*n_days+k]) \
                                   - 0.5 * v[k] * dt )

        return st_path

    def option_price(self, strike, spot, dt, cp):
        st_path = self.stock_price(spot, dt)
        if cp == 1:
            price = np.mean(np.fmax(st_path[:,-1] - strike, 0) * np.exp(-self.intr * self.texp))
        elif cp == 0:
            price = np.mean(np.fmax(strike - st_path[:,-1], 0) * np.exp(-self.intr * self.texp))
        return price

    def implied_vol(self, strike, spot, dt, cp):
        price = self.price(strike, spot, dt, cp)
        impvol = pf.Bsm.impvol(price, strike, spot, self.texp,cp)
        return impvol