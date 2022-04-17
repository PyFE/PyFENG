from .capriotti2018 import ApproxMethod
import scipy.stats as ss
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class GenerateRN(ApproxMethod):
    """
    Simulate the RN of X with f(x) in terms of
    the RN Z with g(x)
    f(x)/g(x) <= C where C>0, for all x.
    Here we use normal distribution as the g(x).
    """
    def __init__(self, a, b, sigma, y0, t):
        super().__init__(a, b, sigma)
        self.y0 = y0
        self.x0 = np.log(y0)
        self.t = t
        self.yvals = ''

    def _get_c(self, num: int = 50):
        yvals = np.r_[np.linspace(0.001, self.y0-0.005, num), np.linspace(self.y0+0.005, 1, num)]
        self.yvals = yvals
        xvals = np.log(yvals)
        wvals = self.w_for_density(x=xvals, x0=self.x0, t=self.t, n=3)
        c = np.exp(-(np.min(wvals) - 1))
        print('C=', c, '\n')
        return c

    def rv_for_garch(self, n: int = 1):
        assert n > 0
        c = self._get_c()
        if n < 2:
            while True:
                (z, u) = (np.random.normal(self.x0, self.sigma*np.sqrt(self.t)), np.random.uniform(0, 1))
                if abs(z-self.x0) < 1e-4:
                    continue
                elif u * c <= self.transition_density_x(x=z, x0=self.x0, t=self.t, n=3) / ss.norm.pdf(z):
                    break
            y_target = np.exp(z)
            return y_target
        else:
            y_target = np.zeros(n)
            i = 0
            while i < n:
                (z, u) = (np.random.normal(self.x0, self.sigma * np.sqrt(self.t)), np.random.uniform(0, 1))
                if abs(z-self.x0) < 1e-4:
                    continue
                elif u * c <= self.transition_density_x(x=z, x0=self.x0, t=self.t, n=3) / ss.norm.pdf(z):
                    y_target[i] = np.exp(z)
                    i += 1
            return y_target


def plot_freq(rv, pdf_y, y):
    fig, ax = plt.subplots(2, 1)
    ax1 = ax[0]
    ax2 = ax[1]
    sns.distplot(rv, rug=True, norm_hist=True, ax=ax1, axlabel='rv Y', label='freq(Y)')
    ax1.legend()
    ax1.grid()

    ax2.plot(y, pdf_y)
    ax2.set_xlabel('rv Y')
    ax2.set_ylabel('pdf(Y)')
    ax2.grid()
    plt.show()


def run():
    rvmodel = GenerateRN(a=0.1, b=0.04, sigma=0.6, y0=0.06, t=2.5)
    rv = rvmodel.rv_for_garch(10000)
    pdf_y = rvmodel.transition_density_y(rvmodel.yvals, y0=0.06, t=0.5)
    print(rv)

    plot_freq(rv, pdf_y, rvmodel.yvals)


if __name__ == '__main__':
    run()
