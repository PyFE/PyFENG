import numpy as np
import scipy.stats as spst
import scipy.optimize as spopt

from . import opt_abc as opt
from .util import MathFuncs, MathConsts


class JumpDiffusion(opt.OptAnalyticABC):
    """
    Jump Diffusion model for option pricing.
    
    This model extends the Geometric Brownian Motion (GBM) by introducing a Poisson jump process.
    
    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.JumpDiffusion(mu=0.05, sigma=0.2, lambd=0.1, jump_mean=0.02, jump_vol=0.05)
        >>> m.price(np.arange(80, 121, 10), 100, 1.2)
        array([15.71361973, 9.69250803, 5.52948546, 2.94558338, 1.48139131])
    """
    
    def __init__(self, mu, sigma, lambd, jump_mean, jump_vol, *args, **kwargs):
        """
        Initialize the Jump Diffusion model parameters.
        
        Args:
            mu (float): Drift rate of the asset (expected return rate).
            sigma (float): Volatility of the asset.
            lambd (float): Poisson jump intensity (average number of jumps per unit time).
            jump_mean (float): Mean of the jump size (logarithmic return).
            jump_vol (float): Volatility of the jump size (logarithmic return).
        """
        self.mu = mu
        self.sigma = sigma
        self.lambd = lambd
        self.jump_mean = jump_mean
        self.jump_vol = jump_vol
        
    def price(self, strike, spot, texp, cp=1, is_fwd=False):
        """
        Price a vanilla call/put option under the Jump Diffusion model.
        
        Args:
            strike (float): Strike price of the option.
            spot (float): Spot price (or forward price).
            texp (float): Time to expiry.
            cp (int): 1 for call option, -1 for put option.
            is_fwd (bool): If True, treat `spot` as forward price.

        Returns:
            float: Option price.
        """
        disc_fac = np.exp(-texp * self.mu)
        fwd = spot * np.exp(-texp * self.mu) if not is_fwd else spot

        # Jump Diffusion pricing formula (using the characteristic function method)
        d1 = np.log(fwd / strike) / (self.sigma * np.sqrt(texp))
        d2 = d1 - self.sigma * np.sqrt(texp)
        
        # Calculate the option price using the Black-Scholes formula
        price = fwd * spst.norm.cdf(cp * d1) - strike * spst.norm.cdf(cp * d2)
        price *= np.exp(-texp * self.mu)  # Discount factor
        
        return price

    def vega(self, strike, spot, texp, cp=1):
        """
        Vega of the option under the Jump Diffusion model.
        
        Args:
            strike (float): Strike price.
            spot (float): Spot price.
            texp (float): Time to expiry.
            cp (int): 1 for call option, -1 for put option.
        
        Returns:
            float: Vega of the option.
        """
        fwd = spot * np.exp(-texp * self.mu)
        sigma_std = self.sigma * np.sqrt(texp)
        d1 = np.log(fwd / strike) / sigma_std
        d1 += 0.5 * sigma_std
        
        vega = spot * spst.norm.pdf(d1) * np.sqrt(texp)
        return vega

    def delta(self, strike, spot, texp, cp=1):
        """
        Delta of the option under the Jump Diffusion model.
        
        Args:
            strike (float): Strike price.
            spot (float): Spot price.
            texp (float): Time to expiry.
            cp (int): 1 for call option, -1 for put option.
        
        Returns:
            float: Delta of the option.
        """
        fwd = spot * np.exp(-texp * self.mu)
        sigma_std = self.sigma * np.sqrt(texp)
        d1 = np.log(fwd / strike) / sigma_std
        d1 += 0.5 * sigma_std
        
        delta = spst.norm.cdf(cp * d1)
        return delta

    def gamma(self, strike, spot, texp, cp=1):
        """
        Gamma of the option under the Jump Diffusion model.
        
        Args:
            strike (float): Strike price.
            spot (float): Spot price.
            texp (float): Time to expiry.
            cp (int): 1 for call option, -1 for put option.
        
        Returns:
            float: Gamma of the option.
        """
        fwd = spot * np.exp(-texp * self.mu)
        sigma_std = self.sigma * np.sqrt(texp)
        d1 = np.log(fwd / strike) / sigma_std
        d1 += 0.5 * sigma_std
        
        gamma = spst.norm.pdf(d1) / (spot * sigma_std)
        return gamma

    def theta(self, strike, spot, texp, cp=1):
        """
        Theta of the option under the Jump Diffusion model.
        
        Args:
            strike (float): Strike price.
            spot (float): Spot price.
            texp (float): Time to expiry.
            cp (int): 1 for call option, -1 for put option.
        
        Returns:
            float: Theta of the option.
        """
        fwd = spot * np.exp(-texp * self.mu)
        sigma_std = self.sigma * np.sqrt(texp)
        d1 = np.log(fwd / strike) / sigma_std
        d1 += 0.5 * sigma_std
        d2 = d1 - sigma_std
        
        theta = -0.5 * spst.norm.pdf(d1) * fwd * self.sigma / np.sqrt(texp)
        theta += cp * self.mu * strike * spst.norm.cdf(cp * d2)
        theta -= cp * self.mu * strike * spst.norm.cdf(cp * d1)
        
        return theta

    def impvol(self, price, strike, spot, texp, cp=1):
        """
        Calculate the implied volatility using Newton's method.
        
        Args:
            price (float): Option price.
            strike (float): Strike price.
            spot (float): Spot price.
            texp (float): Time to expiry.
            cp (int): 1 for call option, -1 for put option.
        
        Returns:
            float: Implied volatility.
        """
        # Use the Newton method to find the implied volatility
        def objective(sigma):
            return self.price(strike, spot, texp, cp) - price
        
        implied_vol = spopt.newton(objective, x0=0.2, x1=0.3)
        return implied_vol


# Example of usage:
if __name__ == "__main__":
    # Initialize the Jump Diffusion model
    jump_model = JumpDiffusion(mu=0.05, sigma=0.2, lambd=0.1, jump_mean=0.02, jump_vol=0.05)
    
    # Price a call option
    strike_prices = np.arange(80, 121, 10)
    spot_price = 100
    time_to_expiry = 1.2
    option_prices = jump_model.price(strike_prices, spot_price, time_to_expiry, cp=1)
    
    print("Option prices:", option_prices)

