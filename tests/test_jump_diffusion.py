import unittest
import numpy as np
from jump_diffusion import JumpDiffusion  # assuming the class is in 'jump_diffusion.py'

class TestJumpDiffusionModel(unittest.TestCase):
    
    def setUp(self):
        """
        This method is called before each test.
        Initialize the Jump Diffusion model with some sample parameters.
        """
        self.model = JumpDiffusion(mu=0.05, sigma=0.2, lambd=0.1, jump_mean=0.02, jump_vol=0.05)

    def test_price(self):
        """
        Test the option pricing function under Jump Diffusion.
        """
        strike = 100
        spot = 100
        texp = 1.0  # 1 year to expiry
        cp = 1  # Call option
        
        price = self.model.price(strike, spot, texp, cp)
        
        # Assert the price is a positive number
        self.assertGreater(price, 0, "Option price should be positive.")
    
    def test_vega(self):
        """
        Test the Vega (sensitivity to volatility) of the option under Jump Diffusion.
        """
        strike = 100
        spot = 100
        texp = 1.0  # 1 year to expiry
        cp = 1  # Call option
        
        vega = self.model.vega(strike, spot, texp, cp)
        
        # Assert that Vega is positive
        self.assertGreater(vega, 0, "Vega should be positive.")

    def test_delta(self):
        """
        Test the Delta (sensitivity to asset price) of the option under Jump Diffusion.
        """
        strike = 100
        spot = 100
        texp = 1.0  # 1 year to expiry
        cp = 1  # Call option
        
        delta = self.model.delta(strike, spot, texp, cp)
        
        # Assert that Delta is between 0 and 1 (for a call option)
        self.assertGreaterEqual(delta, 0, "Delta should be greater than or equal to 0.")
        self.assertLessEqual(delta, 1, "Delta should be less than or equal to 1.")
    
    def test_impvol(self):
        """
        Test the implied volatility calculation under Jump Diffusion.
        """
        price = 10  # Example option price
        strike = 100
        spot = 100
        texp = 1.0  # 1 year to expiry
        cp = 1  # Call option
        
        impvol = self.model.impvol(price, strike, spot, texp, cp)
        
        # Assert that implied volatility is a positive number
        self.assertGreater(impvol, 0, "Implied volatility should be positive.")

    def test_jump_diffusion_behavior(self):
        """
        Test the general behavior of the Jump Diffusion model for extreme parameters.
        This could include very high jump intensity or very large jump sizes.
        """
        strike = 100
        spot = 100
        texp = 1.0  # 1 year to expiry
        cp = 1  # Call option
        
        # Extremely high jump intensity and large jump size
        model = JumpDiffusion(mu=0.05, sigma=0.2, lambd=10, jump_mean=0.5, jump_vol=0.5)
        price = model.price(strike, spot, texp, cp)
        
        # Assert that the price is still reasonable
        self.assertGreater(price, 0, "Option price should be positive even for high jump intensity.")
    
    def test_barrier_option(self):
        """
        Test the pricing of barrier options under the Jump Diffusion model.
        """
        strike = 100
        spot = 100
        texp = 1.0  # 1 year to expiry
        cp = 1  # Call option
        barrier = 120  # Knock-in barrier
        
        price = self.model.price_barrier(strike, barrier, spot, texp, cp)
        
        # Assert that the barrier option price is a positive number
        self.assertGreater(price, 0, "Barrier option price should be positive.")
    
    def tearDown(self):
        """
        This method is called after each test.
        You can use this to clean up if needed.
        """
        pass


# Running the tests
if __name__ == '__main__':
    unittest.main()

