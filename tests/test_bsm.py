import unittest
import pyfeng as pf

class TestBsmMethods(unittest.TestCase):

    def test_bsm(self):
        bsm = pf.BsmDisp(sigma=0.2)
        result = bsm.price(strike=105, spot=100, texp=1)
        expect_result = 5.905593471555491
        self.assertEqual(result, expect_result)

    def test_bsm2(self):
        bsm = pf.BsmDisp(sigma=0.2)
        result = bsm.price(strike=105, spot=100, texp=1)
        expect_result = 5.90559347
        self.assertAlmostEqual(result, expect_result)


if __name__ == '__main__':
    unittest.main()
