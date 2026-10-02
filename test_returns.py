"""Run with: python -m unittest -v"""

import unittest

import numpy as np

from returns import calculate_returns, estimate_mean_return


class ReturnsTests(unittest.TestCase):
    def test_gain_followed_by_loss(self):
        actual = calculate_returns([100, 110, 99])
        np.testing.assert_allclose(actual, [0.10, -0.10])

    def test_constant_prices_have_zero_returns(self):
        actual = calculate_returns([100, 100, 100])
        np.testing.assert_allclose(actual, [0.0, 0.0])

    def test_two_prices_produce_one_return(self):
        actual = calculate_returns([100, 125])
        np.testing.assert_allclose(actual, [0.25])

    def test_empty_prices_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate_returns([])

    def test_one_price_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_returns([100])

    def test_zero_price_is_rejected_before_division(self):
        with self.assertRaises(ValueError):
            calculate_returns([0, 100])

    def test_negative_price_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_returns([100, -10])

    def test_missing_price_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_returns([100, np.nan, 110])

    def test_infinite_price_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_returns([100, np.inf])

    def test_multiple_price_series_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate_returns([[100, 110], [200, 220]])

    def test_mean_return_matches_hand_calculation(self):
        actual = estimate_mean_return([0.01, 0.02, 0.03])
        self.assertAlmostEqual(actual, 0.02)

    def test_arithmetic_mean_is_not_compounded_growth(self):
        returns = calculate_returns([100, 110, 99])
        cumulative_return = np.prod(1 + returns) - 1
        self.assertAlmostEqual(estimate_mean_return(returns), 0.0)
        self.assertAlmostEqual(cumulative_return, -0.01)

    def test_negative_mean_return_is_valid(self):
        actual = estimate_mean_return([-0.10, -0.20])
        self.assertAlmostEqual(actual, -0.15)

    def test_empty_returns_are_rejected(self):
        with self.assertRaises(ValueError):
            estimate_mean_return([])

    def test_missing_return_is_rejected(self):
        with self.assertRaises(ValueError):
            estimate_mean_return([0.01, np.nan])

    def test_infinite_return_is_rejected(self):
        with self.assertRaises(ValueError):
            estimate_mean_return([0.01, np.inf])

    def test_multiple_return_series_are_rejected(self):
        with self.assertRaises(ValueError):
            estimate_mean_return([[0.01, 0.02]])


if __name__ == "__main__":
    unittest.main()
