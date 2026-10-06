"""Hand-calculated mathematics and portfolio-search invariants."""

import unittest

import numpy as np

from portfolio import estimate_parameters, portfolio_metrics
from simulation import select_portfolios, simulate_portfolios


class PortfolioTests(unittest.TestCase):
    def test_sample_covariance_and_annualization_by_hand(self):
        # Centered observations are [-.01, -.02], [0, 0], [.01, .02].
        mean, covariance = estimate_parameters([[0.01, 0.02], [0.02, 0.04], [0.03, 0.06]], 12)
        np.testing.assert_allclose(mean, [0.24, 0.48])
        np.testing.assert_allclose(covariance, [[0.0012, 0.0024], [0.0024, 0.0048]])

    def test_return_volatility_and_sharpe_by_hand(self):
        result = portfolio_metrics([0.6, 0.4], [0.1, 0.2], [[0.04, 0.01], [0.01, 0.09]], 0.02)
        variance = 0.6**2 * 0.04 + 0.4**2 * 0.09 + 2 * 0.6 * 0.4 * 0.01
        np.testing.assert_allclose(result, [0.14, np.sqrt(variance), 0.12 / np.sqrt(variance)])

    def test_negative_covariance_reduces_risk(self):
        _, risk, _ = portfolio_metrics([0.5, 0.5], [0.1, 0.1], [[0.04, -0.02], [-0.02, 0.04]])
        self.assertAlmostEqual(risk, 0.1)

    def test_perfect_hedge_has_undefined_sharpe(self):
        _, risk, sharpe = portfolio_metrics([0.5, 0.5], [0.1, 0.1], [[0.04, -0.04], [-0.04, 0.04]])
        self.assertAlmostEqual(risk, 0)
        self.assertTrue(np.isnan(sharpe))

    def test_zero_volatility_is_not_infinite_sharpe(self):
        _, risk, sharpe = portfolio_metrics([1], [0.1], [[0]], 0.03)
        self.assertEqual(risk, 0)
        self.assertTrue(np.isnan(sharpe))

    def test_single_asset_covariance_keeps_matrix_shape(self):
        means, covariance = estimate_parameters([[0.01], [0.03]], 1)
        self.assertEqual(covariance.shape, (1, 1))
        self.assertAlmostEqual(covariance[0, 0], 0.0002)
        self.assertAlmostEqual(means[0], 0.02)

    def test_vectorized_metrics_match_individual_portfolios(self):
        means = [0.1, 0.2]
        covariance = [[0.04, 0.01], [0.01, 0.09]]
        weights = [[0.3, 0.7], [1, 0], [0.5, 0.5]]
        actual = np.asarray(portfolio_metrics(weights, means, covariance))
        expected = np.asarray([portfolio_metrics(w, means, covariance) for w in weights]).T
        np.testing.assert_allclose(actual, expected)

    def test_invalid_weights_rejected(self):
        for weights in ([0.5, 0.4], [-0.1, 1.1], [np.nan, 1], [1], []):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                portfolio_metrics(weights, [0.1, 0.2], np.eye(2))

    def test_invalid_covariance_rejected(self):
        for matrix in ([[1, 2], [2, 1]], [[1, 0], [0.1, 1]], [[np.inf, 0], [0, 1]], [[1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                portfolio_metrics([0.5, 0.5], [0.1, 0.2], matrix)

    def test_invalid_return_samples_rejected(self):
        for returns in ([], [0.1, 0.2], [[0.1]], [[0.1], [np.nan]], [[0.1], [-1.1]]):
            with self.subTest(returns=returns), self.assertRaises(ValueError):
                estimate_parameters(returns)

    def test_invalid_risk_free_rate_rejected(self):
        with self.assertRaises(ValueError):
            portfolio_metrics([1], [0.1], [[0.04]], np.nan)

    def test_random_weights_are_valid_reproducible_and_include_baselines(self):
        arguments = ([0.1, 0.2], [[0.04, 0.01], [0.01, 0.09]], [0.7, 0.3])
        weights, metrics = simulate_portfolios(*arguments, count=200, seed=42)
        again, _ = simulate_portfolios(*arguments, count=200, seed=42)
        np.testing.assert_array_equal(weights, again)
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        self.assertTrue((weights >= 0).all())
        np.testing.assert_allclose(weights[:4], [[0.7, 0.3], [0.5, 0.5], [1, 0], [0, 1]])
        self.assertEqual(len(metrics), 204)
        selected = select_portfolios(metrics)
        self.assertGreaterEqual(metrics.loc[selected["Best sampled Sharpe"], "sharpe"], metrics.loc[0, "sharpe"])
        self.assertLessEqual(metrics.loc[selected["Lowest sampled volatility"], "volatility"], metrics.loc[0, "volatility"])

    def test_single_asset_simulation(self):
        weights, metrics = simulate_portfolios([0.1], [[0.04]], [1], count=5)
        np.testing.assert_allclose(weights, 1)
        np.testing.assert_allclose(metrics["volatility"], 0.2)

    def test_all_zero_risk_candidates_have_no_best_sharpe(self):
        _, metrics = simulate_portfolios([0, 0], np.zeros((2, 2)), [0.5, 0.5], count=5)
        self.assertNotIn("Best sampled Sharpe", select_portfolios(metrics))

    def test_negative_sharpe_still_selects_largest(self):
        _, metrics = simulate_portfolios([-0.1, -0.2], np.eye(2), [0.5, 0.5], count=5)
        selected = select_portfolios(metrics)
        self.assertEqual(metrics.loc[selected["Best sampled Sharpe"], "sharpe"], metrics["sharpe"].max())

    def test_invalid_simulation_count_and_seed(self):
        for kwargs in ({"count": 0}, {"count": 2.5}, {"count": 200001}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_portfolios([0.1], [[0.04]], [1], **kwargs)


if __name__ == "__main__":
    unittest.main()
