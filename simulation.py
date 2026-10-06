"""Sample portfolio weights, then compare them using the same financial formulas."""

import numpy as np
import pandas as pd

from portfolio import portfolio_metrics


def simulate_portfolios(mean_returns, covariance, current, count=10000, seed=42, risk_free_rate=0.0):
    """Return candidates and metrics, including current, equal-weight, and single-asset baselines."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 200000:
        raise ValueError("Choose between 1 and 200,000 random portfolios.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("The random seed must be a non-negative integer.")
    assets = len(mean_returns)
    portfolio_metrics(current, mean_returns, covariance, risk_free_rate)
    if np.asarray(current).ndim != 1:
        raise ValueError("The current allocation must be a single weight vector.")

    rng = np.random.default_rng(seed)
    random_weights = rng.dirichlet(np.ones(assets), size=count)
    equal_weights = np.full(assets, 1 / assets)
    weights = np.vstack([current, equal_weights, np.eye(assets), random_weights])
    expected_return, volatility, sharpe = portfolio_metrics(weights, mean_returns, covariance, risk_free_rate)
    metrics = pd.DataFrame({"expected_return": expected_return, "volatility": volatility, "sharpe": sharpe})
    return weights, metrics


def select_portfolios(metrics):
    """Identify best candidates. A finite sample is not the exact efficient frontier."""
    selected = {"Current": 0, "Equal weight": 1, "Lowest sampled volatility": int(metrics["volatility"].idxmin())}
    valid_sharpe = metrics["sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    if not valid_sharpe.empty:
        selected["Best sampled Sharpe"] = int(valid_sharpe.idxmax())
    return selected
