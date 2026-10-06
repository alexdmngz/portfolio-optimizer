"""Pure mean-variance mathematics. All inputs and outputs use decimal returns."""

import numpy as np


def estimate_parameters(returns, periods_per_year=252):
    """Estimate annual arithmetic means and sample covariance from rows of daily returns."""
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] == 0:
        raise ValueError("Use at least two return observations, with one column per asset.")
    if not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError("Simple returns must be finite and at least -1.")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("Periods per year must be positive and finite.")
    mean_returns = returns.mean(axis=0) * periods_per_year
    covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1)) * periods_per_year
    return mean_returns, covariance


def portfolio_metrics(weights, mean_returns, covariance, risk_free_rate=0.0):
    """Evaluate one portfolio or a matrix of portfolios; zero-risk Sharpe is undefined."""
    weights = np.asarray(weights, dtype=float)
    mean_returns = np.asarray(mean_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mean_returns.ndim != 1 or mean_returns.size == 0:
        raise ValueError("Expected returns must be a non-empty vector.")
    assets = mean_returns.size
    if weights.ndim not in (1, 2) or weights.shape[-1] != assets or weights.size == 0:
        raise ValueError("Each weight vector must match the number of assets.")
    if covariance.shape != (assets, assets):
        raise ValueError("Covariance must be a square matrix matching the assets.")
    if not all(np.isfinite(array).all() for array in (weights, mean_returns, covariance)):
        raise ValueError("Weights and estimated parameters must be finite.")
    if (weights < 0).any() or not np.allclose(weights.sum(axis=-1), 1, rtol=0, atol=1e-8):
        raise ValueError("Long-only portfolio weights must be non-negative and sum to 1.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("The annual risk-free rate must be finite.")
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-10):
        raise ValueError("Covariance must be symmetric.")
    if np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite.")

    expected_return = weights @ mean_returns
    # Each row calculates w.T @ covariance @ w, without a loop over portfolios.
    variances = np.sum((weights @ covariance) * weights, axis=-1)
    volatility = np.sqrt(np.maximum(variances, 0))
    excess_return = expected_return - risk_free_rate
    sharpe = np.full_like(expected_return, np.nan, dtype=float)
    np.divide(excess_return, volatility, out=sharpe, where=volatility > 1e-12)
    return expected_return, volatility, sharpe
