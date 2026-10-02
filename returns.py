"""Calculate simple returns and estimate a one-period expected return."""

import numpy as np


def calculate_returns(prices):
    """Return decimal returns from one asset's chronological positive prices.

    Accept a list or a one-dimensional NumPy array. Prices must be finite,
    use consistent adjustments, and be observed at equally spaced intervals.
    Missing values are rejected, never silently removed or filled.
    """
    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 1:
        raise ValueError("Prices must be a one-dimensional sequence.")
    if prices.size < 2:
        raise ValueError("At least two prices are required.")
    if not np.all(np.isfinite(prices)):
        raise ValueError("Prices must not contain missing or infinite values.")
    if np.any(prices <= 0):
        raise ValueError("Prices must be greater than zero.")

    previous_prices = prices[:-1]
    current_prices = prices[1:]
    returns = current_prices / previous_prices - 1

    return returns


def estimate_mean_return(returns):
    """Estimate the next-period expected return using the arithmetic mean.

    Input and output are decimals at the same frequency. This estimate is
    neither a compounded return nor a guarantee of future performance.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1:
        raise ValueError("Returns must be a one-dimensional sequence.")
    if returns.size == 0:
        raise ValueError("At least one return is required.")
    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns must not contain missing or infinite values.")

    return float(np.mean(returns))
