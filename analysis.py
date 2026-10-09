"""Shared portfolio analysis for the app and CLI."""

import pandas as pd

from holdings import current_weights
from market_data import daily_returns, demo_prices, download_prices
from portfolio import estimate_parameters
from simulation import select_portfolios, simulate_portfolios


def analyze_portfolio(holdings=None, period="2y", count=10000, seed=42, risk_free_rate=0.0):
    """Analyze holdings, or synthetic demo data when holdings is None."""
    is_demo = holdings is None
    if is_demo:
        prices = demo_prices()
        holdings = pd.DataFrame({"ticker": prices.columns, "quantity": [10.0, 5.0, 15.0]})
        last_close = prices.iloc[-1]
        currency = "SYNTHETIC"
        valuation_date = prices.index[-1]
    else:
        prices, last_close, currency, valuation_date = download_prices(holdings["ticker"].tolist(), period)
    tickers = holdings["ticker"].tolist()
    prices = prices.reindex(columns=tickers)
    returns, omitted = daily_returns(prices)
    current = current_weights(holdings, last_close)
    means, covariance = estimate_parameters(returns)
    weights, metrics = simulate_portfolios(means, covariance, current, count, seed, risk_free_rate)
    selected = select_portfolios(metrics)

    comparison = metrics.loc[list(selected.values())].copy()
    comparison.index = list(selected.keys())
    allocations = pd.DataFrame(index=tickers)
    for label, index in selected.items():
        allocations[label] = weights[index]
    allocations.index.name = "ticker"
    messages = []
    if is_demo:
        messages.append("OFFLINE DEMO: all prices and holdings are synthetic; no market API was called.")
    if omitted:
        messages.append(f"Excluded {omitted} return rows with missing prices. No values were filled.")
    if "Best sampled Sharpe" not in selected:
        messages.append("Sharpe is undefined for every candidate because volatility is zero.")
    if not is_demo and (pd.Timestamp.now().normalize() - valuation_date).days > 7:
        messages.append("The latest available closing price is more than seven days old.")
    messages.append("Estimates use the same historical sample for fitting and selection; this is not an out-of-sample backtest.")

    return {
        "holdings": holdings, "prices": prices, "returns": returns,
        "last_close": last_close.reindex(tickers), "weights": weights, "metrics": metrics,
        "comparison": comparison, "allocations": allocations, "selected": selected,
        "messages": messages,
        "settings": {"source": "synthetic demo" if is_demo else "Yahoo Finance via yfinance",
                     "currency": currency, "valuation_date": str(valuation_date.date()),
                     "first_return_date": str(returns.index[0].date()),
                     "last_return_date": str(returns.index[-1].date()),
                     "observations": len(returns), "omitted_return_rows": omitted,
                     "requested_period": None if is_demo else period,
                     "random_portfolios": count, "total_candidates": len(metrics),
                     "seed": seed, "annual_risk_free_rate": risk_free_rate,
                     "periods_per_year": 252,
                     "initial_weights": "quantity times last close" if "quantity" in holdings else "imported weights"},
    }
