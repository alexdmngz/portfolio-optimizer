"""Yahoo price history, daily returns and offline sample data."""

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers, period="2y"):
    """Return adjusted history, last closes, currency, and the common valuation date."""
    if period not in ("6mo", "1y", "2y", "5y"):
        raise ValueError("Choose a history period of 6mo, 1y, 2y, or 5y.")
    adjusted = {}
    closes = {}
    currencies = {}
    for symbol in tickers:
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval="1d", auto_adjust=False,
                                     actions=False, timeout=20, raise_errors=True)
            metadata = ticker.get_history_metadata()
        except Exception as error:
            raise ValueError(f"Cannot download {symbol}. Check the symbol, network, and Yahoo availability. {error}") from error
        if history.empty or not {"Close", "Adj Close"}.issubset(history.columns):
            raise ValueError(f"Yahoo returned no complete adjusted price history for {symbol}.")
        if metadata.get("instrumentType") not in ("EQUITY", "ETF"):
            raise ValueError(f"{symbol}: only stocks and ETFs are supported (252 trading days per year).")
        currency = metadata.get("currency")
        if not currency:
            raise ValueError(f"Unknown quote currency for {symbol}; cannot compare monetary values safely.")

        # Exclude today's possibly unfinished session, using the exchange timezone.
        today = pd.Timestamp.now(tz=history.index.tz).date()
        history = history.loc[history.index.date < today].copy()
        history.index = history.index.tz_localize(None).normalize()
        if history.empty or history.index.has_duplicates:
            raise ValueError(f"No unique completed daily sessions for {symbol}.")
        history = history.sort_index()
        scale = 0.01 if currency == "GBp" else 1.0
        currencies[symbol] = "GBP" if currency == "GBp" else currency
        adjusted[symbol] = history["Adj Close"] * scale
        closes[symbol] = history["Close"] * scale

    if not adjusted:
        raise ValueError("At least one ticker is required.")
    if len(set(currencies.values())) != 1:
        detail = ", ".join(f"{symbol}: {currency}" for symbol, currency in currencies.items())
        raise ValueError(f"Mixed currencies are not supported yet. Use assets quoted in one currency. {detail}")
    prices = pd.DataFrame(adjusted).sort_index()
    raw_closes = pd.DataFrame(closes).reindex(prices.index)
    last_close = raw_closes.iloc[-1]
    if last_close.isna().any() or not np.isfinite(last_close).all() or (last_close <= 0).any():
        raise ValueError("Some assets have no valid close on the latest session. Check stale symbols or different exchange calendars.")
    return prices, last_close, next(iter(currencies.values())), prices.index[-1]


def daily_returns(prices, min_observations=60):
    """Calculate adjacent returns, then discard incomplete observations."""
    if prices.empty or prices.columns.has_duplicates or prices.index.has_duplicates:
        raise ValueError("Prices must contain unique dates and tickers.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be sorted from oldest to newest.")
    values = prices.to_numpy(dtype=float)
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("Observed prices must be positive and finite.")
    # fill_method=None prevents Pandas from inventing flat returns at missing prices.
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns overflowed. Check the source prices.")
    if len(returns) < min_observations:
        raise ValueError(f"Need at least {min_observations} complete daily returns; found {len(returns)}. Try a longer period.")
    omitted = len(prices) - 1 - len(returns)
    return returns, omitted


def demo_prices():
    """Generate a fixed synthetic price history for three assets."""
    rng = np.random.default_rng(7)
    market = rng.normal(0.0003, 0.008, size=(504, 1))
    noise = rng.normal(0, [0.006, 0.012, 0.004], size=(504, 3))
    returns = market + noise + [0.0001, 0.0002, 0.00005]
    prices = 100 * np.cumprod(1 + returns, axis=0)
    dates = pd.bdate_range("2023-01-02", periods=504)
    return pd.DataFrame(prices, index=dates, columns=["DEMO_A", "DEMO_B", "DEMO_C"])
