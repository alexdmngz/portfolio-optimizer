"""Check the external-data boundary without depending on a live service."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from market_data import daily_returns, download_prices


def fake_ticker(currency="USD", instrument="EQUITY", prices=None):
    ticker = Mock()
    if prices is None:
        prices = pd.DataFrame({"Close": [100.0, 105.0, 110.0], "Adj Close": [90.0, 95.0, 100.0]},
                              index=pd.date_range("2020-01-01", periods=3, tz="America/New_York"))
    ticker.history.return_value = prices
    ticker.get_history_metadata.return_value = {"currency": currency, "instrumentType": instrument}
    return ticker


class MarketDataTests(unittest.TestCase):
    def test_adjusted_returns_and_raw_valuation_are_separate(self):
        ticker = fake_ticker()
        with patch("market_data.yf.Ticker", return_value=ticker):
            prices, last_close, currency, date = download_prices(["AAPL"])
        np.testing.assert_allclose(prices["AAPL"], [90, 95, 100])
        self.assertEqual(last_close["AAPL"], 110)
        self.assertEqual(currency, "USD")
        self.assertEqual(date, pd.Timestamp("2020-01-03"))
        self.assertIsNone(prices.index.tz)

    def test_mixed_currencies_rejected(self):
        with patch("market_data.yf.Ticker", side_effect=[fake_ticker("USD"), fake_ticker("EUR")]):
            with self.assertRaisesRegex(ValueError, "Mixed currencies"):
                download_prices(["AAPL", "SAP.DE"])

    def test_pence_converted_to_pounds(self):
        with patch("market_data.yf.Ticker", return_value=fake_ticker("GBp")):
            prices, last_close, currency, _ = download_prices(["LLOY.L"])
        self.assertEqual(currency, "GBP")
        self.assertAlmostEqual(last_close.iloc[0], 1.1)
        self.assertAlmostEqual(prices.iloc[0, 0], 0.9)

    def test_missing_currency_and_unsupported_asset_rejected(self):
        for ticker in (fake_ticker(None), fake_ticker(instrument="CRYPTOCURRENCY")):
            with patch("market_data.yf.Ticker", return_value=ticker), self.assertRaises(ValueError):
                download_prices(["AAPL"])

    def test_api_failure_never_becomes_fake_data(self):
        ticker = fake_ticker()
        ticker.history.side_effect = RuntimeError("Rate limited")
        with patch("market_data.yf.Ticker", return_value=ticker):
            with self.assertRaisesRegex(ValueError, "Cannot download AAPL"):
                download_prices(["AAPL"])

    def test_missing_adjusted_column_and_empty_response_rejected(self):
        for history in (pd.DataFrame(), pd.DataFrame({"Close": [100]})):
            with patch("market_data.yf.Ticker", return_value=fake_ticker(prices=history)), self.assertRaises(ValueError):
                download_prices(["AAPL"])

    def test_incomplete_today_is_excluded(self):
        today = pd.Timestamp.now(tz="America/New_York").normalize()
        prices = pd.DataFrame({"Close": [100, 110], "Adj Close": [100, 110]},
                              index=pd.DatetimeIndex([today - pd.Timedelta(days=1), today]))
        with patch("market_data.yf.Ticker", return_value=fake_ticker(prices=prices)):
            history, last_close, _, _ = download_prices(["AAPL"])
        self.assertEqual(len(history), 1)
        self.assertEqual(last_close.iloc[0], 100)

    def test_stale_asset_on_last_date_is_rejected(self):
        first, second = fake_ticker(), fake_ticker()
        second.history.return_value = second.history.return_value.iloc[:-1]
        with patch("market_data.yf.Ticker", side_effect=[first, second]), self.assertRaisesRegex(ValueError, "latest session"):
            download_prices(["AAPL", "MSFT"])

    def test_missing_prices_do_not_create_multiday_daily_returns(self):
        prices = pd.DataFrame({"A": [100, 110, np.nan, 121, 133.1], "B": [100, 100, 100, 100, 100]})
        returns, omitted = daily_returns(prices, min_observations=2)
        self.assertEqual(returns.index.tolist(), [1, 4])
        np.testing.assert_allclose(returns["A"], [0.1, 0.1])
        self.assertEqual(omitted, 2)

    def test_insufficient_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "60 complete daily returns"):
            daily_returns(pd.DataFrame({"A": [100, 110, 120]}))

    def test_invalid_observed_prices_rejected(self):
        for value in (0, -1, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                daily_returns(pd.DataFrame({"A": [100, value, 120]}), min_observations=1)

    def test_unsorted_and_duplicate_dates_rejected(self):
        for index in ([1, 0, 2], [0, 0, 1]):
            with self.subTest(index=index), self.assertRaises(ValueError):
                daily_returns(pd.DataFrame({"A": [100, 110, 120]}, index=index), min_observations=1)


if __name__ == "__main__":
    unittest.main()
