"""Exercise full workflows, exports, and the local UI with deterministic data."""

from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from analysis import analyze_portfolio
from holdings import read_holdings
from market_data import demo_prices
from reporting import export_results
from test_market_data import fake_ticker


class WorkflowTests(unittest.TestCase):
    def test_offline_demo_export_is_labelled_and_reproducible(self):
        result = analyze_portfolio(count=100, seed=12)
        with tempfile.TemporaryDirectory() as folder:
            export_results(result, folder)
            metadata = json.loads((Path(folder) / "run.json").read_text())
            self.assertEqual(metadata["source"], "synthetic demo")
            self.assertEqual(metadata["seed"], 12)
            candidates = pd.read_csv(Path(folder) / "candidates.csv")
            self.assertEqual(len(candidates), 105)
            np.testing.assert_allclose(candidates.filter(like="weight_").sum(axis=1), 1)
            self.assertGreater((Path(folder) / "portfolios.png").stat().st_size, 1000)
            self.assertTrue((Path(folder) / "valuation_prices.csv").exists())

    def test_csv_to_mocked_yahoo_to_portfolio_comparison(self):
        holdings = read_holdings(StringIO("Symbol,Shares\nAAPL,2\nMSFT,1\n"), "broker.csv")
        prices = demo_prices().iloc[:, :2]
        first = fake_ticker(prices=pd.DataFrame({"Close": prices.iloc[:, 0] * 2, "Adj Close": prices.iloc[:, 0]}))
        second = fake_ticker(prices=pd.DataFrame({"Close": prices.iloc[:, 1], "Adj Close": prices.iloc[:, 1]}))
        with patch("market_data.yf.Ticker", side_effect=[first, second]):
            result = analyze_portfolio(holdings, count=50)
        self.assertEqual(result["settings"]["source"], "Yahoo Finance via yfinance")
        self.assertEqual(result["settings"]["currency"], "USD")
        self.assertEqual(result["allocations"].index.tolist(), ["AAPL", "MSFT"])
        expected = [4 * prices.iloc[-1, 0], prices.iloc[-1, 1]]
        np.testing.assert_allclose(result["allocations"]["Current"], np.asarray(expected) / sum(expected))

    def test_pipeline_api_failure_is_not_replaced_by_demo(self):
        holdings = read_holdings(StringIO("ticker,weight\nAAPL,1\n"), "broker.csv")
        with patch("analysis.download_prices", side_effect=ValueError("Rate limited")):
            with self.assertRaisesRegex(ValueError, "Rate limited"):
                analyze_portfolio(holdings)

    def test_cli_demo_and_invalid_input_exit_codes(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "main.py", "--demo", "--simulations", "10", "--output", folder],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("OFFLINE DEMO", result.stdout)
        result = subprocess.run([sys.executable, "main.py", "--holdings", "missing.csv"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_local_interface_runs_demo_and_shows_results(self):
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(app.dataframe), 2)
        self.assertEqual(app.session_state["result"]["settings"]["source"], "synthetic demo")

    def test_import_mode_requires_a_file(self):
        app = AppTest.from_file("app.py", default_timeout=30).run()
        app.radio[0].set_value("Import broker file + Yahoo prices").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button[0].disabled)


if __name__ == "__main__":
    unittest.main()
