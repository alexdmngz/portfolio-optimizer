"""File extraction tests: locales, quantities, percentages, and ambiguous inputs."""

from io import BytesIO, StringIO
import unittest

import numpy as np
import pandas as pd

from holdings import clean_holdings, current_weights, read_holdings


class HoldingsTests(unittest.TestCase):
    def test_empty_csv_has_a_readable_error(self):
        for content in ("", "\n\n"):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "Cannot read this CSV"):
                read_holdings(StringIO(content), "positions.csv")

    def test_na_ticker_is_preserved_in_csv_and_excel(self):
        csv_source = StringIO("Ticker,Quantity\nNA,2\n")
        excel_source = BytesIO()
        pd.DataFrame({"Ticker": ["NA"], "Quantity": [2]}).to_excel(excel_source, index=False)
        excel_source.seek(0)
        for source, name in ((csv_source, "positions.csv"), (excel_source, "positions.xlsx")):
            with self.subTest(name=name):
                result = read_holdings(source, name)
                self.assertEqual(result["ticker"].tolist(), ["NA"])
                self.assertEqual(result["quantity"].tolist(), [2])

    def test_blank_rows_are_skipped_but_incomplete_positions_are_rejected(self):
        result = read_holdings(StringIO("Ticker,Quantity\n , \nAAPL,2\n"), "positions.csv")
        self.assertEqual(result["ticker"].tolist(), ["AAPL"])
        with self.assertRaisesRegex(ValueError, "Every holding"):
            read_holdings(StringIO("Ticker,Quantity\nAAPL,\n"), "positions.csv")

    def test_csv_detects_headers_and_combines_duplicates_in_original_order(self):
        source = StringIO("Symbol,Shares\nmsft,2\naapl,1\nMSFT,3\n")
        result = read_holdings(source, "positions.csv")
        self.assertEqual(result["ticker"].tolist(), ["MSFT", "AAPL"])
        np.testing.assert_allclose(result["quantity"], [5, 1])

    def test_spanish_csv_decimal_comma(self):
        source = StringIO("Símbolo;Cantidad\nAAPL;10,5\nMSFT;2,25\n")
        result = read_holdings(source, "positions.csv", decimal=",")
        np.testing.assert_allclose(result["quantity"], [10.5, 2.25])

    def test_excel_numeric_cells_ignore_display_locale(self):
        source = BytesIO()
        pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Cantidad": [1.5, 2]}).to_excel(source, index=False)
        source.seek(0)
        result = read_holdings(source, "positions.xlsx", decimal=",")
        np.testing.assert_allclose(result["quantity"], [1.5, 2])

    def test_weight_formats(self):
        for values in ([0.6, 0.4], [60, 40], ["60%", "40%"]):
            with self.subTest(values=values):
                result = clean_holdings(pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Peso": values}))
                np.testing.assert_allclose(result["weight"], [0.6, 0.4])

    def test_percent_sign_is_not_scaled_twice(self):
        with self.assertRaises(ValueError):
            clean_holdings(pd.DataFrame({"ticker": ["AAPL"], "weight": ["10000%"]}))

    def test_quantities_take_priority_over_broker_weights(self):
        result = clean_holdings(pd.DataFrame({"ticker": ["AAPL"], "quantity": [3], "weight": [100]}))
        self.assertEqual(result.columns.tolist(), ["ticker", "quantity"])

    def test_market_values_use_price_labels_not_position(self):
        holdings = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "quantity": [2, 1]})
        closes = pd.Series({"MSFT": 100, "AAPL": 25})
        np.testing.assert_allclose(current_weights(holdings, closes), [1 / 3, 2 / 3])

    def test_missing_valuation_price_is_rejected(self):
        holdings = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "quantity": [2, 1]})
        with self.assertRaises(ValueError):
            current_weights(holdings, pd.Series({"AAPL": 25}))

    def test_bad_quantities_rejected(self):
        for value in (None, np.nan, np.inf, -1, 0, "1,000", "1.000,5", "oops", "10%"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                clean_holdings(pd.DataFrame({"ticker": ["AAPL"], "quantity": [value]}))

    def test_bad_weights_rejected(self):
        for values in ([0.3, 0.2], [50, -50], ["50%", "50"], [0, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                clean_holdings(pd.DataFrame({"ticker": ["AAPL", "MSFT"], "weight": values}))

    def test_ambiguous_columns_rejected(self):
        with self.assertRaises(ValueError):
            clean_holdings(pd.DataFrame({"Ticker": ["AAPL"], "Symbol": ["MSFT"], "Quantity": [1]}))

    def test_repeated_csv_header_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Several columns"):
            read_holdings(StringIO("Ticker,Ticker,Quantity\nAAPL,MSFT,1\n"), "positions.csv")

    def test_unusable_tickers_rejected(self):
        for ticker in ("Apple Inc.", "US0378331005", "", "=SUM(A1)"):
            with self.subTest(ticker=ticker), self.assertRaises(ValueError):
                clean_holdings(pd.DataFrame({"ticker": [ticker], "quantity": [1]}))

    def test_missing_columns_rejected(self):
        with self.assertRaises(ValueError):
            clean_holdings(pd.DataFrame({"Name": ["Apple"], "quantity": [1]}))

    def test_unsupported_file_type_rejected(self):
        with self.assertRaises(ValueError):
            read_holdings(BytesIO(), "statement.pdf")

    def test_corrupted_excel_has_a_readable_error(self):
        with self.assertRaisesRegex(ValueError, "Cannot read this XLSX"):
            read_holdings(BytesIO(b"not a spreadsheet"), "positions.xlsx")


if __name__ == "__main__":
    unittest.main()
