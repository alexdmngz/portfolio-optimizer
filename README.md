# Portfolio Optimizer

**In development.** The current version is a working prototype.

Compare your current portfolio with thousands of alternative allocations. Import
a CSV or Excel file, download historical prices from Yahoo Finance, and explore
the trade-off between return and volatility in a local Streamlit app.

The calculation uses Markowitz's mean-variance model and Monte Carlo sampling.
It compares your holdings, equal weights, the lowest sampled volatility and the
highest sampled Sharpe ratio. All allocations are long-only and fully invested.

## Run it

Use Python 3.11 or 3.12.

```bash
git clone https://github.com/alexdmngz/portfolio-optimizer.git
cd portfolio-optimizer
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or
`.venv\Scripts\Activate.ps1` in Windows PowerShell, then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Choose **Offline demo** to try synthetic data, or upload your holdings and select
**Run optimization**. The app previews the imported positions before running.
Results keep their original settings until you run again.

The same calculation is available from the terminal:

```bash
python main.py --demo
python main.py --holdings examples/holdings.csv --period 2y --simulations 10000
python main.py --holdings examples/holdings_es.csv --decimal ","
python main.py --holdings private/positions.xlsx --risk-free-rate 0.03 --output output/my-run
python main.py --help
```

## Holdings file

Export current positions with headers on the first row. CSV and the first sheet
of an XLSX file are supported. A minimal example:

```csv
Symbol,Shares
AAPL,10
MSFT,5
SPY,3
```

| Column | Accepted headers |
| --- | --- |
| Yahoo ticker | Ticker, Symbol, Símbolo, Símbolo bursátil |
| Quantity | Quantity, Qty, Shares, Units, Cantidad, Unidades, Títulos |
| Weight instead of quantity | Weight, Weight %, Allocation, Allocation %, Peso, Peso %, Porcentaje |

Headers ignore case and accents. Quantities take priority over weights; repeated
tickers are combined. Weights can sum to 1 or 100, or use percent strings such as
`60%`. Small rounding differences are normalized. Use one decimal convention and
no thousands separators; decimal-comma CSVs need semicolons or quoted fields.

Use exchange-specific Yahoo symbols such as `SAP.DE`. Transaction histories,
ISIN-only exports, PDFs and broker connections are outside the supported input
format. Put personal exports in `private/`, which is ignored by Git.

## Results

The app offers allocation, comparison and chart downloads. The terminal also
saves every candidate, input prices, returns, imported holdings, valuation prices
and settings in `output/`. Choose a different `--output` folder to keep separate
runs; existing report files in the selected folder are replaced.

CSV weights and returns are decimals (`0.10` means 10%). A blank Sharpe ratio
means it is undefined because volatility is effectively zero.

Adjusted closes determine historical returns; unadjusted closes value the
uploaded quantities. The latest session must have a valid close for every asset.
Today's session is excluded. Missing prices are never filled, and at least 60
common daily returns are required.

Portfolios can contain up to 50 stocks or ETFs in one quote currency. Pence quotes
are converted to pounds; other currency conversions are not supported. The model
does not include cash, shorting, leverage, fees, taxes or trading constraints.
Yahoo access uses the unofficial `yfinance` library; a failed download returns
an error.

The best allocation is the best among the sampled candidates. The chart is not
an exact efficient frontier, and this is an in-sample comparison, not a backtest
or a forecast. See [the calculation notes](docs/methodology.md) for the formulas,
sampling method and assumptions.

## Development

`holdings.py` handles imports, `market_data.py` prices, `portfolio.py` the
mathematics, and `simulation.py` sampling. `analysis.py` combines them for both
the app and CLI; `reporting.py` writes tables and charts.

```bash
python -m unittest -v
```

Tests cover the calculations, file imports, missing prices, exports and Streamlit
interface. Market responses are mocked so the suite runs offline.
