# Portfolio Optimizer

A readable Python application that imports a portfolio, downloads historical market
prices, and compares long-only allocations using Markowitz's mean-variance framework
and Monte Carlo sampling. Built as a fintech engineering learning project: small
functions, explicit mathematics, and tests with known answers.

## What it does

- Extracts tickers and quantities or weights from a broker's CSV/XLSX positions export.
- Downloads daily stock/ETF data from Yahoo Finance through `yfinance`.
- Computes annualized arithmetic return, sample covariance, volatility, and Sharpe.
- Compares the current portfolio, equal weights, lowest sampled volatility, and best
  sampled Sharpe; includes single-asset allocations among the candidates.
- Provides a local Streamlit interface and a terminal command.
- Exports allocations, metrics, a chart, and reproducible run inputs/settings.
- Includes a clearly labelled synthetic demo that works offline.

The program analyzes the assets in the imported file. It does not discover which
assets you own without that file, connect to your broker account, or place orders.

## Run locally

Use **Python 3.11 or 3.12**. Clone the repository and create an environment:

```bash
git clone https://github.com/alexdmngz/portfolio-optimizer.git
cd portfolio-optimizer
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install and launch:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Start
with **Offline demo**, or select **Import broker file + Yahoo prices**, upload your
export, check the extracted table, and click **Run optimization**. Downloads include
allocation/comparison CSVs and the chart. Changing inputs requires another run;
previous results retain their original run settings.

### Terminal

```bash
# Explicit offline demonstration; python main.py alone also runs this demo.
python main.py --demo

# Download market history for the sample holdings (internet required).
python main.py --holdings examples/holdings.csv --period 2y --simulations 10000

# Spanish decimal convention, using a semicolon-separated file.
python main.py --holdings examples/holdings_es.csv --decimal ","

# Your own export and explicit annual risk-free rate; 0.03 means 3%.
python main.py --holdings private/positions.xlsx --risk-free-rate 0.03 --seed 42

python main.py --help
python -m unittest -v
```

Put personal exports in `private/` (ignored by git). `output/` is also ignored.
The CLI writes `comparison.csv`, `allocations.csv`, `candidates.csv`,
`adjusted_prices.csv`, `daily_returns.csv`, `imported_holdings.csv`,
`valuation_prices.csv`, `run.json`, and `portfolios.png`. Re-running into the same
folder replaces these report files; use `--output output/run_2` to preserve a run.
CSV returns and weights are decimals: **0.10 = 10%**. An empty Sharpe cell means
undefined, not zero. The interface displays percentages directly.

## Import your portfolio automatically

Export **current positions**, not your transaction history. Headers must be on the
first row; XLSX reads the first sheet. Extra unrelated columns are ignored.

| Required information | Recognized headers (case/accent insensitive) |
| --- | --- |
| Yahoo ticker | Ticker, Symbol, Símbolo, Símbolo bursátil |
| Number of shares/units | Quantity, Qty, Shares, Units, Cantidad, Unidades, Títulos |
| Alternative to quantities | Weight, Weight %, Allocation, Allocation %, Peso, Peso %, Porcentaje |

Quantities take priority when both quantity and weight columns exist. Repeated
symbols are added together in the file's original order. Fractional shares and
zero positions are supported, but there must be a positive total. All imported
symbols, including zero positions, remain eligible for optimization.

Weights may be fractions totaling 1, numbers totaling 100, or consistent percent
strings such as `60%` and `40%`. Totals within 0.1 percentage points of 100% are
normalized for rounding. Quantities must be non-negative. Text numbers use your
chosen decimal separator and **no thousands separators**; numeric Excel cells
are read directly. Use semicolons or quoted fields for decimal-comma CSV files.

For example, this is read without entering any positions in the app:

```csv
Symbol,Shares
AAPL,10
MSFT,5
SPY,3
```

Those are illustrative holdings, not investment recommendations. Use Yahoo's
exchange-specific symbols (e.g. `SAP.DE`). Company names, ISIN-only exports,
PDFs, screenshots, broker login, and transaction-to-position reconstruction are
not supported. Rename an unrecognized header or convert the export once; there
is no need to retype each position. The interface previews what was extracted.

## Mathematical background

For asset i and session t, adjusted simple returns are

$$r_{i,t}=P^{adj}_{i,t}/P^{adj}_{i,t-1}-1.$$

With T common daily observations and D = 252 sessions per year:

$$\hat\mu_i=D\frac{1}{T}\sum_{t=1}^{T}r_{i,t},$$

$$\hat\Sigma_{ij}=\frac{D}{T-1}\sum_{t=1}^{T}(r_{i,t}-\bar r_i)(r_{j,t}-\bar r_j).$$

For weights satisfying w_i >= 0 and sum(w_i) = 1:

$$\hat\mu_p=w^T\hat\mu,\qquad \hat\sigma_p=\sqrt{w^T\hat\Sigma w},\qquad
\hat S_p=\frac{\hat\mu_p-r_f}{\hat\sigma_p}.$$

The return estimate is **annualized arithmetic return**, not CAGR or a forecast of
compounded wealth. Annualizing covariance by 252 assumes stable daily moments
and no material serial correlation. The risk-free input uses the same annual
arithmetic convention and quote currency. If starting from an effective annual
yield y, its matching arithmetic rate is `252 * ((1 + y)**(1/252) - 1)` under a
constant daily rate assumption. The default 0% is an example, not a live rate feed.

Covariance captures how assets move together; diversification changes portfolio
variance through the off-diagonal terms. We use sample covariance (`T - 1`),
complete common return rows, and no inverse matrix. Identical assets and singular
covariance matrices therefore remain usable. Near-zero volatility (<= 1e-12)
produces undefined Sharpe, excluded from Sharpe selection.

Monte Carlo here samples **allocations**, not future price paths:

```python
weights = rng.dirichlet(np.ones(number_of_assets), size=number_of_portfolios)
```

Dirichlet(1, ..., 1) samples uniformly on the long-only weight simplex. A fixed seed
reproduces the draws within the same software environment. Current, equal-weight,
and single-asset allocations are added explicitly. The highest Sharpe and lowest
volatility are best **among evaluated candidates**; the point cloud is not an exact
efficient frontier. Sampling becomes less effective as the number of assets grows.

## Data choices and limits

- Prices come from Yahoo via `yfinance`, an **unofficial** research/personal-use
  integration. No API key is needed. Service availability, licensing, and rate
  limits are controlled by the provider. A failed request stops with an error;
  it never silently switches to synthetic data.
- Adjusted closes are used for return estimation; last unadjusted closes are used
  for `quantity * price` valuation. The current weights describe the uploaded
  quantities at that closing date, not a live brokerage balance.
- Today's potentially unfinished exchange session is excluded. Histories are
  aligned by local session date, not intraday timestamps. At least 60 complete
  daily return observations are required; this threshold is a minimum input rule,
  not a statistical guarantee.
- Returns are calculated **before** incomplete rows are removed, without forward
  filling. Missing rows and actual sample dates are reported. If every asset is
  missing the same date, no external exchange calendar is available to detect it.
- Stocks and ETFs only; at most 50 assets, 200,000 random allocations. One common
  quote currency is required. GBp is converted to GBP; other mixed currencies
  are rejected until explicit FX conversion is implemented. Different exchange
  calendars or stale symbols can make the latest common valuation unavailable.
- No cash balance, short positions, leverage, fees, taxes, share-lot rounding,
  turnover limits, or per-asset caps. A result can allocate 100% to one asset.
- Estimation and selection use the same sample: **this is not a backtest**. Historical
  means are noisy, and favorable in-sample metrics do not establish future returns.
  A negative maximum Sharpe can occur; it is still only the largest of the sampled
  ratios, and can be a poor ranking criterion when excess returns are negative.

## System architecture

| Module | Responsibility | Why it is separate |
| --- | --- | --- |
| `holdings.py` | Read exports, recognize columns, calculate initial weights | File conventions do not belong in financial formulas. |
| `market_data.py` | Yahoo calls, session alignment, missing-data rules, synthetic demo | External services can fail independently of mathematics. |
| `returns.py` | Original single-asset learning functions | Preserves the first lesson and its tests. |
| `portfolio.py` | Means, covariance, return, volatility, Sharpe | Pure functions are easy to check by hand. |
| `simulation.py` | Random weights and candidate selection | Sampling can evolve without changing formulas. |
| `analysis.py` | Connect the steps and assemble labelled results | The UI and CLI share the same workflow. |
| `reporting.py` | Charts and file exports | Presentation never recalculates financial metrics. |
| `app.py` / `main.py` | Local interface / terminal entry point | Two ways to run the same program. |

No production classes, database, background workers, or application framework
beyond the small Streamlit UI. NumPy handles the numerical arrays; Pandas handles
labelled tables; Matplotlib draws the charts; OpenPyXL reads Excel.

## Tests and learning guide

`python -m unittest -v` tests hand-calculated returns/covariance/Sharpe,
diversification, zero risk, singular covariance, random weight constraints,
CSV/XLSX extraction, locales, missing prices, currency checks, service failure,
exports, the CLI, and Streamlit's demo and upload requirement. Network responses
are mocked in automated tests so CI does not depend on Yahoo uptime. GitHub
Actions runs the suite and demo on Python 3.11 and 3.12.

Read [the first returns lesson](docs/01_returns.md), then
[the complete block-by-block walkthrough](docs/02_walkthrough.md).
The repository is in English; mentoring explanations can be in Spanish.

## Future improvements

1. Add a licensed market-data provider and explicit provider selection.
2. Add broker-specific adapters or read-only positions APIs, plus PDF extraction
   with a confirmation table for ambiguous values.
3. Convert foreign prices and returns into a chosen base currency.
4. Compare Monte Carlo against a constrained numerical optimizer and an exact
   mean-variance frontier.
5. Add covariance shrinkage, weight/sector limits, cash, and transaction costs.
6. Evaluate allocations using walk-forward, out-of-sample backtests and
   sensitivity analysis across estimation windows.

## References

- [yfinance documentation and data-use notice](https://ranaroussi.github.io/yfinance/)
- [Pandas percentage change](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html)
- [NumPy sample covariance](https://numpy.org/doc/stable/reference/generated/numpy.cov.html)
- [NumPy Dirichlet sampling](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.dirichlet.html)
- [William F. Sharpe: The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm)
