# The complete program, block by block

Read this guide alongside the named functions. Every block answers one question.
The implementation now includes the full workflow at the learner's request;
the original returns lesson remains available for studying the foundations.

## 1. What goes into the program? — `holdings.py`

**Input:** a broker positions export. **Output:** a small table containing `ticker`
and either `quantity` or `weight`.

`read_holdings` chooses the reader by extension: Pandas reads CSV; OpenPyXL,
through Pandas, reads Excel. A file can be a path from the terminal or an uploaded
object from Streamlit. Both lead to `clean_holdings`.

`normalize_header` removes differences that do not change a column's meaning:
capital letters, accents, underscores, and repeated spaces. `COLUMN_NAMES` is an
ordinary dictionary listing accepted names; adding a broker header is one small
change to this dictionary. We deliberately do not guess symbols from company names
or ISINs, because the exchange/currency can be ambiguous.

`parse_numbers` follows the selected decimal convention. For CSV values, `10,5`
means 10.5 only when comma is selected. Numeric Excel cells already carry numeric
values. Thousands separators are rejected instead of being interpreted silently.
Each input must be finite, non-negative, and contribute to a positive total.

`clean_holdings` then combines repeated tickers using `groupby(...).sum()`.
Quantities take priority over weights because imported quantities can be valued
using the same recent price date. File weights are accepted as the user's stated
allocation; their original valuation date cannot be inferred from two columns.

When there are quantities q_i, `current_weights` computes

$$V_i=q_i P_i^{close},\qquad w_i=V_i/\sum_j V_j.$$

For 2 shares at 25 and 1 share at 100, the values are 50 and 100. The weights are
1/3 and 2/3, not 2/3 and 1/3. Share counts alone cannot measure portfolio weights.
`reindex` aligns each price with its ticker; it does not trust table position.

**Check your understanding:** if the first stock doubles and the second does not
move, what happens to the weights without any trades?

## 2. Which prices do we need? — `market_data.py`

`download_prices` loops over symbols and calls:

```python
ticker.history(period=period, interval="1d", auto_adjust=False,
               actions=False, timeout=20, raise_errors=True)
```

`auto_adjust=False` makes both `Close` and `Adj Close` available. Adjusted closes
are the input to historical returns; the latest unadjusted close values current
shares. Dividends and splits are why these jobs require different columns.
`get_history_metadata` supplies the quote currency and instrument type.

The function rejects missing histories, unsupported assets, unknown currencies,
and mixed currencies. GBp quotes are explicitly divided by 100 to obtain GBP.
This check is necessary even when importing weights: combining an EUR return
with a USD return would omit the exchange-rate exposure of one investment.

Today's row is excluded using the exchange timezone because its close might
still change. Removing timezone information afterwards preserves each local
session date, allowing daily tables to align by date. This is daily research
alignment, not simultaneous intraday pricing.

`daily_returns` is the multi-asset version of the first lesson:

```python
returns = prices.pct_change(fill_method=None).dropna(how="any")
```

Order matters. For `100, missing, 110`, deleting the missing price first would
incorrectly treat a two-period gain as one daily gain. We calculate adjacent-row
returns first, then remove incomplete return rows. `fill_method=None` prevents
forward filling. All assets use the same retained observations when estimating
covariance. The run reports how many return rows were excluded.

An API exception becomes a readable `ValueError`. The demo is selected explicitly;
it never replaces a failed download. That keeps a provider outage visible.

**Check:** why would forward filling a missing price create a zero return, and
how might that affect an estimate of volatility?

## 3. How do we estimate the assets? — `portfolio.py`

Rows in the returns matrix are days; columns are assets. For T rows and N columns:

```python
mean_returns = returns.mean(axis=0) * 252
covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1)) * 252
```

`axis=0` averages down each column. `rowvar=False` tells NumPy that variables
are columns. `ddof=1` divides the centered cross-products by T - 1, giving sample
covariance. `atleast_2d` preserves a 1-by-1 matrix for a single asset.

For two assets:

$$\Sigma=\begin{pmatrix}\sigma_1^2&\sigma_{12}\\\sigma_{12}&\sigma_2^2\end{pmatrix}.$$

Diagonal entries measure individual variances. Off-diagonal entries measure
co-movement. With fixed weights,

$$\sigma_p^2=w_1^2\sigma_1^2+w_2^2\sigma_2^2+2w_1w_2\sigma_{12}.$$

For equal weights, two assets with volatility 20% and covariance -0.02 give
portfolio variance 0.01 and volatility 10%. That is diversification, not an
arithmetic average of the two volatilities.

Multiplying the daily mean and covariance by 252 is an annualization convention,
assuming stable moments and negligible serial correlation. It does not compound
wealth. The README states the matching convention for the risk-free input.

## 4. How do we evaluate a portfolio? — `portfolio_metrics`

For one weight vector, the formulas are:

```python
expected_return = weights @ mean_returns
variance = weights @ covariance @ weights
volatility = np.sqrt(variance)
sharpe = (expected_return - risk_free_rate) / volatility
```

`@` means matrix multiplication. Expected return uses linearity of expectation;
asset independence is not required. Portfolio variance does require covariance,
which is why we cannot average the assets' individual Sharpe ratios.

The implementation handles many weight vectors in one call. If W has shape
(M, N), where M is the number of portfolios, it uses:

```python
variances = np.sum((weights @ covariance) * weights, axis=-1)
```

`weights @ covariance` produces an M-by-N table. Multiplication by `weights`
is elementwise; summing each row completes that row's quadratic form. This
avoids a Python loop over thousands of portfolios without using harder-to-read
index notation such as `einsum`.

Validation checks dimensions, finite values, weights, symmetry, and positive
semidefiniteness. The last property means no portfolio can have negative variance;
`eigvalsh` checks eigenvalues of the symmetric matrix. A singular matrix is fine:
we never invert it. Tiny negative round-off in a computed variance is clipped to
zero after validation.

`np.divide(..., where=volatility > 1e-12)` avoids dividing by zero. Undefined
Sharpe values stay `NaN`. They are shown as undefined and excluded from selection.

**Check:** two identical assets have a singular covariance matrix. Does splitting
your money between them reduce risk? Why does our calculation still work?

## 5. What does Monte Carlo simulate? — `simulation.py`

It explores possible **weights**, not future stock prices. We do not need a
starting portfolio to search the simplex, but importing one allows a meaningful
comparison with the user's current allocation.

```python
rng = np.random.default_rng(seed)
random_weights = rng.dirichlet(np.ones(assets), size=count)
```

Dirichlet(1, ..., 1) generates non-negative vectors summing to 1 uniformly across
the simplex. Dividing independent Uniform(0,1) draws by their sum would produce
a different, non-uniform distribution of weights. The explicit random seed makes
runs reproducible in the same dependency environment.

We prepend the current allocation, equal weights, and the identity matrix.
Each row of the identity matrix invests 100% in one asset. Pure random draws
almost never hit those boundary portfolios exactly.

`portfolio_metrics` evaluates every row. `select_portfolios` selects the largest
finite Sharpe and smallest volatility. Including the current portfolio means the
selected Sharpe cannot be smaller than the current finite Sharpe on this sample.
That comparison says nothing about performance on new data.

If all Sharpe ratios are undefined, we omit the best-Sharpe candidate. If they
are all negative, we still select the largest, but a larger negative ratio is
not evidence of an attractive investment; even the synthetic demo can show this.

**Check:** why can increasing the number of simulations improve search coverage
without fixing a poor estimate of expected returns?

## 6. How are the pieces connected? — `analysis.py`

`analyze_portfolio` is the shared recipe:

1. Get imported holdings and market prices, or the explicit synthetic demo.
2. Align ticker columns and calculate complete daily returns.
3. Compute current weights, annual means, and covariance.
4. Sample and evaluate candidate allocations.
5. Build comparison tables, notes, and run metadata.

It returns a dictionary with named items, including `allocations`, `comparison`,
`prices`, and `settings`. Plain functions plus a dictionary keep inputs and
outputs visible. A larger project might eventually use a typed result object;
this project does not need that machinery yet.

## 7. How do we see results? — `reporting.py`, `app.py`, `main.py`

`reporting.py` consumes already calculated metrics. The scatter plot puts
annualized volatility on the horizontal axis and estimated arithmetic return
on the vertical axis; color represents Sharpe. Markers locate the compared
portfolios. The second plot compares weights. The cloud is labelled as sampled
portfolios, not an exact efficient frontier.

`export_results` saves the tables, original imported holdings, adjusted prices,
valuation closes, retained returns, and run settings. These permit checking the
results later even if Yahoo revises its historical data. Undefined Sharpe values
are blank in CSV. `run.json` contains the period, actual sample dates, seed,
observation count, currency, and annual risk-free rate.

`app.py` only manages upload, controls, preview, execution, and display. Results
remain in the local Streamlit session; changing inputs does not silently claim
that the old result belongs to the new settings. Press **Run optimization** again.
`main.py` provides the same workflow through `argparse` options. Both call the
same `analyze_portfolio` function, so there are no separate mathematical engines.

## 8. How do we know it works? — the tests

Tests use small cases with independently calculable answers:

- `test_returns.py`: first-lesson price ratios and mean returns.
- `test_holdings.py`: CSV/XLSX, Spanish decimals, quantities, percentages, duplicates,
  malformed files, and ticker-to-price alignment.
- `test_market_data.py`: missing rows, separate adjusted/raw prices, currencies,
  unfinished sessions, and service failures.
- `test_portfolio.py`: covariance by hand, quadratic variance, diversification,
  zero risk, singular matrices, and random weight constraints.
- `test_workflow.py`: imports through a simulated Yahoo response, exported results,
  command-line behavior, and Streamlit interactions.

Run `python -m unittest -v`. Automated market-data tests replace only the external
Yahoo boundary with known responses. That makes them reliable offline; it does
not prove Yahoo is reachable from a particular network. Use the real-data terminal
example separately when network access is available.

A useful exercise is to add a test for an allocation of 75%/25% with expected
returns 8%/16%, variances 0.04/0.09, and covariance 0.01. Calculate return, variance,
volatility, and Sharpe at a 2% annual risk-free rate before checking the code.
