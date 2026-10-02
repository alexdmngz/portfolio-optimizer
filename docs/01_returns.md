# Step 1: From prices to returns

## Learning contract

Keep the implementation small and readable. Discuss the mathematics before each
new calculation. Explain one module at a time and review the learner's answers
and tests before adding the next module. Explanations in conversation are in
Spanish; repository code and documentation are in English.

## Mathematics

For consecutive prices, a simple return is:

$$r_t = \frac{P_t}{P_{t-1}} - 1.$$

Prices of 100, 110, and 99 produce returns of 0.10 and -0.10. We store decimals,
so 0.10 means 10%. Prices must be chronological, equally spaced, and adjusted
consistently for corporate actions. This module receives values without dates;
the caller is responsible for chronology, frequency, and adjustments.

The unknown next-period expected return is $\mu = \mathbb{E}[R]$. Our first
estimator is the historical arithmetic mean:

$$\hat{\mu} = \frac{1}{T}\sum_{t=1}^{T}r_t.$$

For the example, this mean is 0%. The realized cumulative return is instead
$(1+0.10)(1-0.10)-1=-0.01$, or -1%. An average of periodic returns and compounded
growth answer different questions. Historical averages are estimates, not promised
future returns. Monthly input produces a monthly estimate; we do not annualize yet.

## The production code, block by block

### 1. Import NumPy

`import numpy as np` gives us numerical arrays. Arithmetic between equally sized
arrays acts element by element. NumPy is our only external dependency at this step.

### 2. Convert the input

`prices = np.asarray(prices, dtype=float)` accepts a list or array and creates a
consistent numerical representation. `dtype=float` enables decimal values.
The same conversion is used for the mean estimator. Inputs must be numeric.

### 3. Validate before calculating

Each `if` states one rule and raises `ValueError` when it is broken:

- `ndim != 1`: this first module accepts one asset's sequence, not a table.
- `size < 2`: a return needs both a previous and a current price.
- `np.isfinite`: rejects missing values (`NaN`) and infinities.
- `prices <= 0`: our initial asset-price model requires strictly positive prices;
  rejecting zero also prevents division by zero.

`np.all` asks whether every element passes a check; `np.any` asks whether at least
one does. Missing prices are rejected rather than removed: removing one could
combine several time periods into a single return while losing that information.
Cleaning and date alignment will belong to the future data-loading module.

### 4. Align consecutive prices

```python
previous_prices = prices[:-1]
current_prices = prices[1:]
returns = current_prices / previous_prices - 1
```

For `[100, 110, 99]`, `prices[:-1]` takes all prices except the last, giving
`[100, 110]`. `prices[1:]` takes all except the first, giving `[110, 99]`.
Elementwise division therefore calculates `[110 / 100, 99 / 110]`; subtracting
one produces `[0.10, -0.10]`. Three prices yield two returns.

The named intermediate variables make the formula visible. We do not modify the
input array or round returns during calculations.

### 5. Estimate the mean

`estimate_mean_return` rejects an empty or non-finite sequence, then uses
`np.mean(returns)`. `float(...)` returns an ordinary Python floating-point number.
Negative and zero returns are valid. There is no price-positivity check here
because returns are different quantities from prices.

### 6. Run the demonstration

`main.py` contains the hypothetical input, calls the two functions, and prints the
results. The mathematical module performs no input/output. The condition
`if __name__ == "__main__"` runs the demonstration when the file is executed,
while allowing imports without printing anything. The `:.2%` display format
shows a decimal as a percentage with two decimal places.

### 7. Check behavior with tests

`test_returns.py` uses Python's built-in `unittest` framework. Each method whose
name starts with `test_` describes one behavior. The `ReturnsTests` class is simply
the grouping required by this framework; production code uses ordinary functions.

We compare known inputs with hand-calculated answers. Floating-point comparisons
use `assertAlmostEqual` or `np.testing.assert_allclose`, because binary floating
point can introduce small rounding differences. `with self.assertRaises(ValueError)`
checks that an invalid input is rejected. A missing value is not replaced with zero.

## Small architecture that can grow

Only the returns module is implemented now. Files below marked "planned" will be
introduced when needed; they are not empty scaffolding in the repository.

| File | Responsibility | Why it is separate |
| --- | --- | --- |
| `returns.py` | Simple returns and their historical mean | Pure calculations can be tested without downloads or charts. |
| `main.py` | Assemble and run an example | Keeps input/output outside mathematical functions. |
| `test_returns.py` | Verify the current mathematical contract | Hand-calculated cases make failures understandable. |
| `data.py` (planned) | Read, validate, and align a CSV price table | A later API integration can change without changing the mathematics. |
| `portfolio.py` (planned) | Covariance, portfolio return, volatility, Sharpe ratio | Groups portfolio-level formulas together. |
| `simulation.py` (planned) | Sample valid portfolio weights and evaluate them | Separates random sampling from financial formulas. |
| `visualization.py` (planned) | Plot risk versus expected return | Charts consume results rather than recalculate them. |

Use NumPy for numerical work. Add a CSV library and plotting dependency when those
steps need them. Keep production functions small, names explicit, and inputs and
outputs visible. Avoid introducing classes or generic frameworks without a clear need.

## Next steps, after understanding checks

1. Explain covariance, portfolio variance, Sharpe ratio, and Monte Carlo sampling
   before implementing those calculations.
2. Add each mathematical function with its tests, one small block at a time.
3. Connect a small reproducible CSV dataset.
4. Sample long-only, fully invested portfolios with a reproducible random seed.
5. Plot the results and identify the best Sharpe ratio among sampled portfolios.
6. Finish the professional English README, including limitations and improvements.

Sampling portfolios explores candidate weights; it does not guarantee the exact
global optimum. We will explain that limitation when introducing Monte Carlo.

## Your checkpoint

1. Run `python main.py` and `python -m unittest -v`.
2. Explain why three prices produce two returns and why the example's arithmetic
   mean differs from its cumulative return.
3. Add a test for prices `[80, 100, 90]`, computing the expected returns by hand.
4. Add a test for a one-element return sequence and predict its estimated mean.

Review these answers and tests before moving to covariance.

## Reference documentation

- [NumPy array conversion](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)
- [NumPy finite-value checks](https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html)
