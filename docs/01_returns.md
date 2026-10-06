# Step 1: From prices to returns

## About this lesson

This was the project's first implemented module. The complete optimizer is now
available; this page preserves the single-asset foundations. Read
[the full walkthrough](02_walkthrough.md) after this lesson. Code and repository
documentation are in English; mentoring explanations are in Spanish.

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

### 6. Run the original calculation

You can reproduce the original small example in a Python session:

```python
from returns import calculate_returns, estimate_mean_return
returns = calculate_returns([100, 110, 99])
print(returns)
print(estimate_mean_return(returns))
```

`main.py` now runs the complete optimizer. Its `--demo` mode uses a larger,
explicitly synthetic multi-asset dataset; this lesson's functions remain
independent and keep their original tests.

### 7. Check behavior with tests

`test_returns.py` uses Python's built-in `unittest` framework. Each method whose
name starts with `test_` describes one behavior. The `ReturnsTests` class is simply
the grouping required by this framework; production code uses ordinary functions.

We compare known inputs with hand-calculated answers. Floating-point comparisons
use `assertAlmostEqual` or `np.testing.assert_allclose`, because binary floating
point can introduce small rounding differences. `with self.assertRaises(ValueError)`
checks that an invalid input is rejected. A missing value is not replaced with zero.

## From this module to the complete program

The [README](../README.md) describes the implemented architecture. The complete
workflow separates holdings extraction, market prices, portfolio mathematics,
Monte Carlo sampling, and presentation. [The next guide](02_walkthrough.md)
explains the formulas and the corresponding code blocks.

The first lesson rejects missing input outright. The multi-asset data module
calculates adjacent returns without filling missing prices, then retains rows
with valid returns for every asset. It reports the omitted observations.

## Your checkpoint

1. Run `python main.py` and `python -m unittest -v`.
2. Explain why three prices produce two returns and why the example's arithmetic
   mean differs from its cumulative return.
3. Add a test for prices `[80, 100, 90]`, computing the expected returns by hand.
4. Add a test for a one-element return sequence and predict its estimated mean.

Use these checks to review the foundations, then study covariance in the full walkthrough.

## Reference documentation

- [NumPy array conversion](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)
- [NumPy finite-value checks](https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html)
