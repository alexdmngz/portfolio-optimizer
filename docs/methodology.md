# Calculation notes

For each asset, daily returns use adjusted closing prices:

$$r_{i,t}=\frac{P^{adj}_{i,t}}{P^{adj}_{i,t-1}}-1.$$

Returns are calculated before incomplete rows are removed. Otherwise, a missing
price could turn a two-day change into a one-day observation. All assets use the
same retained dates. The program reports the number of excluded rows; it cannot
detect a date missing from every asset without an external exchange calendar.
Sessions are aligned by local date, not by simultaneous intraday prices.

For $T$ daily observations and $D=252$ sessions per year:

$$\hat\mu_i=\frac{D}{T}\sum_{t=1}^{T}r_{i,t},$$

$$\hat\Sigma_{ij}=\frac{D}{T-1}\sum_{t=1}^{T}(r_{i,t}-\bar r_i)(r_{j,t}-\bar r_j).$$

The annual return estimate is an arithmetic mean, not CAGR. Multiplying daily
covariance by 252 assumes stable moments and negligible serial correlation.
Historical means are particularly sensitive to the estimation window.

For portfolio weights $w_i\geq0$ and $\sum_i w_i=1$:

$$\hat\mu_p=w^T\hat\mu,\qquad
\hat\sigma_p=\sqrt{w^T\hat\Sigma w},\qquad
\hat S_p=\frac{\hat\mu_p-r_f}{\hat\sigma_p}.$$

The risk-free rate uses the same annual arithmetic convention and quote currency.
For an effective annual yield $y$, a matching rate under a constant daily-rate
assumption is `252 * ((1 + y)**(1/252) - 1)`. The default zero is an input default,
not a market quote.

Covariance must be symmetric and positive semidefinite. Singular matrices are
allowed because the calculation does not invert them. Sharpe is undefined when
volatility is at most `1e-12`; those candidates are excluded from Sharpe selection.
When all excess returns are negative, the largest Sharpe ratio can be a poor
ranking criterion.

## Weights and sampling

Uploaded quantities are valued at their latest unadjusted closes:

$$V_i=q_iP_i^{close},\qquad w_i=\frac{V_i}{\sum_j V_j}.$$

These weights describe the uploaded quantities at that closing date. Imported
weights are used directly, since their original valuation date is unknown.
Zero positions remain available to the allocation search.

Random allocations follow `Dirichlet(1, ..., 1)`, which is uniform on the long-only
simplex. The current allocation, equal weights and every single-asset allocation
are included explicitly. The same seed reproduces the draws in the same software
environment.

Sampling explores allocations, not future price paths. It becomes less effective
as the number of assets grows, and does not solve the continuous optimization
problem. Estimation and selection use the same observations; out-of-sample
performance is not measured.

References: [NumPy covariance](https://numpy.org/doc/stable/reference/generated/numpy.cov.html),
[Dirichlet sampling](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.dirichlet.html),
[Sharpe's original discussion](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm),
[yfinance documentation and data-use notice](https://ranaroussi.github.io/yfinance/).
