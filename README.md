# MSDS451_Assignment02_kshaunishsoni

**Portfolio optimization: a stochastic Monte Carlo with a quadratic volatility constraint**

This assignment looks at what happens when we take long and/or short positions across four assets. The goal is to maximize expected return given an investor's tolerance for risk. The decision variables are the shares of capital put into each asset. Future returns are uncertain, so the problem is solved with stochastic programming. Instead of plugging in point estimates, the optimizer works directly on 10,000 Monte Carlo scenarios of next year's returns.

The study runs on two sets of four assets:

| Asset set | Assets | Source of assumptions |
|---|---|---|
| **Original Assets** | A, B, C, D | Problem setup (R program by Tom Miller, June 20, 2025) |
| **Healthcare Stocks** | LLY, JNJ, ISRG, UNH | 10 years of monthly returns (Oct 2016 – Sep 2026) from yfinance |

## Files

| File | Description |
|---|---|
| `451-assignment02-stochastic-kshaunishsoni.py` | The full study: data, Monte Carlo engine, optimizer, results, plot |
| `451-portfolio-optimization-stochastic-monte-carlo.csv` | Optimal portfolios for every asset set, position rule and risk ceiling |
| `451-portfolio-optimization-stochastic-monte-carlo.tex` | The same table in LaTeX (needs the `booktabs` and `multirow` packages) |
| `451-portfolio-optimization-stochastic-monte-carlo-figure.pdf` | 2×2 figure: opportunity sets, efficient frontiers and optimal portfolios |

## How to run

You need Python 3.10 or later. The study was developed with Python 3.12, numpy 2.4, pandas 3.0, matplotlib 3.11, scipy 1.18 and yfinance 1.7.

```bash
pip install numpy pandas matplotlib scipy yfinance
python 451-assignment02-stochastic-kshaunishsoni.py
```

The script takes about 5 seconds. It needs an internet connection to download prices from Yahoo Finance. It prints its results to the console, writes the CSV, LaTeX and PDF files above to the current directory, and then opens the figure. To run it without a display, such as on a server, set `MPLBACKEND=Agg`.

Results can be reproduced exactly. The scenarios use seed `42` and the random weight sets use seed `9999`. The one exception is the healthcare inputs: Yahoo Finance can revise adjusted prices for dividends and splits, which shifts those estimates slightly.

### Changing the inputs

- **Different stocks:** edit `healthcareTickers`. The code assumes 4 assets (`nAssets = 4`).
- **Different hypothetical assets:** edit `targetMeanVector`, `targetSDVector` and `targetCorMatrix`.
- **Different risk tolerances:** edit `targetVolatilities` in the `assetSets` entry for that asset set.
- **More or fewer scenarios:** edit `numScenarios`. Fewer means more sampling noise in the optimal weights.

## Methods and math

### 1. Base assumptions

For each asset set we need a vector of expected annual returns $\mu$ and a covariance matrix $\Sigma$.

**Original Assets.** The means, standard deviations $\sigma$ and correlation matrix $R$ are given in the problem setup:

$$
\mu = \begin{pmatrix} 0.02 \\ 0.07 \\ 0.15 \\ 0.20 \end{pmatrix}, \quad
\sigma = \begin{pmatrix} 0.05 \\ 0.12 \\ 0.17 \\ 0.25 \end{pmatrix}, \quad
R = \begin{pmatrix} 1 & 0.3 & 0.3 & 0.3 \\ 0.3 & 1 & 0.6 & 0.6 \\ 0.3 & 0.6 & 1 & 0.6 \\ 0.3 & 0.6 & 0.6 & 1 \end{pmatrix}
$$

and the covariance matrix is built as

$$
\Sigma = \operatorname{diag}(\sigma)\, R\, \operatorname{diag}(\sigma).
$$

**Healthcare Stocks.** Adjusted daily closes are resampled to month-end prices $P_t$ and turned into simple monthly returns, $r_t = P_t / P_{t-1} - 1$. There are 120 months. The monthly estimates are then annualized:

$$
\mu = 12\,\bar{r}, \qquad \Sigma = 12\, \widehat{\operatorname{Cov}}(r_t).
$$

Annualizing doesn't change correlations, and it scales volatilities by $\sqrt{12}$.

### 2. Stochastic Monte Carlo engine

We draw $S = 10{,}000$ scenarios of next year's asset returns:

$$
r_s \sim \mathcal{N}(\mu, \Sigma), \qquad s = 1, \dots, S.
$$

Each row of the $S \times 4$ scenario matrix is one possible future year. For any weight vector $w$, the portfolio return in scenario $s$ is $w^\top r_s$. That gives a full distribution of outcomes, not a single number. The script prints the target and simulated means and SDs side by side, to confirm that the simulation matches the assumptions.

### 3. Stochastic program with a quadratic volatility constraint

The optimizer uses the **sample average approximation**: expectations are replaced by averages across the scenarios. For a risk ceiling $\sigma_{\max}$ (the investor's tolerance for risk), it solves

$$
\begin{aligned}
\max_{w} \quad & \frac{1}{S}\sum_{s=1}^{S} w^\top r_s && \text{(mean return across scenarios)} \\
\text{subject to} \quad & \frac{1}{S}\sum_{s=1}^{S} \left(w^\top r_s - w^\top \bar{r}\right)^2 \le \sigma_{\max}^2 && \text{(quadratic volatility constraint)} \\
& \textstyle\sum_i w_i = 1 && \text{(fully invested)} \\
& 0 \le w_i \le 1 && \text{(long positions only, if shorts are not allowed)}
\end{aligned}
$$

Here $\bar{r}$ is the average scenario return vector. The constraint is quadratic because the scenario variance equals $w^\top \hat{\Sigma}_S\, w$, where $\hat{\Sigma}_S$ is the covariance of the scenarios. That makes the feasible region an ellipsoid in weight space, cut by the budget plane. The objective is linear, so the optimum lands on the edge of the ellipsoid: the volatility constraint is binding, unless a corner such as 100% in one asset is reached first.

Each problem is solved with SciPy's SLSQP solver, starting from equal weights, under two position rules:

- **Long Positions Only:** $0 \le w_i \le 1$
- **Shorts OK:** no bounds on $w_i$, so a negative weight is a short sale. The volatility constraint alone keeps the solution bounded.

The risk ceilings are 5/10/15/20% for the Original Assets and 17/20/25/30% for the Healthcare Stocks. Each set is matched to the volatility range of its assets.

### 4. Monte Carlo opportunity set and efficient frontier

To show what the optimizer is choosing from, 700 random weight sets are scored on the **same** scenarios:

- **Long only:** $u_i \sim U(0,1)$, rescaled as $w_i = u_i / \sum_j u_j$.
- **Shorts OK:** $w_1, w_2, w_3 \sim U(-1, 1)$, and $w_4 = 1 - (w_1 + w_2 + w_3)$.

Each random portfolio is plotted at its simulated (volatility, mean return).

The **stochastic efficient frontier** is traced by sweeping $\sigma_{\max}$ across 40 values and solving the stochastic program at each one. The sweep runs from just above the minimum attainable volatility:

$$
\sigma_{\min} = \min_{w} \sqrt{w^\top \hat{\Sigma}_S\, w} \quad \text{s.t.} \quad \textstyle\sum_i w_i = 1.
$$

### 5. Risk measures reported

Each optimal portfolio is summarized by statistics of its 10,000 simulated returns $x_s = w^\top r_s$:

| Column | Definition | Meaning |
|---|---|---|
| `w1`–`w4` | Optimal weights, in asset order | Share of capital in each asset; negative means a short position |
| `Simulated Return` | $\bar{x}$ | Expected annual return |
| `Simulated Vol` | $\operatorname{sd}(x)$ | Annual volatility; equals the ceiling when the constraint is binding |
| `Sharpe` | $\bar{x} / \operatorname{sd}(x)$ | Return per unit of risk (risk-free rate set to 0) |
| `Prob. Loss` | $\frac{1}{S}\#\{x_s < 0\}$ | Chance of a losing year |
| `5% VaR` | $-\text{5th percentile of } x$ | The loss exceeded in the worst 1 year in 20 |

## How to interpret the output

### Reading the figure

Each row is an asset set and each column is a position rule. Rows have their own axis scales.

- **Dots** are the 700 random portfolios. Blue dots have no short positions and red dots have at least one. Most of them are suboptimal, because they sit below the frontier.
- **The black line** is the stochastic efficient frontier: the highest expected return attainable at each level of risk. No portfolio can sit above it.
- **Gold stars** are the optimal portfolios for each risk ceiling. They should always sit on the frontier. If a star sits to the left of its ceiling, a corner was reached first, as with 100% LLY.
- **Comparing columns** shows what short selling buys you. Where the Shorts OK frontier lies above the long-only frontier, shorting adds return at that level of risk.

### Reading the table

1. **Compare Sharpe ratios across ceilings.** The ceiling with the highest Sharpe is the best risk-adjusted portfolio. Beyond it, each extra point of return costs more than a point of risk.
2. **Check whether Shorts OK differs from Long Positions Only.** If the weights are identical, short selling gains nothing at that ceiling.
3. **Look at the downside columns (Prob. Loss, 5% VaR),** not only volatility. Two portfolios can have similar volatility but different chances of a losing year.
4. **Look at concentration.** A weight near 1 means diversification is gone, and the result rests on a single asset's estimated return.

### Key findings (current run)

**Healthcare Stocks** (w1–w4 = LLY, JNJ, ISRG, UNH)

| Ceiling | Long-only weights | Return | Sharpe | Prob. Loss | Shorts OK return |
|---|---|---|---|---|---|
| 17% | .33 / .40 / .11 / .16 | 20.2% | 1.19 | 11.6% | same |
| **20%** | **.54 / .17 / .13 / .16** | **24.5%** | **1.23** | **10.6%** | same |
| 25% | .81 / .00 / .12 / .07 | 29.6% | 1.18 | 11.6% | 29.7% (short JNJ −.11) |
| 30% | 1.00 / 0 / 0 / 0 | 32.1% | 1.12 | 13.0% | 34.2% (short JNJ −.35) |

**Recommended: the long-only portfolio with a 20% volatility ceiling.** It has the highest Sharpe ratio and the lowest chance of a losing year of all eight healthcare portfolios, and it needs no short selling. Moving the ceiling from 17% to 20% adds 4.3 points of return with almost no extra downside: the VaR goes from 7.4% to 7.7%. Above 20%, the Sharpe falls and the portfolio concentrates in LLY.

**Original Assets** (w1–w4 = A, B, C, D)

| Ceiling | Long-only weights | Return | Sharpe | Shorts OK return |
|---|---|---|---|---|
| 5% | .93 / .07 / 0 / 0 | 2.4% | 0.47 | 2.5% |
| 10% | .50 / 0 / .36 / .15 | 9.2% | 0.92 | 9.2% |
| **15%** | **.17 / 0 / .57 / .26** | **14.0%** | **0.93** | 14.0% (short B −.12) |
| 20% | 0 / 0 / .40 / .60 | 17.9% | 0.89 | 18.6% (short B −.20) |

For A–D, the 15% ceiling gives the best risk-adjusted return. The optimizer never holds asset B. C pays more per unit of risk, and B's 0.6 correlation with C and D means it adds little diversification. When shorts are allowed, B becomes the asset sold short to fund larger positions in C and D.

**Across both sets:** short selling adds little, less than 1 point of return, except at the highest ceilings. And that's before borrow costs and margin requirements, which the model doesn't include.

## Limitations

- **The model captures uncertainty in returns, not in the estimates.** Every scenario is drawn from the same $\mu$ and $\Sigma$, so errors in those inputs carry straight into the weights. With 10,000 scenarios, the solution closely matches the deterministic mean-variance answer.
- **The healthcare results depend on one decade.** LLY's 32% annualized return came from an exceptional period. If future returns are lower, the LLY-heavy portfolios at the 25–30% ceilings carry the most risk.
- **Returns are assumed normally distributed.** Real returns have fatter tails, so the true Prob. Loss and VaR are likely somewhat worse than reported.
- **It's a single-period model.** There's no rebalancing, and no transaction costs, taxes or short-sale costs.
- **The risk-free rate is 0 in the Sharpe ratios.** A positive rate lowers every Sharpe ratio but doesn't change the ranking.
