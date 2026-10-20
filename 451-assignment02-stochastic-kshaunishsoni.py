# Portfolio Optimization: Stochastic Monte Carlo with a Quadratic
# Volatility Constraint

# The base assumptions (expected annual returns and covariance matrix) are
# estimated from 10 years of real monthly returns downloaded with yfinance.
# A Monte Carlo engine then generates many scenarios of next year's returns
# from those assumptions, and the optimizer works directly on the scenarios
# (stochastic programming with the sample average approximation):
#   maximize    (1/S) sum_s w'r_s                (mean return across scenarios)
#   subject to  var_s(w'r_s) <= target_vol^2     (quadratic volatility constraint)
#               sum(w) = 1                       (fully invested)
#               w >= 0                           (long positions only)
# The same scenarios are used to evaluate random Monte Carlo weight sets,
# so the optimal portfolios can be compared with the opportunity set.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# show all columns when printing summaries
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)

# --- 1. BASE ASSUMPTIONS (from historical data) ---
tickers = ["LLY", "JNJ", "ISRG", "UNH"]
nAssets = len(tickers)

# 10 years of monthly returns: the September 2016 month-end price is the
# base for the first return (October 2016); yfinance treats end as exclusive
startDate = "2016-09-01"
endDate = "2026-10-01"
periodsPerYear = 12  # monthly returns

# download adjusted (dividend- and split-adjusted) prices
prices = yf.download(tickers, start=startDate, end=endDate,
                     auto_adjust=True, progress=False)["Close"]
prices = prices[tickers]  # keep columns in ticker order
prices = prices.dropna()  # align: keep only dates where every asset has a price

# month-end prices to monthly simple returns
monthlyPrices = prices.resample("ME").last()
historicalReturns = monthlyPrices.pct_change().dropna()
print(f"Historical monthly returns: {len(historicalReturns)} months, "
      f"{historicalReturns.index[0]:%Y-%m} to {historicalReturns.index[-1]:%Y-%m}")

# annualized mean returns and covariance matrix
mu = historicalReturns.mean().to_numpy() * periodsPerYear
covMatrix = historicalReturns.cov().to_numpy() * periodsPerYear

print("\nAnnualized statistics from historical data:")
print(pd.DataFrame({"Mean": mu, "SD": np.sqrt(np.diag(covMatrix))}, index=tickers))
print("\nCorrelation of historical returns:")
print(historicalReturns.corr())

# --- 2. THE STOCHASTIC MONTE CARLO ENGINE ---
numScenarios = 10000
np.random.seed(42)  # locked for reproducibility

# Each row is one possible future: a year of returns for every asset.
print(f"\nGenerating {numScenarios} Monte Carlo market scenarios...")
scenarios = np.random.multivariate_normal(mu, covMatrix, numScenarios)

print("\nTarget vs. simulated annual means and SDs:")
print(pd.DataFrame({"Target Mean": mu, "Simulated Mean": scenarios.mean(axis=0),
                    "Target SD": np.sqrt(np.diag(covMatrix)),
                    "Simulated SD": scenarios.std(axis=0)}, index=tickers))


def simulatedMean(w):
    return np.mean(scenarios @ w)


def simulatedVol(w):
    return np.std(scenarios @ w)


# --- 3. THE QUADRATIC OPTIMIZER ---
# Risk ceilings: the investor's tolerance for risk as a maximum annual volatility
targetVolatilities = [0.17, 0.20, 0.25, 0.30]


def optimizePortfolio(targetVolatility, shortsOK):
    def objectiveFunction(w):
        # maximize the average return across all scenarios
        return -simulatedMean(w)

    def riskConstraint(w):
        # quadratic constraint: simulated variance must not exceed the
        # target variance (SciPy requires the value to be >= 0)
        return targetVolatility ** 2 - np.var(scenarios @ w)

    budgetConstraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    varianceLimit = {"type": "ineq", "fun": riskConstraint}
    bounds = None if shortsOK else [(0.0, 1.0)] * nAssets
    initialGuess = np.full(nAssets, 1 / nAssets)  # start equal-weight
    return minimize(objectiveFunction, initialGuess, method="SLSQP",
                    bounds=bounds, constraints=[budgetConstraint, varianceLimit])


positionConditions = {"Long Positions Only": False, "Shorts OK": True}

print("\nRunning SLSQP optimizer over simulated scenarios...")
optimalRows = {}
for conditionName, shortsOK in positionConditions.items():
    for targetVolatility in targetVolatilities:
        result = optimizePortfolio(targetVolatility, shortsOK)
        name = f"Vol <= {targetVolatility:.0%}"
        if not result.success:  # e.g., ceiling below the minimum attainable volatility
            print(f"{conditionName}, {name}: optimization failed: {result.message}")
            continue
        w = result.x
        portfolioReturns = scenarios @ w
        row = {f"w_{ticker}": weight for ticker, weight in zip(tickers, w)}
        row.update({"Simulated Return": portfolioReturns.mean(),
                    "Simulated Vol": portfolioReturns.std(),
                    "Sharpe": portfolioReturns.mean() / portfolioReturns.std(),
                    "Prob. Loss": (portfolioReturns < 0).mean(),
                    "5% VaR": -np.percentile(portfolioReturns, 5)})
        optimalRows[(conditionName, name)] = row

# --- 4. RESULTS ---
optimalTable = pd.DataFrame(optimalRows).T
optimalTable.index.names = ["Positions", "Risk Ceiling"]
print("\n=== OPTIMAL PORTFOLIOS (stochastic, quadratic volatility constraint) ===")
print(optimalTable)

# export as CSV and as LaTeX (for inclusion in LaTeX document alongside the
# pdf plot); escapes LaTeX special characters (requires booktabs and multirow)
stem = "451-portfolio-optimization-stochastic-monte-carlo"
optimalTable.to_csv(f"{stem}.csv", float_format="%.4f")
optimalTable.rename(
    columns=lambda name: name.replace("_", r"\_").replace("%", r"\%"),
    index=lambda name: name.replace("<=", r"$\leq$").replace("%", r"\%")
).to_latex(f"{stem}.tex", float_format="%.4f",
           caption="Stochastic Monte Carlo optimal portfolios with a quadratic volatility constraint",
           label="tab:stochastic-monte-carlo")

# --- 5. MONTE CARLO OPPORTUNITY SET ON THE SAME SCENARIOS ---
# Random weight sets, each summing to 1. With shorts, the first N - 1
# weights are uniform from -1 to 1 and the last makes the sum 1.
numWeightSets = 700
np.random.seed(9999)  # set seed so results are reproducible


def makeWeights(shortsOK):
    if shortsOK:
        firstWeights = np.random.uniform(low=-1, high=1, size=nAssets - 1)
        return np.append(firstWeights, 1 - firstWeights.sum())
    initialWeights = np.random.uniform(low=0, high=1, size=nAssets)
    return initialWeights / initialWeights.sum()


# Efficient frontier: sweep the volatility ceiling from just above the
# minimum attainable volatility, solving the stochastic program at each
frontierVolatilities = np.linspace(0.16, 0.40, 40)

fig, axes = plt.subplots(1, len(positionConditions), figsize=(11, 5.5),
                         sharex=True, sharey=True)
for ax, (conditionName, shortsOK) in zip(axes, positionConditions.items()):
    weightSets = np.array([makeWeights(shortsOK) for _ in range(numWeightSets)])
    portfolioReturns = scenarios @ weightSets.T  # scenarios x weight sets
    hasShorts = (weightSets < 0).any(axis=1)
    for mask, label, color in [(hasShorts, "Has Short(s)", "red"),
                               (~hasShorts, "No Shorts", "darkblue")]:
        ax.scatter(portfolioReturns.std(axis=0)[mask],
                   portfolioReturns.mean(axis=0)[mask], s=4, color=color, label=label)

    frontier = []
    for targetVolatility in frontierVolatilities:
        result = optimizePortfolio(targetVolatility, shortsOK)
        if result.success:
            frontier.append((simulatedVol(result.x), simulatedMean(result.x)))
    frontier = np.array(frontier)
    ax.plot(frontier[:, 0], frontier[:, 1], color="black", linewidth=1.5,
            label="Stochastic Efficient Frontier")

    optima = optimalTable.loc[conditionName]
    ax.scatter(optima["Simulated Vol"], optima["Simulated Return"], marker="*",
               s=120, color="gold", edgecolor="black", zorder=3,
               label="Optimum for Risk Ceiling")
    ax.set_title(f"Simulated Returns: {conditionName}")
    ax.set_xlabel("Risk: Standard Deviation of Portfolio Returns", fontsize=12)
axes[0].set_ylabel("Return: Mean of Portfolio Returns", fontsize=12)
fig.suptitle(f"Stochastic Monte Carlo Portfolio Optimization ({numScenarios} scenarios): "
             f"{', '.join(tickers)}")
fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=4)
fig.tight_layout(rect=(0, 0.07, 1, 1))

# export pdf plot for inclusion in LaTeX document
fig.savefig(f"{stem}-figure.pdf")
plt.show()

print("\nRun Complete")
