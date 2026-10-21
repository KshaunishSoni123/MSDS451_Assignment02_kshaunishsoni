# Portfolio Optimization: Stochastic Monte Carlo with a Quadratic
# Volatility Constraint

# The study is run for two sets of four assets:
#   1. Original Assets: hypothetical assets A, B, C, and D with the means,
#      standard deviations, and correlations from the problem setup
#      (R program prepared by Tom Miller, June 20, 2025).
#   2. Healthcare Stocks: LLY, JNJ, ISRG, and UNH, with expected annual
#      returns and covariance matrix estimated from 10 years of real monthly
#      returns downloaded with yfinance.
# For each, a Monte Carlo engine generates many scenarios of next year's
# returns, and the optimizer works directly on the scenarios (stochastic
# programming with the sample average approximation):
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

# --- 1a. BASE ASSUMPTIONS: ORIGINAL ASSETS (problem setup) ---
originalTickers = ["A", "B", "C", "D"]
# mean returns for portfolio assets A, B, C, and D
targetMeanVector = np.array([0.02, 0.07, 0.15, 0.20])
# standard deviations of returns
targetSDVector = np.array([0.05, 0.12, 0.17, 0.25])
# correlation matrix for the four portfolio assets
targetCorMatrix = np.array([[1, 0.3, 0.3, 0.3],
                            [0.3, 1, 0.6, 0.6],
                            [0.3, 0.6, 1, 0.6],
                            [0.3, 0.6, 0.6, 1]])
# compute the covariance matrix
targetCovMatrix = np.diag(targetSDVector) @ targetCorMatrix @ np.diag(targetSDVector)

# --- 1b. BASE ASSUMPTIONS: HEALTHCARE STOCKS (historical data) ---
healthcareTickers = ["LLY", "JNJ", "ISRG", "UNH"]

# 10 years of monthly returns: the September 2016 month-end price is the
# base for the first return (October 2016); yfinance treats end as exclusive
startDate = "2016-09-01"
endDate = "2026-10-01"
periodsPerYear = 12  # monthly returns

# download adjusted (dividend- and split-adjusted) prices
prices = yf.download(healthcareTickers, start=startDate, end=endDate,
                     auto_adjust=True, progress=False)["Close"]
prices = prices[healthcareTickers]  # keep columns in ticker order
prices = prices.dropna()  # align: keep only dates where every asset has a price

# month-end prices to monthly simple returns
monthlyPrices = prices.resample("ME").last()
historicalReturns = monthlyPrices.pct_change().dropna()
print(f"Historical monthly returns: {len(historicalReturns)} months, "
      f"{historicalReturns.index[0]:%Y-%m} to {historicalReturns.index[-1]:%Y-%m}")

# annualized mean returns and covariance matrix
healthcareMeanVector = historicalReturns.mean().to_numpy() * periodsPerYear
healthcareCovMatrix = historicalReturns.cov().to_numpy() * periodsPerYear

# Each asset set has its own risk ceilings (the investor's tolerance for
# risk as a maximum annual volatility), scaled to the assets' volatilities,
# and the highest volatility shown on its efficient frontier
assetSets = {
    "Original Assets": {"tickers": originalTickers,
                        "mu": targetMeanVector, "covMatrix": targetCovMatrix,
                        "targetVolatilities": [0.05, 0.10, 0.15, 0.20],
                        "frontierMaxVolatility": 0.70},
    "Healthcare Stocks": {"tickers": healthcareTickers,
                          "mu": healthcareMeanVector, "covMatrix": healthcareCovMatrix,
                          "targetVolatilities": [0.17, 0.20, 0.25, 0.30],
                          "frontierMaxVolatility": 0.40},
}
nAssets = 4
positionConditions = {"Long Positions Only": False, "Shorts OK": True}
numScenarios = 10000
numWeightSets = 700


# --- 2. THE QUADRATIC OPTIMIZER ---
def optimizePortfolio(scenarios, targetVolatility, shortsOK):
    def objectiveFunction(w):
        # maximize the average return across all scenarios
        return -np.mean(scenarios @ w)

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


def minimumVolatility(scenarios, shortsOK):
    # lowest attainable volatility: no risk ceiling can be set below this
    budgetConstraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = None if shortsOK else [(0.0, 1.0)] * nAssets
    result = minimize(lambda w: np.var(scenarios @ w), np.full(nAssets, 1 / nAssets),
                      method="SLSQP", bounds=bounds, constraints=[budgetConstraint])
    return np.std(scenarios @ result.x)


# Random weight sets, each summing to 1. With shorts, the first N - 1
# weights are uniform from -1 to 1 and the last makes the sum 1.
def makeWeights(shortsOK):
    if shortsOK:
        firstWeights = np.random.uniform(low=-1, high=1, size=nAssets - 1)
        return np.append(firstWeights, 1 - firstWeights.sum())
    initialWeights = np.random.uniform(low=0, high=1, size=nAssets)
    return initialWeights / initialWeights.sum()


# complete plot: rows are asset sets, columns are position conditions;
# each row has its own axis scales because the asset sets differ in risk
fig, axes = plt.subplots(len(assetSets), len(positionConditions), figsize=(11, 10),
                         sharex="row", sharey="row")
optimalTables = {}
for irow, (assetSetName, assetSet) in enumerate(assetSets.items()):
    tickers = assetSet["tickers"]
    mu = assetSet["mu"]
    covMatrix = assetSet["covMatrix"]
    print(f"\n========== {assetSetName}: {', '.join(tickers)} ==========")
    print("\nAnnualized means and standard deviations:")
    print(pd.DataFrame({"Mean": mu, "SD": np.sqrt(np.diag(covMatrix))}, index=tickers))
    print("\nCorrelation matrix:")
    sdVector = np.sqrt(np.diag(covMatrix))
    print(pd.DataFrame(covMatrix / np.outer(sdVector, sdVector),
                       index=tickers, columns=tickers))

    # --- 3. THE STOCHASTIC MONTE CARLO ENGINE ---
    np.random.seed(42)  # locked for reproducibility
    # Each row is one possible future: a year of returns for every asset.
    print(f"\nGenerating {numScenarios} Monte Carlo market scenarios...")
    scenarios = np.random.multivariate_normal(mu, covMatrix, numScenarios)

    print("\nTarget vs. simulated annual means and SDs:")
    print(pd.DataFrame({"Target Mean": mu, "Simulated Mean": scenarios.mean(axis=0),
                        "Target SD": sdVector,
                        "Simulated SD": scenarios.std(axis=0)}, index=tickers))

    print("\nRunning SLSQP optimizer over simulated scenarios...")
    optimalRows = {}
    for conditionName, shortsOK in positionConditions.items():
        for targetVolatility in assetSet["targetVolatilities"]:
            result = optimizePortfolio(scenarios, targetVolatility, shortsOK)
            name = f"Vol <= {targetVolatility:.0%}"
            if not result.success:  # e.g., ceiling below the minimum attainable volatility
                print(f"{conditionName}, {name}: optimization failed: {result.message}")
                continue
            portfolioReturns = scenarios @ result.x
            # weights are labeled by position (Asset 1-4) so that both asset
            # sets share one results table; tickers are given in each caption
            row = {f"w{i + 1}": weight for i, weight in enumerate(result.x)}
            row.update({"Simulated Return": portfolioReturns.mean(),
                        "Simulated Vol": portfolioReturns.std(),
                        "Sharpe": portfolioReturns.mean() / portfolioReturns.std(),
                        "Prob. Loss": (portfolioReturns < 0).mean(),
                        "5% VaR": -np.percentile(portfolioReturns, 5)})
            optimalRows[(conditionName, name)] = row

    # --- 4. RESULTS ---
    optimalTable = pd.DataFrame(optimalRows).T
    optimalTable.index.names = ["Positions", "Risk Ceiling"]
    optimalTables[assetSetName] = optimalTable
    print(f"\n=== OPTIMAL PORTFOLIOS: {assetSetName} "
          f"(w1-w4 = {', '.join(tickers)}) ===")
    print(optimalTable)

    # --- 5. MONTE CARLO OPPORTUNITY SET ON THE SAME SCENARIOS ---
    np.random.seed(9999)  # set seed so results are reproducible
    for icol, (conditionName, shortsOK) in enumerate(positionConditions.items()):
        ax = axes[irow, icol]
        weightSets = np.array([makeWeights(shortsOK) for _ in range(numWeightSets)])
        portfolioReturns = scenarios @ weightSets.T  # scenarios x weight sets
        hasShorts = (weightSets < 0).any(axis=1)
        for mask, label, color in [(hasShorts, "Has Short(s)", "red"),
                                   (~hasShorts, "No Shorts", "darkblue")]:
            ax.scatter(portfolioReturns.std(axis=0)[mask],
                       portfolioReturns.mean(axis=0)[mask], s=4, color=color, label=label)

        # Efficient frontier: sweep the volatility ceiling from the minimum
        # attainable volatility, solving the stochastic program at each
        frontier = []
        for targetVolatility in np.linspace(minimumVolatility(scenarios, shortsOK) + 0.001,
                                            assetSet["frontierMaxVolatility"], 40):
            result = optimizePortfolio(scenarios, targetVolatility, shortsOK)
            if result.success:
                frontierReturns = scenarios @ result.x
                frontier.append((frontierReturns.std(), frontierReturns.mean()))
        frontier = np.array(frontier)
        ax.plot(frontier[:, 0], frontier[:, 1], color="black", linewidth=1.5,
                label="Stochastic Efficient Frontier")

        optima = optimalTable.loc[conditionName]
        ax.scatter(optima["Simulated Vol"], optima["Simulated Return"], marker="*",
                   s=120, color="gold", edgecolor="black", zorder=3,
                   label="Optimum for Risk Ceiling")
        ax.set_title(f"{assetSetName} ({', '.join(tickers)}): {conditionName}",
                     fontsize=10)
    axes[irow, 0].set_ylabel("Return: Mean of Portfolio Returns", fontsize=12)
for ax in axes[-1, :]:
    ax.set_xlabel("Risk: Standard Deviation of Portfolio Returns", fontsize=12)
fig.suptitle(f"Stochastic Monte Carlo Portfolio Optimization ({numScenarios} scenarios)")
fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="lower center", ncol=4)
fig.tight_layout(rect=(0, 0.04, 1, 1))

# combine the results into one table and export it as CSV and as LaTeX
# (for inclusion in LaTeX document alongside the pdf plot); escapes LaTeX
# special characters (requires booktabs and multirow packages)
stem = "451-portfolio-optimization-stochastic-monte-carlo"
optimalTable = pd.concat(optimalTables, names=["Asset Set", "Positions", "Risk Ceiling"])
optimalTable.to_csv(f"{stem}.csv", float_format="%.4f")
assetLabels = "; ".join(f"{name}: w1-w4 = {', '.join(assetSet['tickers'])}"
                        for name, assetSet in assetSets.items())
optimalTable.rename(
    columns=lambda name: name.replace("%", r"\%"),
    index=lambda name: name.replace("<=", r"$\leq$").replace("%", r"\%")
).to_latex(f"{stem}.tex", float_format="%.4f",
           caption="Stochastic Monte Carlo optimal portfolios with a quadratic "
                   f"volatility constraint ({assetLabels})",
           label="tab:stochastic-monte-carlo")

# export pdf plot for inclusion in LaTeX document
fig.savefig(f"{stem}-figure.pdf")
plt.show()

print("\nRun Complete")
