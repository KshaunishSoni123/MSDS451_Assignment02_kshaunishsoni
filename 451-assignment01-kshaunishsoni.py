# Portfolio Optimization: A Monte Carlo Study

# Python translation of the R program prepared by Tom Miller, June 20, 2025,
# extended to use real asset data downloaded with yfinance.

# Monte Carlo Study Showing Portfolio Opportunity Sets

# Generate 700 random asset allocation sets: N-by-1 vectors.
# For each set, the portfolio asset weights must sum to 1.
# So, we generate N - 1 and set the last so the sum is 1.
# Weights can be negative (indicating a short position in the asset).
# Many sets of weights will be suboptimal.

# Each set of weights is evaluated against two sources of returns:
#   1. Simulated: multivariate normal returns with the means, standard
#      deviations, and correlations estimated from the historical data.
#   2. Historical: the actual monthly returns of the assets, annualized.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# show all columns when printing summaries
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# portfolio assets: replace with your own tickers
tickers = ["LLY", "JNJ", "ISRG"]
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

# annualized mean returns, standard deviations, and correlation matrix
# (correlation is unaffected by annualizing)
targetMeanVector = historicalReturns.mean().to_numpy() * periodsPerYear
targetSDVector = historicalReturns.std().to_numpy() * np.sqrt(periodsPerYear)
targetCorMatrix = historicalReturns.corr().to_numpy()

print("\nAnnualized statistics from historical data:")
print(pd.DataFrame({"Mean": targetMeanVector, "SD": targetSDVector}, index=tickers))
print("\nCorrelation of historical returns:")
print(historicalReturns.corr())

# compute the covariance matrix
targetCovMatrix = np.diag(targetSDVector) @ targetCorMatrix @ np.diag(targetSDVector)

# number of return sets to generate
sampleSize = 700

# set seed so results are reproducible across executions
np.random.seed(1111)
# generate multivariate normal (annual) returns
returnsData = np.random.multivariate_normal(mean=targetMeanVector,
                                            cov=targetCovMatrix,
                                            size=sampleSize)
returnsDataFrame = pd.DataFrame(returnsData, columns=tickers)

# check statistics from the generated returns data
print("\nSummary of simulated returns:")
print(returnsDataFrame.describe())

# check correlation matrix of the generated returns data
print("\nTarget Correlation of Returns for Generated Data:")
print(targetCorMatrix)
print("\nActual Correlation of Returns in Generated Data:")
print(returnsDataFrame.corr())

# compute covariance matrix for the sample data
# to be used later in portfolio calculations
print("\nTarget Covariance of Returns for Generated Data:")
print(targetCovMatrix)
print("\nActual Covariance of Returns in Generated Data:")
dataCovMatrix = returnsDataFrame.cov().to_numpy()
print(returnsDataFrame.cov())

# covariance matrix of the historical monthly returns
historicalCovMatrix = historicalReturns.cov().to_numpy()


# Generate a random sets of weights using
# a uniform distribution from -1 to 1 if
# shortsOK is True. Otherwise use a uniform
# distribution from 0 to 1 with scaling to
# ensure that the weights sum to 1.
def makeWeights(shortsOK):
    if shortsOK:
        firstWeights = np.random.uniform(low=-1, high=1, size=nAssets - 1)
        lastWeight = 1 - firstWeights.sum()  # ensures sum of 1
        return np.append(firstWeights, lastWeight)
    initialWeights = np.random.uniform(low=0, high=1, size=nAssets)
    return initialWeights / initialWeights.sum()  # ensures sum of 1


def makeWeightsMatrix(shortsOK):
    np.random.seed(9999)  # set seed so results are reproducible
    weightsMatrix = np.full((sampleSize, nAssets), np.nan)
    for iset in range(sampleSize):
        weightsMatrix[iset, :] = makeWeights(shortsOK)
    return weightsMatrix


# Compute the portfolio return for each set of weights
# for each set of weights applied to each set of returns,
# storing calculations in a new data frame for the portfolios.
# Returns are annualized with periodsPerYear (1 for the simulated
# annual returns, 12 for the historical monthly returns).
def computePortfolioResults(weightsMatrix, returns, covMatrix, periods):
    rows = []
    for iset in range(sampleSize):
        weights = weightsMatrix[iset, :]
        Positions = 1  # has shorts is 1
        if (weights > 0).all():
            Positions = 2  # no shorts
        returnVector = weights @ returns.T
        returnMean = returnVector.mean() * periods
        returnSD = float(np.sqrt(weights @ covMatrix @ weights * periods))
        row = {f"w_{ticker}": w for ticker, w in zip(tickers, weights)}
        row.update({"Positions": Positions,
                    "returnMean": returnMean, "returnSD": returnSD})
        rows.append(row)
    return pd.DataFrame(rows)


# the same weight sets are applied to both sources of returns
returnSources = {
    "Simulated": (returnsData, dataCovMatrix, 1),
    "Historical": (historicalReturns.to_numpy(), historicalCovMatrix, periodsPerYear),
}
weightConditions = {
    "Shorts OK": makeWeightsMatrix(shortsOK=True),
    "Long Positions Only": makeWeightsMatrix(shortsOK=False),
}

allResults = []
for sourceName, (returns, covMatrix, periods) in returnSources.items():
    for conditionName, weightsMatrix in weightConditions.items():
        portfolioResults = computePortfolioResults(weightsMatrix, returns,
                                                   covMatrix, periods)

        # check portfolio results
        print(f"\nSummary of portfolio results ({sourceName} returns, "
              f"{conditionName}):")
        print(portfolioResults.describe())

        portfolioResults["Returns"] = sourceName
        portfolioResults["ShortsOK"] = conditionName
        allResults.append(portfolioResults)

# merge the results into one data frame
plottingFrame = pd.concat(allResults, ignore_index=True)

# Portfolio optimization: maximize expected return given the investor's
# tolerance for risk, expressed as a maximum annualized portfolio SD.
# Decision variables are the weights (capital allocation) across the assets.
riskTolerances = [0.17, 0.20, 0.25, 0.30]
riskFreeRate = 0.0  # used for Sharpe ratios
weightColumns = [f"w_{ticker}" for ticker in tickers]
pd.set_option("display.float_format", "{:.4f}".format)


def summarizeRow(weights, returnMean, returnSD):
    row = dict(zip(weightColumns, weights))
    row.update({"returnMean": returnMean, "returnSD": returnSD,
                "Sharpe": (returnMean - riskFreeRate) / returnSD})
    return row


# Approach 1: select the best portfolios from the Monte Carlo results.
# These are approximate: the best of 700 random weight sets.
def selectFromSimulation(results):
    sharpe = (results["returnMean"] - riskFreeRate) / results["returnSD"]
    picks = {"Minimum Variance": results["returnSD"].idxmin(),
             "Maximum Sharpe": sharpe.idxmax()}
    for cap in riskTolerances:
        eligible = results[results["returnSD"] <= cap]
        picks[f"Max Return, SD <= {cap:.0%}"] = (
            eligible["returnMean"].idxmax() if len(eligible) else None)
    rows = {}
    for name, index in picks.items():
        if index is None:  # no simulated portfolio within the risk tolerance
            rows[name] = dict.fromkeys(weightColumns + ["returnMean", "returnSD", "Sharpe"], np.nan)
            continue
        best = results.loc[index]
        rows[name] = summarizeRow(best[weightColumns].to_numpy(dtype=float),
                                  best["returnMean"], best["returnSD"])
    return pd.DataFrame(rows).T


simulationSelections = {}
for (sourceName, conditionName), results in plottingFrame.groupby(["Returns", "ShortsOK"]):
    selection = selectFromSimulation(results.reset_index(drop=True))
    simulationSelections[(sourceName, conditionName)] = selection
    print(f"\nBest portfolios selected from simulation ({sourceName} returns, "
          f"{conditionName}):")
    print(selection)

# combine the selections into one table and export it as CSV and as
# LaTeX (for inclusion in LaTeX document alongside the pdf plot)
simulationSelectionTable = pd.concat(simulationSelections,
                                     names=["Returns", "Positions", "Portfolio"])
simulationSelectionTable.to_csv("451-portfolio-optimization-monte-carlo-selections.csv",
                                float_format="%.4f")
# escape LaTeX special characters (requires booktabs and multirow packages)
latexTable = simulationSelectionTable.rename(
    columns=lambda name: name.replace("_", r"\_"),
    index=lambda name: name.replace("<=", r"$\leq$").replace("%", r"\%"))
latexTable.to_latex("451-portfolio-optimization-monte-carlo-selections.tex",
                                  float_format="%.4f", na_rep="--",
                                  caption="Best portfolios selected from the Monte Carlo simulation",
                                  label="tab:simulation-selections")


# Approach 2: solve the constrained optimization exactly with the
# annualized historical means and covariance matrix:
#   maximize    w'mu
#   subject to  sqrt(w' Sigma w) <= risk tolerance
#               sum(w) = 1
#               w >= 0 (long positions only)
def portfolioMean(weights):
    return weights @ targetMeanVector


def portfolioSD(weights):
    return float(np.sqrt(weights @ targetCovMatrix @ weights))


def solvePortfolio(objective, shortsOK, extraConstraints=()):
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                   *extraConstraints]
    bounds = None if shortsOK else [(0, 1)] * nAssets
    result = minimize(objective, np.full(nAssets, 1 / nAssets),
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else None


def optimizeExactly(shortsOK):
    minVarianceWeights = solvePortfolio(
        lambda w: w @ targetCovMatrix @ w, shortsOK)
    maxSharpeWeights = solvePortfolio(
        lambda w: -(portfolioMean(w) - riskFreeRate) / portfolioSD(w), shortsOK)
    solutions = {"Minimum Variance": minVarianceWeights,
                 "Maximum Sharpe": maxSharpeWeights}
    for cap in riskTolerances:
        name = f"Max Return, SD <= {cap:.0%}"
        if portfolioSD(minVarianceWeights) > cap:
            solutions[name] = None  # risk tolerance is below the minimum attainable SD
            continue
        sdConstraint = {"type": "ineq",
                        "fun": lambda w, cap=cap: cap ** 2 - w @ targetCovMatrix @ w}
        solutions[name] = solvePortfolio(lambda w: -portfolioMean(w), shortsOK,
                                         [sdConstraint])
    rows = {}
    for name, weights in solutions.items():
        if weights is None:
            rows[name] = dict.fromkeys(weightColumns + ["returnMean", "returnSD", "Sharpe"], np.nan)
            continue
        rows[name] = summarizeRow(weights, portfolioMean(weights), portfolioSD(weights))
    return pd.DataFrame(rows).T


# Efficient frontier: the minimum SD for each target return, from the
# minimum variance portfolio up to the top of the plotted return range
def efficientFrontier(shortsOK, maxReturn, nPoints=50):
    minVarianceWeights = solvePortfolio(lambda w: w @ targetCovMatrix @ w, shortsOK)
    targetReturns = np.linspace(portfolioMean(minVarianceWeights), maxReturn, nPoints)
    frontier = []
    for target in targetReturns:
        returnConstraint = {"type": "eq",
                            "fun": lambda w, target=target: portfolioMean(w) - target}
        weights = solvePortfolio(lambda w: w @ targetCovMatrix @ w, shortsOK,
                                 [returnConstraint])
        if weights is not None:
            frontier.append((portfolioSD(weights), target))
    return pd.DataFrame(frontier, columns=["returnSD", "returnMean"])


exactOptima = {}
frontiers = {}
for conditionName in weightConditions:
    shortsOK = conditionName == "Shorts OK"
    exactOptima[conditionName] = optimizeExactly(shortsOK)
    print(f"\nExact optimal portfolios ({conditionName}):")
    print(exactOptima[conditionName])

    if shortsOK:
        maxReturn = plottingFrame.loc[plottingFrame["ShortsOK"] == conditionName,
                                      "returnMean"].max()
    else:
        maxReturn = targetMeanVector.max()  # best long-only return is the best single asset
    frontiers[conditionName] = efficientFrontier(shortsOK, maxReturn)
plottingFrame["Positions"] = plottingFrame["Positions"].map(
    {1: "Has Short(s)", 2: "No Shorts"})

# complete plot: rows are return sources, columns are weight conditions
# (column facets ordered alphabetically, as with an R factor)
colors = {"Has Short(s)": "red", "No Shorts": "darkblue"}
sources = list(returnSources)
facets = sorted(weightConditions)
fig, axes = plt.subplots(len(sources), len(facets), figsize=(11, 8.5),
                         sharex=True, sharey=True)
for irow, source in enumerate(sources):
    for icol, facet in enumerate(facets):
        ax = axes[irow, icol]
        subset = plottingFrame[(plottingFrame["Returns"] == source)
                               & (plottingFrame["ShortsOK"] == facet)]
        for label, color in colors.items():
            points = subset[subset["Positions"] == label]
            ax.scatter(points["returnSD"], points["returnMean"],
                       s=4, color=color, label=label)
        # exact frontier and risk-tolerance optima from the historical estimates
        frontier = frontiers[facet]
        ax.plot(frontier["returnSD"], frontier["returnMean"], color="black",
                linewidth=1.5, label="Exact Efficient Frontier")
        optima = exactOptima[facet].filter(like="Max Return", axis=0).dropna()
        ax.scatter(optima["returnSD"], optima["returnMean"], marker="*", s=120,
                   color="gold", edgecolor="black", zorder=3,
                   label="Exact Optimum for Risk Tolerance")
        ax.set_title(f"{source} Returns: {facet}")
for ax in axes[-1, :]:
    ax.set_xlabel("Risk: Standard Deviation of Portfolio Returns", fontsize=12)
for ax in axes[:, 0]:
    ax.set_ylabel("Return: Mean of Portfolio Returns", fontsize=12)
fig.suptitle(f"Annualized Portfolio Opportunity Sets: {', '.join(tickers)}")
fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="lower center", ncol=4)
fig.tight_layout(rect=(0, 0.05, 1, 1))

# export pdf plot for inclusion in LaTeX document
fig.savefig("451-portfolio-optimization-monte-carlo-figure.pdf")
plt.show()

print("\nRun Complete")
