# Portfolio Optimization: A Monte Carlo Study

# Prepared by Tom Miller, June 20, 2025 

# Monte Carlo Study Showing Portfolio Opportunity Sets

# Generate above 700 random asset allocation sets: 4-by-1 vectors. 
# For each set, the four portfolio asset weights must sum to 1.
# So, we generate three and set the fourth so the sum is 1. 
# Weights can be negative (indicating a short position in the asset).
# Many sets of weights will be suboptimal.

# Assume multivariate normal distributions of returns
# with means, standard deviations, and correlations
# as specified in the problem setup.

# functions from 'Modern Applied Statistics in S' (Venables and Ripley 2002)
library(MASS) 
library(ggplot2) # visualization of the simulation

# mean returns for portfolio assets A, B, C, and D                                       
targetMeanVector <- c(0.02, 0.07, 0.15, 0.20) 

# standard deviations of returns
targetSDVector <- c(0.05, 0.12, 0.17, 0.25)

# correlation matrix for the four portfolio assets
targetCorMatrix <- matrix(c(1, 0.3, 0.3, 0.3, 
	                  0.3, 1, 0.6, 0.6, 
	                  0.3, 0.6, 1, 0.6,
	                  0.3, 0.6, 0.6, 1),
                     nrow = 4, ncol = 4) 

# compute the covariance matrix
targetCovMatrix <- diag(targetSDVector) %*% targetCorMatrix %*% diag(targetSDVector)
 
# number of return sets to generate
sampleSize <- 700 

# set seed so results are reproducible across executions 
set.seed(1111) 
# generate multivariate normal returns
returnsData <- mvrnorm(n = sampleSize, 
                       mu = targetMeanVector,  
                       Sigma = targetCovMatrix) 
returnsDataFrame <- as.data.frame(returnsData)
names(returnsDataFrame) <- c("A", "B", "C", "D")

# check statistics from the generated returns data
print(summary(returnsDataFrame))

# check correlation matrix of the generated returns data
cat("\nTarget Correlation of Returns for Generated Data:\n")
print(targetCorMatrix)
cat("\nActual Correlation of Returns in Generated Data:\n")
print(cor(returnsDataFrame))

# compute covariance matrix for the sample data
# to be used later in portfolio calculations 
cat("\nTarget Covariance of Returns for Generated Data:\n")
print(targetCovMatrix)
cat("\nActual Covariance of Returns in Generated Data:\n")
dataCovMatrix <- cov(returnsDataFrame) 
print(dataCovMatrix)

# Generate a random sets of weights using
# a uniform distribution from -1 to 1 if
# shortsOK is TRUE. Otherwise use a uniform
# distribution from 0 to 1 with scaling to
# ensure that the weights sum to 1.
makeWeights <- function(shortsOK) {
	if (shortsOK) {
		threeWeights <- runif(3, min = -1, max = 1)
		fourthWeight <- 1 - sum(threeWeights) # ensures sum of 1
		return(c(threeWeights,fourthWeight))
	}
	if (!shortsOK) {
		initialWeights <- runif(4, min = 0, max = 1)
		return(initialWeights/sum(initialWeights)) # ensures sum of 1
	}
} 

# generate sets of portfolio weights in shortsOK state
set.seed(9999) # set seed so results are reproducible 
weightsMatrix <- matrix(NA, nrow = sampleSize, ncol = 4)
for (iset in 1:sampleSize) 
    weightsMatrix[iset,] <- makeWeights(shortsOK = TRUE)

# Compute the portfolio return for each set of weights
# for each set of weights applied to each set of returns, 
# storing calculations in a new data frame for the portfolios.
portfolioResults = NULL
for (iset in 1:sampleSize) {
	Positions <- 1 # has shorts is 1
	w1 <- weightsMatrix[iset,1]
	w2 <- weightsMatrix[iset,2]
	w3 <- weightsMatrix[iset,3]
	w4 <- weightsMatrix[iset,4]
	if (w1 > 0 && w2 > 0 && w3 > 0 && w4 > 0) Positions <- 2 # no shorts

    returnVector <- weightsMatrix[iset,] %*% t(returnsData) 
    returnMean <- mean(returnVector)
    returnSD <- as.numeric(sqrt(t(weightsMatrix[iset,]) %*% dataCovMatrix %*% weightsMatrix[iset,]))
    thisPortfolioResult <- data.frame(w1, w2, w3, w4, Positions, returnMean, returnSD)
    portfolioResults <- rbind(portfolioResults, thisPortfolioResult)
}

# check portfolio results
cat("\nSummary of portfolio results with short positions allowed:\n")
print(summary(portfolioResults))

with(portfolioResults, plot(returnSD, returnMean))

shortsOKResults <- portfolioResults
shortsOKResults$ShortsOK = rep("Shorts OK", times = sampleSize)

# generate sets of portfolio weights with no shorts allowed
set.seed(9999) # set seed so results are reproducible 
weightsMatrix <- matrix(NA, nrow = sampleSize, ncol = 4)
for (iset in 1:sampleSize) 
    weightsMatrix[iset,] <- makeWeights(shortsOK = FALSE)

# Compute the portfolio return for each set of weights
# for each set of weights applied to each set of returns, 
# storing calculations in a new data frame for the portfolios.
portfolioResults = NULL
for (iset in 1:sampleSize) {
	Positions <- 1 # has shorts is 1
	w1 <- weightsMatrix[iset,1]
	w2 <- weightsMatrix[iset,2]
	w3 <- weightsMatrix[iset,3]
	w4 <- weightsMatrix[iset,4]
	if (w1 > 0 && w2 > 0 && w3 > 0 && w4 > 0) Positions <- 2 # no shorts
    returnVector <- weightsMatrix[iset,] %*% t(returnsData) 
    returnMean <- mean(returnVector)
    returnSD <- as.numeric(sqrt(t(weightsMatrix[iset,]) %*% dataCovMatrix %*% weightsMatrix[iset,]))
    thisPortfolioResult <- data.frame(w1, w2, w3, w4, Positions, returnMean, returnSD)
    portfolioResults <- rbind(portfolioResults, thisPortfolioResult)
}

# check portfolio results
cat("\nSummary of portfolio results with long positions only:\n")
print(summary(portfolioResults))

with(portfolioResults, plot(returnSD, returnMean)) # preliminary plot on console

noShortsResults <- portfolioResults
noShortsResults$ShortsOK = rep("Long Positions Only", times = sampleSize)

# merge the two data frames
plottingFrame = rbind(shortsOKResults,noShortsResults)
plottingFrame$Positions <- factor(plottingFrame$Positions, 
	labels = c("Has Short(s)", "No Shorts"))

plottingFrame$ShortsOK = factor(plottingFrame$ShortsOK)

# complete plot for both conditions: Shorts Allowed and No Shorts
facetPlot <- ggplot(plottingFrame, aes(x=returnSD,y=returnMean, 
	colour=Positions)) +
    geom_point(size = 1) +
    xlab("Risk: Standard Deviation of Portfolio Returns") +
    ylab("Return: Mean of Portfolio Returns") +
    scale_color_manual(values = c("red","darkblue")) +
    theme(axis.title = element_text(size=15)) +
    facet_wrap( ~ ShortsOK, ncol = 2) 
    
print(facetPlot)

# export pdf plot for inclusion in LaTeX document
pdf(file = "451-portfolio-optimization-monte-carlo-figure.pdf", width = 11, height = 8.5)
print(facetPlot)
dev.off()

cat("\nRun Complete\n")
