#Question 1
# Set parameters
N <- 200  # New sample size
true_betas <- c(0.1, 1.1, -0.9)  # (β0, β1, β2)

# Step 1: Generate covariates from Uniform[-2, 2]
set.seed(40)  # Set seed for reproducibility
X_covariates <- matrix(runif(2 * N, min = -2, max = 2), nrow = N, ncol = 2)

# Add a column of ones for the intercept term
X <- cbind(1, X_covariates)

# Step 2: Compute probabilities using logistic model
logits <- X %*% true_betas  # Matrix multiplication (includes intercept term)
probabilities <- 1 / (1 + exp(-logits))  # Apply sigmoid to logits

# Step 3: Generate binary responses (y) from Bernoulli distribution
y <- rbinom(N, size = 1, prob = probabilities)

# Report statistics: how many 0's and 1's
zeros <- sum(y == 0)
ones <- sum(y == 1)

cat("Statistics for N =", N, ":\n")
cat("  Number of zeros:", zeros, "\n")
cat("  Number of ones:", ones, "\n")
cat("  Proportion of zeros:", round(zeros / N, 2), "\n")
cat("  Proportion of ones:", round(ones / N, 2), "\n")




#Question 2
# Part 1: Data Generation
N <- 200
true_betas <- c(0.1, 1.1, -0.9)
set.seed(40)
X_covariates <- matrix(runif(2 * N, -2, 2), nrow = N, ncol = 2)
X <- cbind(1, X_covariates)
probabilities <- 1 / (1 + exp(-X %*% true_betas))
y <- rbinom(N, 1, probabilities)

# Report data statistics
zeros <- sum(y == 0)
ones <- sum(y == 1)
cat("Data Statistics:\n",
    "Zeros:", zeros, "Proportion:", round(zeros/N, 2), "\n",
    "Ones:", ones, "Proportion:", round(ones/N, 2), "\n")

# Part 2: MCMC Setup and Sampling
prior_mean <- c(0, 0, 0)
prior_sd <- 2
n_iter <- 10000
n_chains <- 20
n_params <- 3

# Log posterior function
log_posterior <- function(beta, X, y, prior_mean, prior_sd) {
  linear_pred <- X %*% beta
  log_lik <- sum(y * linear_pred - log(1 + exp(linear_pred)))
  log_prior <- sum(dnorm(beta, prior_mean, prior_sd, log = TRUE))
  return(log_lik + log_prior)
}

# Initialize and run chains
chains <- array(NA, dim = c(n_iter, n_params, n_chains))
acceptance_counts <- numeric(n_chains)

for(chain in 1:n_chains) {
  beta_current <- rnorm(n_params, prior_mean, prior_sd)
  proposal_sd <- 0.3  # Adjusted from 0.2 to 0.3
  chain_samples <- matrix(NA, n_iter, n_params)
  
  for(i in 1:n_iter) {
    beta_proposal <- beta_current + rnorm(n_params, 0, proposal_sd)
    log_r <- log_posterior(beta_proposal, X, y, prior_mean, prior_sd) - 
      log_posterior(beta_current, X, y, prior_mean, prior_sd)
    
    if(log(runif(1)) < log_r) {
      beta_current <- beta_proposal
      acceptance_counts[chain] <- acceptance_counts[chain] + 1
    }
    chain_samples[i,] <- beta_current
  }
  chains[,,chain] <- chain_samples
}

# Calculate acceptance rates
acceptance_rates <- acceptance_counts / n_iter
cat("Mean acceptance rate:", mean(acceptance_rates), "\n")

# Part 3: Visualization
windows(width = 12, height = 10)
par(mfrow = c(3,1), mar = c(4,4,2,1))

# Trace plots
for(i in 1:3) {
  plot(chains[,i,1], type = "l", 
       main = paste("Trace Plot - Parameter", i), 
       ylab = "Value", xlab = "Iteration")
  abline(h = true_betas[i], col = "red")
}

# Histograms after burn-in
burn_in <- 1000
chains_burned <- chains[(burn_in+1):n_iter,,]

windows(width = 12, height = 10)
par(mfrow = c(3,1), mar = c(4,4,2,1))
for(i in 1:3) {
  hist(chains_burned[,i,1], breaks = 30, 
       main = paste("Parameter", i, "Posterior Distribution"), 
       xlab = "Value")
  abline(v = true_betas[i], col = "red", lwd = 2)
  abline(v = mean(chains_burned[,i,1]), col = "blue", lwd = 2)
  legend("topright", 
         legend = c("True Value", "Posterior Mean"), 
         col = c("red", "blue"), 
         lwd = 2)
}

# Part 4: Gelman-Rubin Convergence Diagnostics
library(coda)
assessment_points <- seq(100, n_iter, by = 100)
rhat_values <- matrix(NA, length(assessment_points), n_params)

for(i in 1:length(assessment_points)) {
  chains_subset <- chains[1:assessment_points[i],,]
  mcmc_list <- lapply(1:n_chains, function(j) mcmc(chains_subset[,,j]))
  rhat_values[i,] <- gelman.diag(mcmc_list)$psrf[,1]
}

windows(width = 12, height = 10)
par(mfrow = c(3,1), mar = c(4,4,2,1))
for(i in 1:3) {
  plot(assessment_points, rhat_values[,i], type = "l",
       main = paste("R-hat for Parameter", i),
       xlab = "Iteration", ylab = "R-hat")
  abline(h = 1.1, col = "red", lty = 2)
}

# Calculate and display posterior summaries
posterior_means <- apply(chains_burned, 2, mean)
posterior_sds <- apply(chains_burned, 2, sd)
cat("\nPosterior Summaries:\n")
for(i in 1:3) {
  cat(sprintf("Parameter %d: Mean = %.3f, SD = %.3f\n", 
              i, posterior_means[i], posterior_sds[i]))
}




#question 3 
# Part 1: Data Generation with 9 dimensions
N <- 200

# Set seed for reproducibility
set.seed(40)

# Generate true parameters
initial_betas <- c(0.1, 1.1, -0.9)
additional_betas <- runif(6, -2, 2)
true_betas <- c(initial_betas, additional_betas)

# Print the parameters being used
cat("True parameter values:\n")
print(true_betas)

# Generate covariates and response
X_covariates <- matrix(runif(8 * N, -2, 2), nrow = N, ncol = 8)
X <- cbind(1, X_covariates)
probabilities <- 1 / (1 + exp(-X %*% true_betas))
y <- rbinom(N, 1, probabilities)

# Report data statistics
zeros <- sum(y == 0)
ones <- sum(y == 1)
cat("\nData Statistics:\n",
    "Zeros:", zeros, "Proportion:", round(zeros/N, 2), "\n",
    "Ones:", ones, "Proportion:", round(ones/N, 2), "\n")

# Part 2: MCMC Setup and Sampling
prior_mean <- rep(0, 9)
prior_sd <- 2
n_iter <- 10000
n_chains <- 20
n_params <- 9

# Log posterior function
log_posterior <- function(beta, X, y, prior_mean, prior_sd) {
  linear_pred <- X %*% beta
  log_lik <- sum(y * linear_pred - log(1 + exp(linear_pred)))
  log_prior <- sum(dnorm(beta, prior_mean, prior_sd, log = TRUE))
  return(log_lik + log_prior)
}

# Initialize and run chains
chains <- array(NA, dim = c(n_iter, n_params, n_chains))
acceptance_counts <- numeric(n_chains)

for(chain in 1:n_chains) {
  beta_current <- rnorm(n_params, prior_mean, prior_sd)
  proposal_sd <- 0.2  # Optimized value for target 23.4% acceptance rate
  chain_samples <- matrix(NA, n_iter, n_params)
  
  for(i in 1:n_iter) {
    beta_proposal <- beta_current + rnorm(n_params, 0, proposal_sd)
    log_r <- log_posterior(beta_proposal, X, y, prior_mean, prior_sd) - 
      log_posterior(beta_current, X, y, prior_mean, prior_sd)
    
    if(log(runif(1)) < log_r) {
      beta_current <- beta_proposal
      acceptance_counts[chain] <- acceptance_counts[chain] + 1
    }
    chain_samples[i,] <- beta_current
  }
  chains[,,chain] <- chain_samples
}

# Calculate acceptance rates
acceptance_rates <- acceptance_counts / n_iter
cat("\nMean acceptance rate:", mean(acceptance_rates), "\n")

# Part 3: Visualization
windows(width = 12, height = 10)
par(mfrow = c(3,3), mar = c(4,4,2,1))

# Trace plots
for(i in 1:9) {
  plot(chains[,i,1], type = "l", 
       main = paste("Trace Plot - Parameter", i), 
       ylab = "Value", xlab = "Iteration")
  abline(h = true_betas[i], col = "red")
}

# Histograms after burn-in
burn_in <- 1000
chains_burned <- chains[(burn_in+1):n_iter,,]

windows(width = 12, height = 10)
par(mfrow = c(3,3), mar = c(4,4,2,1))
for(i in 1:9) {
  hist(chains_burned[,i,1], breaks = 30, 
       main = paste("Parameter", i, "Posterior Distribution"), 
       xlab = "Value")
  abline(v = true_betas[i], col = "red", lwd = 2)
  abline(v = mean(chains_burned[,i,1]), col = "blue", lwd = 2)
  legend("topright", 
         legend = c("True Value", "Posterior Mean"), 
         col = c("red", "blue"), 
         lwd = 2)
}

# Part 4: Gelman-Rubin Convergence Diagnostics
library(coda)
assessment_points <- seq(100, n_iter, by = 100)
rhat_values <- matrix(NA, length(assessment_points), n_params)

for(i in 1:length(assessment_points)) {
  chains_subset <- chains[1:assessment_points[i],,]
  mcmc_list <- lapply(1:n_chains, function(j) mcmc(chains_subset[,,j]))
  rhat_values[i,] <- gelman.diag(mcmc_list)$psrf[,1]
}

windows(width = 12, height = 10)
par(mfrow = c(3,3), mar = c(4,4,2,1))
for(i in 1:9) {
  plot(assessment_points, rhat_values[,i], type = "l",
       main = paste("R-hat for Parameter", i),
       xlab = "Iteration", ylab = "R-hat")
  abline(h = 1.1, col = "red", lty = 2)
}

# Calculate and display posterior summaries
posterior_means <- apply(chains_burned, 2, mean)
posterior_sds <- apply(chains_burned, 2, sd)
cat("\nPosterior Summaries:\n")
for(i in 1:9) {
  cat(sprintf("Parameter %d: Mean = %.3f, SD = %.3f\n", 
              i, posterior_means[i], posterior_sds[i]))
}






















#QUESTION 4 
# Part 1: Data Generation with 9 dimensions
N <- 200

# Set seed for reproducibility
set.seed(40)

# Generate true parameters
initial_betas <- c(0.1, 1.1, -0.9)
additional_betas <- runif(6, -2, 2)
true_betas <- c(initial_betas, additional_betas)

# Print the parameters being used
cat("True parameter values:\n")
print(true_betas)

# Generate covariates and response
X_covariates <- matrix(runif(8 * N, -2, 2), nrow = N, ncol = 8)
X <- cbind(1, X_covariates)
probabilities <- 1 / (1 + exp(-X %*% true_betas))
y <- rbinom(N, 1, probabilities)

# Report data statistics
zeros <- sum(y == 0)
ones <- sum(y == 1)
cat("\nData Statistics:\n",
    "Zeros:", zeros, "Proportion:", round(zeros/N, 2), "\n",
    "Ones:", ones, "Proportion:", round(ones/N, 2), "\n")

# Metropolis-within-Gibbs implementation
n_iter <- 10000
n_chains <- 10
n_params <- 9
prior_mean <- rep(0, 9)
prior_sd <- 2

# Function for single parameter conditional
log_conditional <- function(beta_prop, beta_current, param_idx, X, y, prior_mean, prior_sd) {
  beta_temp <- beta_current
  beta_temp[param_idx] <- beta_prop
  linear_pred <- X %*% beta_temp
  log_lik <- sum(y * linear_pred - log(1 + exp(linear_pred)))
  log_prior <- dnorm(beta_prop, prior_mean[param_idx], prior_sd, log = TRUE)
  return(log_lik + log_prior)
}

# Initialize storage
chains <- array(NA, dim = c(n_iter, n_params, n_chains))
acceptance_counts <- matrix(0, n_chains, n_params)

# Individual proposal scales for each parameter
proposal_sds <- rep(1.8, n_params)  # Tuned for ~15% acceptance rate

# Run chains
for(chain in 1:n_chains) {
  beta_current <- rnorm(n_params, prior_mean, prior_sd)
  
  for(i in 1:n_iter) {
    # Update each parameter conditionally
    for(j in 1:n_params) {
      beta_proposal <- beta_current[j] + rnorm(1, 0, proposal_sds[j])
      
      log_r <- log_conditional(beta_proposal, beta_current, j, X, y, prior_mean, prior_sd) - 
        log_conditional(beta_current[j], beta_current, j, X, y, prior_mean, prior_sd)
      
      if(log(runif(1)) < log_r) {
        beta_current[j] <- beta_proposal
        acceptance_counts[chain, j] <- acceptance_counts[chain, j] + 1
      }
    }
    chains[i,,chain] <- beta_current
  }
}

# Calculate acceptance rates per parameter
acceptance_rates <- acceptance_counts / n_iter
mean_rates <- colMeans(acceptance_rates)
cat("\nMean acceptance rates per parameter:\n")
print(mean_rates)

# Trace plots
windows(width = 12, height = 10)
par(mfrow = c(3,3), mar = c(4,4,2,1))
for(i in 1:9) {
  plot(chains[,i,1], type = "l", 
       main = paste("Trace Plot - Parameter", i), 
       ylab = "Value", xlab = "Iteration")
  abline(h = true_betas[i], col = "red")
}

# Gelman-Rubin diagnostics
library(coda)
assessment_points <- seq(100, n_iter, by = 100)
rhat_values <- matrix(NA, length(assessment_points), n_params)

for(i in 1:length(assessment_points)) {
  chains_subset <- chains[1:assessment_points[i],,]
  mcmc_list <- lapply(1:n_chains, function(j) mcmc(chains_subset[,,j]))
  rhat_values[i,] <- gelman.diag(mcmc_list)$psrf[,1]
}

# Enhanced Gelman-Rubin plots
windows(width = 12, height = 10)
par(mfrow = c(3,3), mar = c(4,4,2,1))
for(i in 1:9) {
  plot(assessment_points, rhat_values[,i], type = "l",
       main = paste("R-hat for Parameter", i),
       xlab = "Iteration", ylab = "R-hat",
       ylim = c(1, max(rhat_values)))
  abline(h = 1.1, col = "red", lty = 2, lwd = 2)
}

# Calculate and display posterior summaries
burn_in <- 1000
chains_burned <- chains[(burn_in+1):n_iter,,]
posterior_means <- apply(chains_burned, 2, mean)
posterior_sds <- apply(chains_burned, 2, sd)

cat("\nPosterior Summaries:\n")
for(i in 1:9) {
  cat(sprintf("Parameter %d: Mean = %.3f, SD = %.3f\n", 
              i, posterior_means[i], posterior_sds[i]))
}













#Question 5
# Part 1: Data Generation with 9 dimensions
N <- 200

# Set seed for reproducibility
set.seed(40)

# Generate true parameters
initial_betas <- c(0.1, 1.1, -0.9)
additional_betas <- runif(6, -2, 2)
true_betas <- c(initial_betas, additional_betas)

# Print the parameters being used
cat("True parameter values:\n")
print(true_betas)

# Generate covariates and response
X_covariates <- matrix(runif(8 * N, -2, 2), nrow = N, ncol = 8)
X <- cbind(1, X_covariates)
probabilities <- 1 / (1 + exp(-X %*% true_betas))
y <- rbinom(N, 1, probabilities)

# Report data statistics
zeros <- sum(y == 0)
ones <- sum(y == 1)
cat("\nData Statistics:\n",
    "Zeros:", zeros, "Proportion:", round(zeros/N, 2), "\n",
    "Ones:", ones, "Proportion:", round(ones/N, 2), "\n")

# MCMC Setup
n_iter <- 10000
n_chains <- 20
n_params <- 9
prior_mean <- rep(0, 9)
prior_sd <- 2

# Function for single parameter conditional
log_conditional <- function(beta_prop, beta_current, param_idx, X, y, prior_mean, prior_sd) {
  beta_temp <- beta_current
  beta_temp[param_idx] <- beta_prop
  linear_pred <- X %*% beta_temp
  log_lik <- sum(y * linear_pred - log(1 + exp(linear_pred)))
  log_prior <- dnorm(beta_prop, prior_mean[param_idx], prior_sd, log = TRUE)
  return(log_lik + log_prior)
}

# Two different proposal scales
proposal_sd_low <- 2.5  # For ~10% acceptance rate
proposal_sd_high <- 1.1  # For ~30% acceptance rate

# Initialize storage
chains <- array(NA, dim = c(n_iter, n_params, n_chains))
acceptance_counts_low <- matrix(0, n_chains, n_params)
acceptance_counts_high <- matrix(0, n_chains, n_params)
proposal_counts_low <- matrix(0, n_chains, n_params)
proposal_counts_high <- matrix(0, n_chains, n_params)

# Run MCMC
for(chain in 1:n_chains) {
  beta_current <- rnorm(n_params, prior_mean, prior_sd)
  
  for(i in 1:n_iter) {
    for(j in 1:n_params) {
      # Track which proposal is used
      use_low_proposal <- runif(1) < 0.5
      proposal_sd <- ifelse(use_low_proposal, proposal_sd_low, proposal_sd_high)
      
      if(use_low_proposal) {
        proposal_counts_low[chain, j] <- proposal_counts_low[chain, j] + 1
      } else {
        proposal_counts_high[chain, j] <- proposal_counts_high[chain, j] + 1
      }
      
      beta_proposal <- beta_current[j] + rnorm(1, 0, proposal_sd)
      
      log_r <- log_conditional(beta_proposal, beta_current, j, X, y, prior_mean, prior_sd) - 
        log_conditional(beta_current[j], beta_current, j, X, y, prior_mean, prior_sd)
      
      if(log(runif(1)) < log_r) {
        beta_current[j] <- beta_proposal
        if(use_low_proposal) {
          acceptance_counts_low[chain, j] <- acceptance_counts_low[chain, j] + 1
        } else {
          acceptance_counts_high[chain, j] <- acceptance_counts_high[chain, j] + 1
        }
      }
    }
    chains[i,,chain] <- beta_current
  }
}

# Print acceptance rates for both proposal types
cat("\nAcceptance rates for low proposal (target 10%):\n")
low_rates <- colMeans(acceptance_counts_low/proposal_counts_low)
print(round(low_rates, 3))

cat("\nAcceptance rates for high proposal (target 30%):\n")
high_rates <- colMeans(acceptance_counts_high/proposal_counts_high)
print(round(high_rates, 3))

# Calculate and print mixing statistics
cat("\nProposal mix statistics:\n")
low_prop <- colMeans(proposal_counts_low/(proposal_counts_low + proposal_counts_high))
cat("Proportion of low proposals:", round(mean(low_prop), 3), "\n")

# Gelman-Rubin diagnostics
library(coda)
assessment_points <- c(seq(10, 1000, by = 10), seq(1100, n_iter, by = 100))
rhat_values <- matrix(NA, length(assessment_points), n_params)

for(i in 1:length(assessment_points)) {
  chains_subset <- chains[1:assessment_points[i],,]
  mcmc_list <- lapply(1:n_chains, function(j) mcmc(chains_subset[,,j]))
  rhat_values[i,] <- gelman.diag(mcmc_list)$psrf[,1]
}

# Print convergence speed metrics
cat("\nConvergence speed metrics:\n")
for(i in 1:n_params) {
  first_converged <- which(rhat_values[,i] < 1.1)[1]
  if(!is.na(first_converged)) {
    cat(sprintf("Parameter %d first reached R-hat < 1.1 at iteration %d\n", 
                i, assessment_points[first_converged]))
  }
}

# Plot Gelman-Rubin statistics
windows(width = 12, height = 10)
par(mfrow = c(3,3), mar = c(4,4,2,1))
for(i in 1:9) {
  plot(assessment_points, rhat_values[,i], type = "l",
       main = paste("R-hat for Parameter", i),
       xlab = "Iteration", ylab = "R-hat",
       ylim = c(1, max(rhat_values)))
  abline(h = 1.1, col = "red", lty = 2, lwd = 2)
}

# Print final R-hat values
cat("\nFinal R-hat values:\n")
print(round(rhat_values[nrow(rhat_values),], 3))

# Calculate and display posterior summaries
burn_in <- 1000
chains_burned <- chains[(burn_in+1):n_iter,,]
posterior_means <- apply(chains_burned, 2, mean)
posterior_sds <- apply(chains_burned, 2, sd)

cat("\nPosterior Summaries:\n")
for(i in 1:9) {
  cat(sprintf("Parameter %d: Mean = %.3f, SD = %.3f\n", 
              i, posterior_means[i], posterior_sds[i]))
}

