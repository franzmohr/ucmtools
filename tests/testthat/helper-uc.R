# A short sample and short chains: what is tested here is the plumbing -- shapes,
# dispatch, refusals, seeds -- and the numbers are BayesTS's to test.
uc_data <- function() {
  stats::window(at_gdp, end = c(2019, 4)) * 100
}

uc_fit <- function(..., y = uc_data()[, "y"], seed = 1, iterations = 60, burnin = 40) {
  model <- create_ucmodel(y, ..., iterations = iterations, burnin = burnin)
  model <- add_seed(add_initial_values(add_priors(model)), seed)
  add_posterior_coefficients(model)
}
