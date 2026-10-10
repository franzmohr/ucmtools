# ucmtools

Bayesian inference of unobserved components models in R: trend-cycle
decompositions of one or more time series, with a random walk or smooth trend,
a drift and breaks in it, a stationary autoregressive cycle, an optional
irregular component, and shocks that are independent, correlated, stochastically
volatile or skewed.

The posterior is simulated by the Gibbs samplers of
[BayesTS](https://github.com/franzmohr/BayesTS), vendored into the package, and
the workflow -- priors, initial values, posterior draws, forecasts, model
comparison, model files -- is the one of [bayests](https://github.com/franzmohr/bayests),
shared with bvartools and dfmtools.

| `error`        | Sampler          | Shocks                                       |
|----------------|------------------|----------------------------------------------|
| `"gamma"`      | UcNormalGamma    | independent, gamma priors on the precisions  |
| `"gamma+covar"`| UcNormalGamma    | correlated through a covariance block        |
| `"wishart"`    | UcNormalWishart  | unrestricted covariance, Wishart prior       |
| `"sv"`, `"sv+covar"` | UcNormalStochvol | stochastic volatility                  |
| `"skew"`       | UcNormalSkew     | skew normal or asymmetric Laplace, scales moving with an index |

## Example

```r
library(ucmtools)

data("at_gdp")
y <- stats::window(at_gdp[, "y"], end = c(2019, 4)) * 100

model <- create_ucmodel(y, p = 2, breaks = list(c(2008, 3)))
model <- add_priors(model, a = list(mu = c(1.3, -0.7), v_inv = 1))
model <- add_initial_values(model)
model <- add_posterior_coefficients(model)

plot(model)            # trend over the data, cycle with its band
gap <- cycle(model)    # posterior bands of the cycle

model <- add_posterior_loglik(model)
bayests::selection_criteria(model)

model <- add_forecast_input(model, n_ahead = 8)
model <- add_posterior_forecasts(model)
plot(predict(model))
```
