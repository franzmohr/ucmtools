# ucmtools 0.0.0.9000

* **A damped trend and signal equations: the European Commission's output gap
  models.** `create_ucmodel(trend = "damped")` makes the trend's slope a
  stationary AR(1), one `rho` per series under a normal prior truncated to
  (-1, 1), `add_priors(rho = )`, drawn as `posterior$rho`. Argument `signal`
  adds further observed series that load on the cycle at lags 0 to
  `signal_lags`, with a constant (`signal_const`) and regressors of their own
  (`signal_x`): the capacity utilisation indicator on the cycle of total factor
  productivity, or a Phillips curve on the unemployment gap (Havik et al.
  2014). Their priors are `add_priors(signal = , signal_noise = )`, their draws
  `posterior$signal`, named `<signal>.<regressor>`, and
  `posterior$signal_noise_inv`. The log likelihood is the density of the series
  and the signals together; a forecast and its score are of the series alone.
  Both are available for `error = "gamma"`, `"gamma+covar"` and `"wishart"`,
  and signals not with `cycle_init = "stationary"`. Both are written to and
  read from the model file under the names BayesTS gives them. Draws of a model
  without them are unchanged: fixed-seed coefficients, log likelihoods and
  forecasts of every error specification, both cycle starts, are identical to
  those of a clean build of the previous version.

* The vendored BayesTS core moves to `02a41de`, which adds a damped trend and
  signal equations to UcNormalGamma and UcNormalWishart: the European
  Commission's output gap models, nested. Neither is reachable from R yet, so
  nothing changes for a model built here. Draws are unchanged: fixed-seed
  coefficients, log likelihoods and forecasts of every error specification,
  with the cycle starting at zero and from its stationary distribution, are
  identical to those of a clean build of the previous version.

* **`create_ucmodel(cycle_init = "stationary")` starts the cycle from its
  stationary distribution rather than from zero.** With the default, `"zero"`,
  the cycle is zero before the sample, as in Grant and Chan (2017), which pins
  the first estimates of the cycle near zero whatever the data say. The new
  value draws the `p` values before the sample from the cycle's stationary
  distribution under each draw's coefficients and shock covariance, returns
  them as `posterior$cycle0`, and integrates them out of the log likelihood. It
  is available for `error = "gamma"`, `"gamma+covar"` and `"wishart"`, and
  needs a stationary cycle with lags. The vendored BayesTS core moves to
  `4808dff`, which adds it; draws of a model without it are unchanged.

* `stationary` now defaults to `TRUE` for every error specification with a
  cycle with lags, and is accepted by all of them: the four samplers all draw a
  stationary cycle when asked to. Before it was limited to the gamma errors.
  Draws of the Wishart, stochastic volatility and skewed models change, since
  their cycle is now drawn stationary by default.

* **First version: the four unobserved components models of BayesTS in R.**
  `create_ucmodel()` sets up a trend-cycle decomposition of one or more series
  -- a random walk or smooth trend with a drift and breaks in it, a stationary
  autoregressive cycle, an optional irregular component -- and its `error`
  argument picks the sampler: `"gamma"` and `"gamma+covar"` (UcNormalGamma),
  `"wishart"` (UcNormalWishart), `"sv"` and `"sv+covar"` (UcNormalStochvol) and
  `"skew"` (UcNormalSkew). The samplers are vendored from BayesTS `75d35c6`,
  44 files reached from the four entry points.

  The methods of bayests' generics give the shared workflow:
  `add_priors()`, `add_initial_values()`, `add_seed()`,
  `add_posterior_coefficients()`, `add_forecast_input()`,
  `add_posterior_forecasts()`, `add_posterior_loglik()`,
  `add_predictive_loglik()`, `write_to_hdf5()` and reading a model file back,
  so that lists of models, expanding windows and `selection_criteria()` work as
  they do for VARs and factor models. `trend()` and `cycle()` return the
  posterior of the components, `plot()` draws them, and `predict()` returns the
  forecasts in the form bayests plots.

  The posterior is drawn with the model's seed, and the forecast and the score
  with seeds derived from it, by the rule bvartools and dfmtools follow. The
  log likelihood integrates the trend out, as Grant and Chan (2017) compare
  these models by. The cycle is drawn stationary by default for the gamma
  shocks. Scoring a forecast is refused for skewed shocks, whose predictive
  density BayesTS does not write.

* Data set `at_gdp`: Austrian and foreign log real GDP, 1979Q2 to 2023Q3, from
  bvartools' `at_macrodata`.
