# ucmtools 0.0.0.9000

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
