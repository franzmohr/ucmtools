#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib ucmtools, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom bayests add_priors add_initial_values add_posterior_coefficients
#' @importFrom bayests add_forecast_input prepare_forecast_input add_posterior_forecasts
#' @importFrom bayests add_posterior_loglik add_predictive_loglik add_seed
#' @importFrom bayests write_to_hdf5 from_bayests_tree
#' @importFrom bayests .check_seed .with_model_seed .with_stage_seed
#' @importFrom stats predict
## usethis namespace: end
NULL

# Unload the DLL when the package is unloaded
.onUnload <- function(libpath) {
  library.dynam.unload("ucmtools", libpath)
}
