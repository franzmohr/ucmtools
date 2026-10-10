#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "bayests_r_io.h"
#include "bayests_reporter.h"
#include "uc_r_translation.h"
#include "bayests/uc_normal_skew.h"

// Unobserved components model whose cycle or trend shocks are skewed, with
// scales that move with an index, after Huang, Luo and Wang (2026). The index
// is data$skew_x over the sample and data$forecast$skew_x over the horizon.
// The numerics are the vendored BayesTS core; what is here is the translation
// between the R model object and the core's structs. See src/core/VENDORED.md.

namespace {

using namespace bayests_r;
using namespace ucmtools;

bayests::UcNormalSkewInput read_input(const Rcpp::List &object) {

  bayests::UcNormalSkewInput input;
  read_uc_common(object, input);

  if (has(object, "data")) {
    const Rcpp::List data = object["data"];
    read_mat_if_present(data, "skew_x", input.train_skew_x);
    if (has(data, "forecast")) {
      read_mat_if_present(Rcpp::List(data["forecast"]), "skew_x", input.forecast_skew_x);
    }
  }
  read_uc_normal_block(object, "skew", input.skew_prior, input.initial.skew);
  if (has(object, "priors")) {
    const Rcpp::List priors = object["priors"];
    if (has(priors, "u_sigma")) {
      input.u_sigma_prior = read_gamma_prior(Rcpp::List(priors["u_sigma"]));
    }
  }
  if (has(object, "initial")) {
    const Rcpp::List initial = object["initial"];
    read_vec_if_present(initial, "u_omega_inv", input.initial.u_omega_inv);
  }
  return input;
}

bayests::UcNormalSkewDraws read_draws(const Rcpp::List &object) {
  bayests::UcNormalSkewDraws draws;
  read_uc_common_draws(object, draws);
  if (has(object, "posterior")) {
    const Rcpp::List posterior = object["posterior"];
    read_block_draws(posterior, "skew", draws.skew);
    read_block_draws(posterior, "skew_latent", draws.skew_latent);
    read_block_draws(posterior, "u_omega_inv", draws.u_omega_inv);
  }
  return draws;
}

Rcpp::List write_draws(const bayests::UcNormalSkewDraws &draws) {
  Rcpp::List posterior = write_uc_common_draws(draws);
  posterior = with_block_draws(posterior, "skew", draws.skew);
  posterior = with_block_draws(posterior, "skew_latent", draws.skew_latent);
  posterior = with_block_draws(posterior, "u_omega_inv", draws.u_omega_inv);
  return posterior;
}

} // namespace

// [[Rcpp::export(.UcNormalSkewCoefficients)]]
Rcpp::List UcNormalSkewCoefficients(Rcpp::List object) {
  const bayests::UcNormalSkewInput input = read_input(object);
  ucmtools::RcppReporter reporter(read_verbose(object["model"]));
  const bayests::UcNormalSkewDraws draws =
    bayests::UcNormalSkewSampler().draw_coefficients(input, reporter);
  object["posterior"] = write_draws(draws);
  return object;
}

// [[Rcpp::export(.UcNormalSkewForecasts)]]
Rcpp::List UcNormalSkewForecasts(Rcpp::List object) {
  const bayests::UcNormalSkewInput input = read_input(object);
  const bayests::UcNormalSkewDraws draws = read_draws(object);
  ucmtools::RcppReporter reporter;
  const bayests::ForecastDraws forecast =
    bayests::UcNormalSkewSampler().forecast(input, draws, reporter);
  object["posterior"] = with_forecasts(object, forecast);
  return object;
}

// [[Rcpp::export(.UcNormalSkewLogLik)]]
Rcpp::List UcNormalSkewLogLik(Rcpp::List object) {
  const bayests::UcNormalSkewInput input = read_input(object);
  const bayests::UcNormalSkewDraws draws = read_draws(object);
  // Draws by periods already, the orientation WAIC and PSIS-LOO expect.
  const arma::mat loglik = bayests::UcNormalSkewSampler().log_likelihood(input, draws);
  Rcpp::List posterior = object["posterior"];
  posterior["loglik"] = loglik;
  object["posterior"] = posterior;
  return object;
}
