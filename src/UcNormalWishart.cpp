#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "bayests_r_io.h"
#include "bayests_reporter.h"
#include "uc_r_translation.h"
#include "bayests/uc_normal_wishart.h"

// Unobserved components model with a Wishart prior on the precision of its 2k
// shocks, which leaves their covariance unrestricted.
// The numerics are the vendored BayesTS core; what is here is the translation
// between the R model object and the core's structs. See src/core/VENDORED.md.

namespace {

using namespace bayests_r;
using namespace ucmtools;

bayests::UcNormalWishartInput read_input(const Rcpp::List &object) {

  bayests::UcNormalWishartInput input;
  read_uc_common(object, input);

  if (has(object, "priors")) {
    const Rcpp::List priors = object["priors"];
    if (has(priors, "u_sigma")) {
      const Rcpp::List group = priors["u_sigma"];
      input.u_sigma_prior.df = optional_int(group, "df", 0);
      read_mat_if_present(group, "scale", input.u_sigma_prior.scale);
    }
  }
  if (has(object, "initial")) {
    const Rcpp::List initial = object["initial"];
    read_mat_if_present(initial, "u_sigma_inv", input.initial.u_sigma_inv);
  }
  return input;
}

bayests::UcNormalWishartDraws read_draws(const Rcpp::List &object) {
  bayests::UcNormalWishartDraws draws;
  read_uc_common_draws(object, draws);
  if (has(object, "posterior")) {
    const Rcpp::List posterior = object["posterior"];
    read_block_draws(posterior, "u_sigma_inv", draws.u_sigma_inv);
  }
  return draws;
}

Rcpp::List write_draws(const bayests::UcNormalWishartDraws &draws) {
  Rcpp::List posterior = write_uc_common_draws(draws);
  posterior = with_block_draws(posterior, "u_sigma_inv", draws.u_sigma_inv);
  return posterior;
}

} // namespace

// [[Rcpp::export(.UcNormalWishartCoefficients)]]
Rcpp::List UcNormalWishartCoefficients(Rcpp::List object) {
  const bayests::UcNormalWishartInput input = read_input(object);
  ucmtools::RcppReporter reporter(read_verbose(object["model"]));
  const bayests::UcNormalWishartDraws draws =
    bayests::UcNormalWishartSampler().draw_coefficients(input, reporter);
  object["posterior"] = write_draws(draws);
  return object;
}

// [[Rcpp::export(.UcNormalWishartForecasts)]]
Rcpp::List UcNormalWishartForecasts(Rcpp::List object) {
  const bayests::UcNormalWishartInput input = read_input(object);
  const bayests::UcNormalWishartDraws draws = read_draws(object);
  ucmtools::RcppReporter reporter;
  const bayests::ForecastDraws forecast =
    bayests::UcNormalWishartSampler().forecast(input, draws, reporter);
  object["posterior"] = with_forecasts(object, forecast);
  return object;
}

// [[Rcpp::export(.UcNormalWishartLogLik)]]
Rcpp::List UcNormalWishartLogLik(Rcpp::List object) {
  const bayests::UcNormalWishartInput input = read_input(object);
  const bayests::UcNormalWishartDraws draws = read_draws(object);
  // Draws by periods already, the orientation WAIC and PSIS-LOO expect.
  const arma::mat loglik = bayests::UcNormalWishartSampler().log_likelihood(input, draws);
  Rcpp::List posterior = object["posterior"];
  posterior["loglik"] = loglik;
  object["posterior"] = posterior;
  return object;
}

// [[Rcpp::export(.UcNormalWishartScore)]]
Rcpp::List UcNormalWishartScore(Rcpp::List object) {
  const bayests::UcNormalWishartInput input = read_input(object);
  const bayests::UcNormalWishartDraws draws = read_draws(object);
  // Draws by scored periods already; it crosses as it stands.
  const arma::mat score = bayests::UcNormalWishartSampler().predictive_log_density(input, draws);
  object["posterior"] = with_forecast_member(Rcpp::List(object["posterior"]), "loglik", score);
  return object;
}
