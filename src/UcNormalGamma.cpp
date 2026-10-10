#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "bayests_r_io.h"
#include "bayests_reporter.h"
#include "uc_r_translation.h"
#include "bayests/uc_normal_gamma.h"

// Unobserved components model with gamma priors on the precisions of its 2k
// shocks and, under error "gamma+covar", a constant covariance block over them.
// The numerics are the vendored BayesTS core; what is here is the translation
// between the R model object and the core's structs. See src/core/VENDORED.md.

namespace {

using namespace bayests_r;
using namespace ucmtools;

bayests::UcNormalGammaInput read_input(const Rcpp::List &object) {

  bayests::UcNormalGammaInput input;
  read_uc_common(object, input);

  if (input.use_psi()) {
    read_uc_normal_block(object, "psi", input.psi_prior, input.initial.psi);
  }
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

bayests::UcNormalGammaDraws read_draws(const Rcpp::List &object) {
  bayests::UcNormalGammaDraws draws;
  read_uc_common_draws(object, draws);
  if (has(object, "posterior")) {
    const Rcpp::List posterior = object["posterior"];
    read_block_draws(posterior, "psi", draws.psi);
    read_block_draws(posterior, "u_omega_inv", draws.u_omega_inv);
    read_block_draws(posterior, "u_sigma_inv", draws.u_sigma_inv);
  }
  return draws;
}

Rcpp::List write_draws(const bayests::UcNormalGammaDraws &draws) {
  Rcpp::List posterior = write_uc_common_draws(draws);
  posterior = with_block_draws(posterior, "psi", draws.psi);
  posterior = with_block_draws(posterior, "u_omega_inv", draws.u_omega_inv);
  posterior = with_block_draws(posterior, "u_sigma_inv", draws.u_sigma_inv);
  return posterior;
}

} // namespace

// [[Rcpp::export(.UcNormalGammaCoefficients)]]
Rcpp::List UcNormalGammaCoefficients(Rcpp::List object) {
  const bayests::UcNormalGammaInput input = read_input(object);
  ucmtools::RcppReporter reporter(read_verbose(object["model"]));
  const bayests::UcNormalGammaDraws draws =
    bayests::UcNormalGammaSampler().draw_coefficients(input, reporter);
  object["posterior"] = write_draws(draws);
  return object;
}

// [[Rcpp::export(.UcNormalGammaForecasts)]]
Rcpp::List UcNormalGammaForecasts(Rcpp::List object) {
  const bayests::UcNormalGammaInput input = read_input(object);
  const bayests::UcNormalGammaDraws draws = read_draws(object);
  ucmtools::RcppReporter reporter;
  const bayests::ForecastDraws forecast =
    bayests::UcNormalGammaSampler().forecast(input, draws, reporter);
  object["posterior"] = with_forecasts(object, forecast);
  return object;
}

// [[Rcpp::export(.UcNormalGammaLogLik)]]
Rcpp::List UcNormalGammaLogLik(Rcpp::List object) {
  const bayests::UcNormalGammaInput input = read_input(object);
  const bayests::UcNormalGammaDraws draws = read_draws(object);
  // Draws by periods already, the orientation WAIC and PSIS-LOO expect.
  const arma::mat loglik = bayests::UcNormalGammaSampler().log_likelihood(input, draws);
  Rcpp::List posterior = object["posterior"];
  posterior["loglik"] = loglik;
  object["posterior"] = posterior;
  return object;
}

// [[Rcpp::export(.UcNormalGammaScore)]]
Rcpp::List UcNormalGammaScore(Rcpp::List object) {
  const bayests::UcNormalGammaInput input = read_input(object);
  const bayests::UcNormalGammaDraws draws = read_draws(object);
  // Draws by scored periods already; it crosses as it stands.
  const arma::mat score = bayests::UcNormalGammaSampler().predictive_log_density(input, draws);
  object["posterior"] = with_forecast_member(Rcpp::List(object["posterior"]), "loglik", score);
  return object;
}
