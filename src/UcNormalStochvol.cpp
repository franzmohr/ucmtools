#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "bayests_r_io.h"
#include "bayests_reporter.h"
#include "uc_r_translation.h"
#include "bayests/uc_normal_stochvol.h"

// Unobserved components model with stochastic volatility on each of its 2k
// shocks and, under error "sv+covar", a constant covariance block in front of
// them. The stochastic volatility prior is read under the names bvartools and
// dfmtools give it -- offset, shape, rate, omega_v, mu, v_inv -- and `sigma`,
// the starting value of the log-volatility innovation variance, sits in the
// prior group because that is where the model file keeps it.
// The numerics are the vendored BayesTS core; what is here is the translation
// between the R model object and the core's structs. See src/core/VENDORED.md.

namespace {

using namespace bayests_r;
using namespace ucmtools;

bayests::UcNormalStochvolInput read_input(const Rcpp::List &object) {

  bayests::UcNormalStochvolInput input;
  read_uc_common(object, input);

  if (input.use_psi()) {
    read_uc_normal_block(object, "psi", input.psi_prior, input.initial.psi);
  }
  if (has(object, "priors")) {
    const Rcpp::List priors = object["priors"];
    if (has(priors, "u_sigma")) {
      const Rcpp::List group = priors["u_sigma"];
      read_vec_if_present(group, "offset", input.u_sigma_prior.offset);
      read_vec_if_present(group, "shape", input.u_sigma_prior.state.sigma.shape);
      read_vec_if_present(group, "rate", input.u_sigma_prior.state.sigma.rate);
      read_vec_if_present(group, "omega_v", input.u_sigma_prior.state.omega_v);
      read_vec_if_present(group, "mu", input.u_sigma_prior.state.initial_state.mu);
      read_mat_if_present(group, "v_inv", input.u_sigma_prior.state.initial_state.v_inv);
      read_vec_if_present(group, "sigma", input.initial.h_sigma);
    }
  }
  if (has(object, "initial")) {
    const Rcpp::List initial = object["initial"];
    read_mat_if_present(initial, "h", input.initial.h);
    read_vec_if_present(initial, "h_init", input.initial.h_init);
  }
  return input;
}

bayests::UcNormalStochvolDraws read_draws(const Rcpp::List &object) {
  bayests::UcNormalStochvolDraws draws;
  read_uc_common_draws(object, draws);
  if (has(object, "posterior")) {
    const Rcpp::List posterior = object["posterior"];
    read_block_draws(posterior, "psi", draws.psi);
    read_block_draws(posterior, "u_omega_inv", draws.u_omega_inv);
    read_block_draws(posterior, "u_sigma_inv", draws.u_sigma_inv);
    if (has(posterior, "u_sigma_inv")) {
      read_draws_if_present(Rcpp::List(posterior["u_sigma_inv"]), "sigma", draws.h_sigma);
    }
  }
  return draws;
}

Rcpp::List write_draws(const bayests::UcNormalStochvolDraws &draws) {
  Rcpp::List posterior = write_uc_common_draws(draws);
  posterior = with_block_draws(posterior, "psi", draws.psi);
  posterior = with_block_draws(posterior, "u_omega_inv", draws.u_omega_inv);
  // The precision path with the variance of the log-volatility innovations
  // beside it and, under the non-centred prior, the signed standard deviation
  // and the Savage-Dickey ordinates.
  Rcpp::List block = Rcpp::List::create(Rcpp::Named("coeffs") = draws_to_r(draws.u_sigma_inv),
                                        Rcpp::Named("sigma") = draws_to_r(draws.h_sigma));
  posterior.push_back(with_noncentred(block, draws.h_noncentred), "u_sigma_inv");
  return posterior;
}

} // namespace

// [[Rcpp::export(.UcNormalStochvolCoefficients)]]
Rcpp::List UcNormalStochvolCoefficients(Rcpp::List object) {
  const bayests::UcNormalStochvolInput input = read_input(object);
  ucmtools::RcppReporter reporter(read_verbose(object["model"]));
  const bayests::UcNormalStochvolDraws draws =
    bayests::UcNormalStochvolSampler().draw_coefficients(input, reporter);
  object["posterior"] = write_draws(draws);
  return object;
}

// [[Rcpp::export(.UcNormalStochvolForecasts)]]
Rcpp::List UcNormalStochvolForecasts(Rcpp::List object) {
  const bayests::UcNormalStochvolInput input = read_input(object);
  const bayests::UcNormalStochvolDraws draws = read_draws(object);
  ucmtools::RcppReporter reporter;
  const bayests::ForecastDraws forecast =
    bayests::UcNormalStochvolSampler().forecast(input, draws, reporter);
  object["posterior"] = with_forecasts(object, forecast);
  return object;
}

// [[Rcpp::export(.UcNormalStochvolLogLik)]]
Rcpp::List UcNormalStochvolLogLik(Rcpp::List object) {
  const bayests::UcNormalStochvolInput input = read_input(object);
  const bayests::UcNormalStochvolDraws draws = read_draws(object);
  // Draws by periods already, the orientation WAIC and PSIS-LOO expect.
  const arma::mat loglik = bayests::UcNormalStochvolSampler().log_likelihood(input, draws);
  Rcpp::List posterior = object["posterior"];
  posterior["loglik"] = loglik;
  object["posterior"] = posterior;
  return object;
}

// [[Rcpp::export(.UcNormalStochvolScore)]]
Rcpp::List UcNormalStochvolScore(Rcpp::List object) {
  const bayests::UcNormalStochvolInput input = read_input(object);
  const bayests::UcNormalStochvolDraws draws = read_draws(object);
  // Draws by scored periods already; it crosses as it stands.
  const arma::mat score = bayests::UcNormalStochvolSampler().predictive_log_density(input, draws);
  object["posterior"] = with_forecast_member(Rcpp::List(object["posterior"]), "loglik", score);
  return object;
}
