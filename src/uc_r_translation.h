// SPDX-License-Identifier: GPL-2.0-or-later

#ifndef UCMTOOLS_UC_R_TRANSLATION_H
#define UCMTOOLS_UC_R_TRANSLATION_H

#include <RcppArmadillo.h>

#include "bayests_r_io.h"

#include <string>

// What the four unobserved components bindings read and write alike: the
// specification, the data, the three coefficient blocks every member of the
// family has, the irregular component, and the trend and cycle paths. Each
// binding adds its own error block.
//
// The R object names everything as the BayesTS model file does -- priors$tau0,
// initial$u_omega_inv, posterior$trend -- so that write_to_hdf5() writes it as
// it stands and a reader of one knows the other. A normal block's precision is
// therefore `v_inv` here, where bvartools and dfmtools say `vinv`.
namespace ucmtools
{

using namespace bayests_r;

/// Whether the error specification carries a covariance block: "gamma+covar"
/// or "sv+covar", the spellings of the model files.
inline bool error_has_covar(const std::string &error)
{
  const std::string suffix = "+covar";
  return error.size() > suffix.size() &&
         error.compare(error.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// `priors$<name>` as a normal block, `initial$<name>` as its start. Unlike
/// read_normal_block() in bayests_r_io.h this reads `v_inv`, the name the model
/// file and this package use.
inline void read_uc_normal_block(const Rcpp::List &object, const char *name,
                                 bayests::NormalPrior &prior, arma::vec &initial)
{
  if (has(object, "priors")) {
    const Rcpp::List priors = object["priors"];
    if (has(priors, name)) {
      const Rcpp::List group = priors[name];
      read_mat_if_present(group, "v_inv", prior.v_inv);
      read_vec_if_present(group, "mu", prior.mu);
      if (prior.mu.is_empty()) {
        prior.mu = arma::vec(prior.v_inv.n_rows, arma::fill::zeros);
      }
    }
  }
  if (has(object, "initial")) {
    const Rcpp::List start = object["initial"];
    read_vec_if_present(start, name, initial);
  }
}

/// The specification, the data and the shared blocks of any of the four inputs.
template <typename Input>
void read_uc_common(const Rcpp::List &object, Input &input)
{
  const Rcpp::List model = object["model"];
  input.spec.k = Rcpp::as<int>(model["k"]);
  input.spec.p = optional_int(model, "p", 0);
  input.spec.iterations = Rcpp::as<int>(model["iterations"]);
  input.spec.burnin = Rcpp::as<int>(model["burnin"]);
  input.spec.thin = optional_int(model, "thin", 1);
  input.spec.h = optional_int(model, "h", 0);
  input.spec.forecast_states = read_forecast_states(model);

  if (has(model, "trend")) {
    input.spec.trend = bayests::uc_trend_from_string(Rcpp::as<std::string>(model["trend"]));
  }
  if (has(model, "cycle")) {
    input.spec.cycle = bayests::uc_cycle_from_string(Rcpp::as<std::string>(model["cycle"]));
  }
  if (has(model, "irregular")) {
    input.spec.irregular = Rcpp::as<bool>(model["irregular"]);
  }
  if (has(model, "stationary")) {
    input.spec.stationary = Rcpp::as<bool>(model["stationary"]);
  }
  if (has(model, "error")) {
    input.spec.covar = error_has_covar(Rcpp::as<std::string>(model["error"]));
  }
  if (has(model, "skew")) {
    input.spec.skew = bayests::uc_skew_from_string(Rcpp::as<std::string>(model["skew"]));
  }
  if (has(model, "skew_shock")) {
    input.spec.skew_shock =
      bayests::uc_skew_shock_from_string(Rcpp::as<std::string>(model["skew_shock"]));
  }

  if (has(object, "data")) {
    const Rcpp::List data = object["data"];
    read_mat_if_present(data, "y", input.train.y);
    // What the horizon realised, where the model carries it: one row per period
    // and one column per series, as `data$y` is.
    if (has(data, "test")) {
      const Rcpp::List test = data["test"];
      read_mat_if_present(test, "y", input.test.y);
    }
  }

  // The drift's deterministic terms, data$deterministic and over the horizon
  // data$forecast$x, and their count, which is the core's `n`.
  read_deterministic_terms(object, input.spec, input.train, input.forecast);

  read_uc_normal_block(object, "c", input.c_prior, input.initial.c);
  if (input.spec.p > 0) {
    read_uc_normal_block(object, "a", input.a_prior, input.initial.a);
  }
  read_uc_normal_block(object, "tau0", input.tau0_prior, input.initial.tau0);

  if (input.spec.irregular) {
    if (has(object, "priors")) {
      const Rcpp::List priors = object["priors"];
      if (has(priors, "irregular")) {
        input.irregular_prior = read_gamma_prior(Rcpp::List(priors["irregular"]));
      }
    }
    if (has(object, "initial")) {
      const Rcpp::List initial = object["initial"];
      read_vec_if_present(initial, "irregular_inv", input.initial.irregular_inv);
    }
  }
}

/// The shared blocks of the posterior, as the core keeps draws.
template <typename Draws>
void read_uc_common_draws(const Rcpp::List &object, Draws &draws)
{
  if (!has(object, "posterior")) {
    return;
  }
  const Rcpp::List posterior = object["posterior"];
  read_block_draws(posterior, "trend", draws.trend);
  read_block_draws(posterior, "cycle", draws.cycle);
  read_block_draws(posterior, "a", draws.a);
  read_block_draws(posterior, "c", draws.c);
  read_block_draws(posterior, "tau0", draws.tau0);
  read_block_draws(posterior, "irregular_inv", draws.irregular_inv);
}

/// The shared blocks as the R posterior list, rows per draw. A block the model
/// does not have -- the cycle without an irregular component, `a` at p = 0, `c`
/// without deterministic terms -- is left out rather than stored empty.
template <typename Draws>
Rcpp::List write_uc_common_draws(const Draws &draws)
{
  Rcpp::List posterior;
  posterior = with_block_draws(posterior, "trend", draws.trend);
  posterior = with_block_draws(posterior, "cycle", draws.cycle);
  posterior = with_block_draws(posterior, "a", draws.a);
  posterior = with_block_draws(posterior, "c", draws.c);
  posterior = with_block_draws(posterior, "tau0", draws.tau0);
  posterior = with_block_draws(posterior, "irregular_inv", draws.irregular_inv);
  return posterior;
}

/// The forecast group of the posterior with the simulated paths, one row per
/// draw and k * h columns stacked by period.
inline Rcpp::List with_forecasts(const Rcpp::List &object, const bayests::ForecastDraws &forecast)
{
  return with_forecast_member(Rcpp::List(object["posterior"]), "forecasts",
                              draws_to_r(forecast.values));
}

} // namespace ucmtools

#endif // UCMTOOLS_UC_R_TRANSLATION_H
