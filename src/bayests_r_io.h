// SPDX-License-Identifier: GPL-2.0-or-later

#ifndef UCMTOOLS_BAYESTS_R_IO_H
#define UCMTOOLS_BAYESTS_R_IO_H

#include <RcppArmadillo.h>

#include "bayests/data.h"
#include "bayests/priors.h"
#include "bayests/results.h"
#include "bayests/spec.h"

#include <string>

// Translation between the R model object and the structs the vendored BayesTS
// core takes. This is the R counterpart of that project's src/io/hdf5/ layer:
// the only place that knows both an `Rcpp::List` and a `bayests::` struct, so
// that the sampler knows neither.
//
// One convention is worth stating once, because getting it wrong is silent
// rather than loud: draws run along the columns inside the core and along the
// rows in R, so everything crossing this boundary goes through draws_to_r() or
// read_draws_if_present().
//
// This is dfmtools' copy of the same header in bvartools, unchanged but for
// this paragraph. What an unobserved components model reads beyond it -- the
// four models' shared blocks -- is in uc_r_translation.h.
namespace bayests_r
{

inline bool has(const Rcpp::List &list, const char *name)
{
  return list.containsElementNamed(name) && !Rf_isNull(list[name]);
}

inline int optional_int(const Rcpp::List &list, const char *name, int fallback)
{
  return has(list, name) ? Rcpp::as<int>(list[name]) : fallback;
}

inline void read_mat_if_present(const Rcpp::List &list, const char *name, arma::mat &out)
{
  if (has(list, name)) {
    out = Rcpp::as<arma::mat>(list[name]);
  }
}

inline void read_vec_if_present(const Rcpp::List &list, const char *name, arma::vec &out)
{
  if (has(list, name)) {
    out = Rcpp::as<arma::vec>(list[name]);
  }
}

/// Draws as the core keeps them, from the row-per-draw matrix R keeps.
inline void read_draws_if_present(const Rcpp::List &list, const char *name, arma::mat &out)
{
  if (has(list, name)) {
    out = arma::trans(Rcpp::as<arma::mat>(list[name]));
  }
}

/// Draws as R expects them, from the column-per-draw matrix the core returns.
inline arma::mat draws_to_r(const arma::mat &draws)
{
  return arma::trans(draws);
}

/// A block's draws with what the non-centred parameterisation adds beside its
/// `sigma`: the signed standard deviation and the log ordinates at zero the
/// Savage-Dickey test for time variation is built from. Unchanged for a block
/// the caller gave `shape` and `rate`, which is what makes it safe to call on
/// every random walk of a model whether or not `omega_v` was set.
///
/// The draws arrive in the core's ordering of the states, so a block whose
/// rows R orders differently -- the free loadings, which this package permutes
/// at both ends -- is permuted by its binding before it gets here. The joint
/// ordinate is one number per draw and has no ordering to get wrong.
inline Rcpp::List with_noncentred(Rcpp::List block, const bayests::NoncentredStateDraws &nc)
{
  if (!nc.empty()) {
    block["omega"] = draws_to_r(nc.omega);
    block["omega_log_zero"] = draws_to_r(nc.log_zero);
    block["omega_log_zero_joint"] = draws_to_r(nc.log_zero_joint);
  }
  return block;
}

/// Whether a forecast carries the random walks of a time-varying model over the
/// horizon or holds them at the last sample period: `model$forecast_states`,
/// written by add_posterior_forecasts() when it was given one, and `simulate`
/// -- the core's own default -- when it is absent.
/// Whether the sampler reports its progress: `model$verbose`, written by
/// add_posterior_coefficients() when it was asked to. Read from the model list
/// rather than taken as an argument, the way `forecast_states` is, so that no
/// binding signature has to change for it.
inline bool read_verbose(const Rcpp::List &model)
{
  return has(model, "verbose") && Rcpp::as<bool>(model["verbose"]);
}

inline bayests::ForecastStates read_forecast_states(const Rcpp::List &model)
{
  return has(model, "forecast_states")
           ? bayests::forecast_states_from_string(Rcpp::as<std::string>(model["forecast_states"]))
           : bayests::ForecastStates::simulate;
}

/// The (shape, rate) pair every gamma prior is stored as.
inline bayests::GammaPrior read_gamma_prior(const Rcpp::List &group)
{
  bayests::GammaPrior prior;
  read_vec_if_present(group, "shape", prior.shape);
  read_vec_if_present(group, "rate", prior.rate);
  return prior;
}

/// The deterministic terms of a factor model: `data$deterministic`, one row per
/// period and one column per term, which create_dfmodel() and
/// create_favarmodel() build, and `data$forecast$x`, the same terms over the
/// horizon, which add_forecast_input() adds. Their count is the core's `n`, so
/// a model without them -- no `data$deterministic` -- has none, and nothing
/// below reads the other two.
inline void read_deterministic_terms(const Rcpp::List &object, bayests::VarSpec &spec,
                                     bayests::TrainData &train, bayests::ForecastData &forecast)
{
  if (!has(object, "data")) {
    return;
  }
  const Rcpp::List data = object["data"];
  read_mat_if_present(data, "deterministic", train.x);
  spec.n = static_cast<int>(train.x.n_cols);
  if (has(data, "forecast")) {
    const Rcpp::List horizon = data["forecast"];
    read_mat_if_present(horizon, "x", forecast.x);
  }
}

/// A normal coefficient block's prior at `priors$<name>` -- `mu` and `vinv`, the
/// R spelling of the core's `v_inv` -- and its starting value at
/// `initial$<name>`, each where present. An absent mean is zero.
inline void read_normal_block(const Rcpp::List &object, const char *name,
                              bayests::NormalPrior &prior, arma::vec &initial)
{
  if (has(object, "priors")) {
    const Rcpp::List priors = object["priors"];
    if (has(priors, name)) {
      const Rcpp::List group = priors[name];
      read_mat_if_present(group, "vinv", prior.v_inv);
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

/// `posterior$<name>$coeffs`, where present, as the core keeps draws.
inline void read_block_draws(const Rcpp::List &posterior, const char *name, arma::mat &out)
{
  if (has(posterior, name)) {
    read_draws_if_present(Rcpp::List(posterior[name]), "coeffs", out);
  }
}

/// The posterior list with `draws` added under `name` as a `coeffs` block, or
/// unchanged where there are none -- which is every model without
/// deterministic terms.
inline Rcpp::List with_block_draws(Rcpp::List posteriors, const char *name, const arma::mat &draws)
{
  if (draws.n_elem > 0) {
    posteriors.push_back(Rcpp::List::create(Rcpp::Named("coeffs") = draws_to_r(draws)), name);
  }
  return posteriors;
}

/// The forecast group of the posterior, with `value` under `name`.
///
/// posterior$forecast is a list rather than a matrix of draws: everything the
/// forecast periods produce hangs below it -- `forecasts`, the simulated paths,
/// and `loglik`, the score against what those periods realised. It is the layout
/// bvartools uses and the one the model file writes as /posterior/forecast, and
/// the members are named after what they hold because all of them are draws.
inline Rcpp::List with_forecast_member(Rcpp::List posterior, const char *name,
                                       const arma::mat &value) {
  const Rcpp::RObject wrapped = Rcpp::wrap(value);
  Rcpp::List forecast;
  if (posterior.containsElementNamed("forecast") &&
      Rf_isNewList(posterior["forecast"])) {
    forecast = Rcpp::List(Rf_shallow_duplicate(posterior["forecast"]));
  }

  if (forecast.containsElementNamed(name)) {
    forecast[name] = wrapped;
  } else {
    forecast.push_back(wrapped, name);
  }

  if (posterior.containsElementNamed("forecast")) {
    posterior["forecast"] = forecast;
  } else {
    posterior.push_back(forecast, "forecast");
  }
  return posterior;
}

} // namespace bayests_r

#endif // UCMTOOLS_BAYESTS_R_IO_H
