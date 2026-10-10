// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_CORE_MODELS_UC_CONSTANT_SUPPORT_H
#define BAYESTS_CORE_MODELS_UC_CONSTANT_SUPPORT_H

// The forecast, the log likelihood and the score of an unobserved components
// model whose shock precision is one matrix per draw -- UcNormalGamma and
// UcNormalWishart, which differ only in how that matrix is drawn. Templates over
// the draws struct, which both lay out the same way: `trend`, `cycle`, `a`, `c`,
// `tau0`, `irregular_inv` and a (2k)^2-row `u_sigma_inv`.

#include "bayests/results.h"
#include "bayests/reporter.h"
#include "core/models/forecast_states.h"
#include "core/models/model_support.h"
#include "core/models/predictive_score.h"
#include "core/models/uc_support.h"

#include <stdexcept>
#include <string>

namespace bayests::core
{

template <typename Draws>
void require_uc_constant_draws(const Draws &draws, const VarSpec &spec, const arma::uword tt)
{
    const arma::uword shocks = static_cast<arma::uword>(spec.n_uc_shocks());
    if (draws.u_sigma_inv.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of u_sigma_inv are missing");
    }
    if (draws.u_sigma_inv.n_rows != shocks * shocks)
    {
        throw std::invalid_argument("posterior draws of u_sigma_inv must have " +
                                    std::to_string(shocks * shocks) +
                                    " rows, the precision of the 2k shocks, got " +
                                    std::to_string(draws.u_sigma_inv.n_rows));
    }
    require_uc_coefficient_draws(spec, draws.a, draws.c, draws.tau0, draws.irregular_inv,
                                 draws.trend, draws.cycle, tt);
}

/// The pointwise log density over the periods of `y`, one row per draw, with
/// the trend -- and the cycle, under an irregular component -- integrated out.
template <typename Draws>
arma::mat uc_constant_filtered_log_density(const VarSpec &spec, const Draws &draws,
                                           const arma::mat &y, const arma::mat &d)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword shocks = 2 * k;
    const arma::uword n_draws = draws.iterations();
    arma::mat out(n_draws, y.n_cols);
    for (arma::uword i = 0; i < n_draws; i++)
    {
        const UcDrawParts one = uc_draw_parts(spec, draws.a, draws.c, draws.tau0,
                                              draws.irregular_inv, i);
        const arma::mat q = arma::reshape(draws.u_sigma_inv.col(i), shocks, shocks);
        const arma::mat sigma = uc_covariance_of_precision(q);
        const UcStateSpace ss = state_space(spec, one.phi, sigma);
        UcFilterTerms terms;
        terms.irregular_var = one.irregular_var;
        // Under UcCycleInit::stationary the cycle before the sample is
        // integrated out from its stationary distribution, as the trend path is.
        const arma::mat initial_cycle_cov =
            spec.cycle_init == UcCycleInit::stationary
                ? cycle_stationary_covariance(one.phi, sigma.submat(0, 0, k - 1, k - 1))
                : arma::mat();
        out.row(i) = filter_log_density(ss, y, drift_path(one.c, d, k, y.n_cols), one.tau0, terms,
                                        initial_cycle_cov);
    }
    return out;
}

/// The sample as k x tt.
inline arma::mat uc_sample(const VarSpec &spec, const TrainData &train)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::vec stacked = stacked_response(train);
    return arma::reshape(stacked, k, stacked.n_elem / k);
}

/// The checks a forecast makes before it draws: a horizon, the drift's terms
/// over it, and the paths it starts from.
template <typename Draws>
void require_uc_forecastable(const VarSpec &spec, const ForecastData &forecast,
                             const Draws &draws, const arma::uword tt)
{
    if (spec.h <= 0)
    {
        throw std::invalid_argument("forecast horizon (h) must be positive");
    }
    const arma::uword h = static_cast<arma::uword>(spec.h);
    if (draws.trend.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of the trend path are missing, and the "
                                    "forecast starts from its end");
    }
    if (spec.irregular && draws.cycle.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of the cycle path are missing, and with an "
                                    "irregular component the forecast starts from its end");
    }
    if (spec.n > 0 &&
        (forecast.x.n_rows != h || forecast.x.n_cols != static_cast<arma::uword>(spec.n)))
    {
        throw std::invalid_argument(
            "the drift's " + std::to_string(spec.n) + " deterministic terms have to be continued "
            "over the horizon: /data/forecast/x must be " + std::to_string(h) + " x " +
            std::to_string(spec.n) + ", got " + std::to_string(forecast.x.n_rows) + " x " +
            std::to_string(forecast.x.n_cols));
    }
    if (spec.trend == UcTrend::smooth && tt < 2)
    {
        throw std::invalid_argument("a smooth trend is forecast from its last two periods, and the "
                                    "sample has one");
    }
}

/// One forecast path per draw, the 2k shocks of each horizon drawn through the
/// draw's covariance root.
template <typename Draws>
ForecastDraws uc_constant_forecast(const VarSpec &spec, const TrainData &train,
                                   const ForecastData &forecast, const Draws &draws,
                                   Reporter &reporter)
{
    if (spec.k <= 0)
    {
        throw std::invalid_argument("model must have at least one series (k)");
    }
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword shocks = 2 * k;
    const arma::mat y = uc_sample(spec, train);
    const arma::uword tt = y.n_cols;

    require_uc_forecastable(spec, forecast, draws, tt);
    require_uc_constant_draws(draws, spec, tt);

    const arma::uword h = static_cast<arma::uword>(spec.h);
    const arma::uword n_draws = draws.iterations();
    arma::mat fcst(h * k, n_draws);

    for (arma::uword i = 0; i < n_draws; i++)
    {
        reporter.check_interrupt();
        reporter.progress(static_cast<long long>(i) + 1, static_cast<long long>(n_draws));

        const UcDrawParts one = uc_draw_parts(spec, draws.a, draws.c, draws.tau0,
                                              draws.irregular_inv, i);
        const arma::mat root =
            covariance_root(arma::reshape(draws.u_sigma_inv.col(i), shocks, shocks));
        const arma::mat tau = arma::reshape(draws.trend.col(i), k, tt);
        const arma::mat cycle = uc_cycle_at(spec, y, tau, draws.cycle, i);
        const arma::vec irregular_sd =
            spec.irregular ? arma::vec(arma::sqrt(one.irregular_var)) : arma::vec();

        fcst.col(i) = uc_forecast_path(
            spec, tau, cycle, one.phi, one.c, forecast.x, irregular_sd,
            [&](arma::uword) { return arma::vec(root * arma::randn<arma::vec>(shocks)); });
    }

    reporter.finish();
    return ForecastDraws{fcst};
}

template <typename Draws>
arma::mat uc_constant_log_likelihood(const VarSpec &spec, const TrainData &train,
                                     const Draws &draws)
{
    if (spec.k <= 0)
    {
        throw std::invalid_argument("model must have at least one series (k)");
    }
    const arma::mat y = uc_sample(spec, train);
    require_uc_constant_draws(draws, spec, y.n_cols);
    if (spec.n > 0 && train.x.n_rows != y.n_cols)
    {
        throw std::invalid_argument("the deterministic terms /data/train/x must have one row per "
                                    "period, " + std::to_string(y.n_cols) + ", got " +
                                    std::to_string(train.x.n_rows));
    }
    return uc_constant_filtered_log_density(spec, draws, y, train.x);
}

/// The sample followed by the realised periods, and the deterministic terms
/// over both, for a filter that runs on through the scored periods. Returns the
/// number of scored periods.
inline arma::uword uc_scored_sample(const VarSpec &spec, const TrainData &train,
                                    const ForecastData &forecast, const TestData &test,
                                    arma::mat &y, arma::mat &d)
{
    const arma::uword periods = scored_horizons(test.y, spec);
    const arma::uword k = static_cast<arma::uword>(spec.k);
    if (test.y.n_cols != k)
    {
        throw std::invalid_argument("the realised observations /data/test/y must have one column "
                                    "per series, " + std::to_string(k) + ", got " +
                                    std::to_string(test.y.n_cols));
    }
    y = arma::join_horiz(uc_sample(spec, train), arma::trans(test.y.head_rows(periods)));
    d.reset();
    if (spec.n > 0)
    {
        if (forecast.x.n_rows < periods || forecast.x.n_cols != static_cast<arma::uword>(spec.n))
        {
            throw std::invalid_argument(
                "scoring needs the drift's deterministic terms over the scored periods: "
                "/data/forecast/x must have at least " + std::to_string(periods) + " rows and " +
                std::to_string(spec.n) + " columns");
        }
        d = arma::join_vert(train.x, forecast.x.head_rows(periods));
    }
    return periods;
}

template <typename Draws>
arma::mat uc_constant_predictive_log_density(const VarSpec &spec, const TrainData &train,
                                             const ForecastData &forecast, const TestData &test,
                                             const Draws &draws, const char *algorithm)
{
    require_scorable(spec, algorithm);
    arma::mat y, d;
    const arma::uword periods = uc_scored_sample(spec, train, forecast, test, y, d);
    require_uc_constant_draws(draws, spec, y.n_cols - periods);

    // The sample and then the scored periods, as one series the filter runs
    // over: each scored period is predicted from everything before it.
    const arma::mat all = uc_constant_filtered_log_density(spec, draws, y, d);
    return all.tail_cols(periods);
}

} // namespace bayests::core

#endif // BAYESTS_CORE_MODELS_UC_CONSTANT_SUPPORT_H
