// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "bayests/uc_normal_skew.h"

#include "core/models/adaptive_walk.h"
#include "core/models/model_support.h"
#include "core/models/uc_constant_support.h"
#include "core/models/uc_skew_support.h"
#include "core/models/uc_support.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace bayests
{

namespace
{

using core::AdaptiveWalk;
using core::conditional_moments;
using core::draw_latent;
using core::latent_mean;
using core::log_marginal;
using core::skew_scales;
using core::SkewScales;

/// [1, z_t'] for every period.
arma::mat skew_design(const arma::mat &z, const arma::uword periods)
{
    arma::mat out(periods, 1 + z.n_cols, arma::fill::ones);
    if (z.n_cols > 0)
    {
        out.cols(1, z.n_cols) = z.head_rows(periods);
    }
    return out;
}

/// Which of the 2k shocks is series i's skewed one, and which its normal one.
arma::uword skewed_row(const VarSpec &spec, const arma::uword i)
{
    return spec.skew_shock == UcSkewShock::cycle ? i : static_cast<arma::uword>(spec.k) + i;
}
arma::uword normal_row(const VarSpec &spec, const arma::uword i)
{
    return spec.skew_shock == UcSkewShock::cycle ? static_cast<arma::uword>(spec.k) + i : i;
}

/// The shock means and variances of every period given the scales and the
/// latents: 2k x periods each, the normal shocks' mean zero.
void shock_moments(const VarSpec &spec, const arma::mat &design, const arma::vec &skew,
                   const arma::mat &latent, const arma::vec &omega, arma::mat &mean,
                   arma::mat &var)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword periods = latent.n_cols;
    const arma::uword d = 2 * design.n_cols;
    mean = arma::zeros<arma::mat>(2 * k, periods);
    var = arma::mat(2 * k, periods);
    for (arma::uword i = 0; i < k; i++)
    {
        const SkewScales s = skew_scales(design, skew.subvec(i * d, (i + 1) * d - 1));
        const arma::uword r = skewed_row(spec, i);
        for (arma::uword t = 0; t < periods; t++)
        {
            conditional_moments(spec.skew, s.a(t), s.b(t), latent(i, t), mean(r, t), var(r, t));
        }
        var.row(normal_row(spec, i)).fill(1.0 / omega(i));
    }
}

/// diag(1 / var_t) for every period, stacked.
arma::mat precision_stack(const arma::mat &var)
{
    const arma::uword w = var.n_rows;
    arma::mat out(w * var.n_cols, w, arma::fill::zeros);
    for (arma::uword t = 0; t < var.n_cols; t++)
    {
        out.rows(t * w, (t + 1) * w - 1).diag() = 1.0 / var.col(t);
    }
    return out;
}

/// diag(var_t) for every period, stacked.
arma::mat covariance_stack(const arma::mat &var)
{
    const arma::uword w = var.n_rows;
    arma::mat out(w * var.n_cols, w, arma::fill::zeros);
    for (arma::uword t = 0; t < var.n_cols; t++)
    {
        out.rows(t * w, (t + 1) * w - 1).diag() = var.col(t);
    }
    return out;
}

double log_prior(const NormalPrior &prior, const arma::vec &theta)
{
    const arma::vec dev = theta - prior.mu;
    return -0.5 * arma::as_scalar(arma::trans(dev) * prior.v_inv * dev);
}

void require_uc_skew_draws(const UcNormalSkewDraws &draws, const VarSpec &spec,
                           const arma::uword tt, const arma::uword n_skew_coef)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    if (draws.u_omega_inv.n_rows != k || draws.u_omega_inv.n_cols == 0)
    {
        throw std::invalid_argument("posterior draws of u_omega_inv must have " +
                                    std::to_string(k) + " rows, got " +
                                    std::to_string(draws.u_omega_inv.n_rows));
    }
    if (draws.skew.n_rows != n_skew_coef)
    {
        throw std::invalid_argument("posterior draws of the skew coefficients must have " +
                                    std::to_string(n_skew_coef) + " rows, two log scales per "
                                    "series each on a constant and the index, got " +
                                    std::to_string(draws.skew.n_rows));
    }
    if (draws.skew_latent.n_rows != k * tt)
    {
        throw std::invalid_argument("posterior draws of the skew latents must have " +
                                    std::to_string(k * tt) + " rows, got " +
                                    std::to_string(draws.skew_latent.n_rows));
    }
    core::require_uc_coefficient_draws(spec, draws.a, draws.c, draws.tau0, draws.irregular_inv,
                                       draws.trend, draws.cycle, tt);
}

} // namespace

UcNormalSkewDraws UcNormalSkewSampler::draw_coefficients(const UcNormalSkewInput &input,
                                                         Reporter &reporter) const
{
    input.validate();

    const VarSpec &spec = input.spec;
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword w = 2 * k;
    const int iterations = spec.iterations;
    const int draws = spec.draws();

    const arma::mat y = core::uc_sample(spec, input.train);
    const arma::uword tt = y.n_cols;
    const arma::mat &d = input.train.x;
    const bool use_irregular = input.use_irregular();

    core::UcCoefficients coef(spec, input.a_prior, input.initial.a, input.c_prior,
                              input.initial.c, input.tau0_prior, input.initial.tau0);

    const arma::mat design = skew_design(input.train_skew_x, tt);
    const arma::uword d_skew = 2 * design.n_cols;
    arma::vec skew = input.initial.skew;
    arma::mat latent(k, tt, arma::fill::value(latent_mean(spec.skew)));
    arma::vec omega = input.initial.u_omega_inv;
    const arma::vec omega_post_shape = input.u_sigma_prior.shape + static_cast<double>(tt) * 0.5;
    arma::vec irregular_inv = use_irregular ? input.initial.irregular_inv : arma::vec();

    std::vector<AdaptiveWalk> walks(k, AdaptiveWalk(d_skew));
    arma::uword proposals = 0, accepted = 0;

    arma::mat mean, var;
    shock_moments(spec, design, skew, latent, omega, mean, var);
    arma::mat q = precision_stack(var);

    UcNormalSkewDraws out;
    out.trend = arma::mat(k * tt, iterations);
    if (coef.use_a())
    {
        out.a = arma::mat(k * k * p, iterations);
    }
    if (coef.use_c())
    {
        out.c = arma::mat(k * static_cast<arma::uword>(spec.n), iterations);
    }
    out.tau0 = arma::mat(k * static_cast<arma::uword>(spec.uc_trend_order()), iterations);
    out.skew = arma::mat(k * d_skew, iterations);
    out.skew_latent = arma::mat(k * tt, iterations);
    out.u_omega_inv = arma::mat(k, iterations);
    if (use_irregular)
    {
        out.cycle = arma::mat(k * tt, iterations);
        out.irregular_inv = arma::mat(k, iterations);
    }

    int unstationary_sweeps = 0;
    arma::mat shocks_path(w, tt);

    for (int draw = 0; draw < draws; draw++)
    {
        reporter.check_interrupt();
        reporter.progress(draw + 1, draws);

        // Update the unobserved path, Gaussian given the latents
        const core::UcPath path = core::draw_uc_path(spec, y, coef.phi(),
                                                     core::drift_path(coef.c(), d, k, tt),
                                                     coef.tau0(), q, irregular_inv, mean);

        // Update the cycle coefficients, the drift and tau0
        const core::ShockRegression reg =
            core::shock_regression_paths(spec, path.trend, path.cycle, d, coef.positions());
        if (!coef.draw(reg, q, mean))
        {
            unstationary_sweeps++;
        }
        for (arma::uword t = 0; t < tt; t++)
        {
            shocks_path.col(t) = reg.w.col(t) - reg.x.slice(t) * coef.theta();
        }

        for (arma::uword i = 0; i < k; i++)
        {
            // Update the precision of the normal shock
            const arma::rowvec normal = shocks_path.row(normal_row(spec, i));
            omega(i) = arma::randg<double>(arma::distr_param(
                omega_post_shape(i),
                1.0 / (input.u_sigma_prior.rate(i) + arma::dot(normal, normal) * 0.5)));

            // Update the scale coefficients against the skewed shocks' marginal
            // density, the latents integrated out
            const arma::rowvec skewed = shocks_path.row(skewed_row(spec, i));
            const arma::uword at = i * d_skew;
            const auto log_target = [&](const arma::vec &theta_i) {
                arma::vec all = skew;
                all.subvec(at, at + d_skew - 1) = theta_i;
                return log_marginal(spec.skew, skewed, skew_scales(design, theta_i)) +
                       log_prior(input.skew_prior, all);
            };
            const arma::vec current = skew.subvec(at, at + d_skew - 1);
            const arma::vec proposal = walks[i].propose(current);
            const double log_ratio = log_target(proposal) - log_target(current);
            const bool accept = std::log(arma::randu<double>()) < log_ratio;
            if (accept)
            {
                skew.subvec(at, at + d_skew - 1) = proposal;
            }
            if (draw < spec.burnin)
            {
                walks[i].learn(skew.subvec(at, at + d_skew - 1), std::exp(std::min(log_ratio, 0.0)));
            }
            else
            {
                proposals++;
                accepted += accept ? 1 : 0;
            }

            // Update the latents given the scales
            const SkewScales s = skew_scales(design, skew.subvec(at, at + d_skew - 1));
            for (arma::uword t = 0; t < tt; t++)
            {
                latent(i, t) = draw_latent(spec.skew, s.a(t), s.b(t), skewed(t));
            }
        }

        shock_moments(spec, design, skew, latent, omega, mean, var);
        q = precision_stack(var);

        // Update the irregular component's precisions
        if (use_irregular)
        {
            irregular_inv = core::draw_irregular_precision(input.irregular_prior, y, path);
        }

        if (spec.keeps(draw))
        {
            const int pos = spec.kept_index(draw);
            out.trend.col(pos) = arma::vectorise(path.trend);
            if (coef.use_a())
            {
                out.a.col(pos) = arma::vectorise(coef.phi());
            }
            if (coef.use_c())
            {
                out.c.col(pos) = coef.c_vec();
            }
            out.tau0.col(pos) = coef.tau0();
            out.skew.col(pos) = skew;
            out.skew_latent.col(pos) = arma::vectorise(latent);
            out.u_omega_inv.col(pos) = omega;
            if (use_irregular)
            {
                out.cycle.col(pos) = arma::vectorise(path.cycle);
                out.irregular_inv.col(pos) = irregular_inv;
            }
        }
    }

    reporter.finish();
    if (proposals > 0)
    {
        reporter.message("the skew coefficients' Metropolis-Hastings step accepted " +
                         std::to_string(accepted) + " of " + std::to_string(proposals) +
                         " proposals after the burn-in");
    }
    if (spec.burnin < static_cast<int>(10 * d_skew))
    {
        reporter.message("a burn-in of " + std::to_string(spec.burnin) + " is too short for the "
                         "skew coefficients' proposal to adapt, which takes " +
                         std::to_string(10 * d_skew) + " draws; it stayed a scaled identity");
    }
    if (unstationary_sweeps > 0)
    {
        reporter.message(std::to_string(unstationary_sweeps) + " of " + std::to_string(draws) +
                         " sweeps found no stationary draw of the cycle in " +
                         std::to_string(core::kStationaryTries) +
                         " tries and kept the one before: the posterior may put much of its mass "
                         "on explosive cycle coefficients");
    }
    return out;
}

ForecastDraws UcNormalSkewSampler::forecast(const UcNormalSkewInput &input,
                                            const UcNormalSkewDraws &draws,
                                            Reporter &reporter) const
{
    const VarSpec &spec = input.spec;
    if (spec.k <= 0)
    {
        throw std::invalid_argument("model must have at least one series (k)");
    }
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword w = 2 * k;
    const arma::mat y = core::uc_sample(spec, input.train);
    const arma::uword tt = y.n_cols;

    core::require_uc_forecastable(spec, input.forecast, draws, tt);
    const arma::uword h = static_cast<arma::uword>(spec.h);
    const arma::uword n_skew = input.n_skew();
    if (n_skew > 0 && (input.forecast_skew_x.n_rows != h || input.forecast_skew_x.n_cols != n_skew))
    {
        throw std::invalid_argument(
            "the skew index has to be continued over the horizon: /data/forecast/skew_x must be " +
            std::to_string(h) + " x " + std::to_string(n_skew) + ", got " +
            std::to_string(input.forecast_skew_x.n_rows) + " x " +
            std::to_string(input.forecast_skew_x.n_cols));
    }
    const arma::mat design = skew_design(input.forecast_skew_x, h);
    const arma::uword d_skew = 2 * design.n_cols;
    require_uc_skew_draws(draws, spec, tt, k * d_skew);

    const arma::uword n_draws = draws.iterations();
    arma::mat fcst(h * k, n_draws);

    for (arma::uword i = 0; i < n_draws; i++)
    {
        reporter.check_interrupt();
        reporter.progress(static_cast<long long>(i) + 1, static_cast<long long>(n_draws));

        const core::UcDrawParts one =
            core::uc_draw_parts(spec, draws.a, draws.c, draws.tau0, draws.irregular_inv, i);
        const arma::mat tau = arma::reshape(draws.trend.col(i), k, tt);
        const arma::mat cycle = core::uc_cycle_at(spec, y, tau, draws.cycle, i);
        const arma::vec irregular_sd =
            spec.irregular ? arma::vec(arma::sqrt(one.irregular_var)) : arma::vec();
        const arma::vec skew = draws.skew.col(i);
        const arma::vec omega = draws.u_omega_inv.col(i);

        std::vector<SkewScales> scales(k);
        for (arma::uword j = 0; j < k; j++)
        {
            scales[j] = skew_scales(design, skew.subvec(j * d_skew, (j + 1) * d_skew - 1));
        }

        fcst.col(i) = core::uc_forecast_path(
            spec, tau, cycle, one.phi, one.c, input.forecast.x, irregular_sd,
            [&](arma::uword step) {
                arma::vec e(w);
                for (arma::uword j = 0; j < k; j++)
                {
                    const double latent =
                        spec.skew == UcSkew::snd
                            ? std::abs(arma::randn<double>())
                            : arma::randg<double>(arma::distr_param(1.0, 1.0));
                    double m = 0.0, v = 0.0;
                    conditional_moments(spec.skew, scales[j].a(step), scales[j].b(step), latent,
                                        m, v);
                    e(skewed_row(spec, j)) = m + std::sqrt(v) * arma::randn<double>();
                    e(normal_row(spec, j)) = arma::randn<double>() / std::sqrt(omega(j));
                }
                return e;
            });
    }

    reporter.finish();
    return ForecastDraws{fcst};
}

arma::mat UcNormalSkewSampler::log_likelihood(const UcNormalSkewInput &input,
                                              const UcNormalSkewDraws &draws) const
{
    const VarSpec &spec = input.spec;
    if (spec.k <= 0)
    {
        throw std::invalid_argument("model must have at least one series (k)");
    }
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::mat y = core::uc_sample(spec, input.train);
    const arma::uword tt = y.n_cols;
    const arma::mat design = skew_design(input.train_skew_x, tt);
    const arma::uword d_skew = 2 * design.n_cols;
    require_uc_skew_draws(draws, spec, tt, k * d_skew);

    const arma::uword n_draws = draws.iterations();
    arma::mat out(n_draws, tt);
    arma::mat mean, var;
    for (arma::uword i = 0; i < n_draws; i++)
    {
        const core::UcDrawParts one =
            core::uc_draw_parts(spec, draws.a, draws.c, draws.tau0, draws.irregular_inv, i);
        shock_moments(spec, design, draws.skew.col(i), arma::reshape(draws.skew_latent.col(i), k, tt),
                      draws.u_omega_inv.col(i), mean, var);
        const core::UcStateSpace ss = core::state_space(spec, one.phi, arma::mat());
        core::UcFilterTerms terms;
        terms.sigma_path = covariance_stack(var);
        terms.shock_mean = mean;
        terms.irregular_var = one.irregular_var;
        out.row(i) = core::filter_log_density(ss, y, core::drift_path(one.c, input.train.x, k, tt),
                                              one.tau0, terms);
    }
    return out;
}

} // namespace bayests
