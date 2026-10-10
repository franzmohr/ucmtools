// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "bayests/uc_normal_wishart.h"

#include "core/algorithms/wishart.h"
#include "core/models/model_support.h"
#include "core/models/uc_constant_support.h"
#include "core/models/uc_support.h"

#include <algorithm>
#include <string>

namespace bayests
{

UcNormalWishartDraws UcNormalWishartSampler::draw_coefficients(const UcNormalWishartInput &input,
                                                               Reporter &reporter) const
{
    input.validate();

    const VarSpec &spec = input.spec;
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword shocks = 2 * k;
    const int iterations = spec.iterations;
    const int draws = spec.draws();

    const arma::mat y = core::uc_sample(spec, input.train);
    const arma::uword tt = y.n_cols;
    const arma::mat &d = input.train.x;
    const bool use_irregular = input.use_irregular();

    core::UcCoefficients coef(spec, input.a_prior, input.initial.a, input.c_prior,
                              input.initial.c, input.tau0_prior, input.initial.tau0);

    const int post_df = input.u_sigma_prior.df + static_cast<int>(tt);
    arma::mat q = input.initial.u_sigma_inv;
    arma::vec irregular_inv = use_irregular ? input.initial.irregular_inv : arma::vec();

    UcNormalWishartDraws out;
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
    out.u_sigma_inv = arma::mat(shocks * shocks, iterations);
    if (use_irregular)
    {
        out.cycle = arma::mat(k * tt, iterations);
        out.irregular_inv = arma::mat(k, iterations);
    }

    // The cycle before the sample; see UcNormalGammaSampler.
    core::UcCyclePresample presample(spec, coef.phi(), q);
    if (presample.active())
    {
        out.cycle0 = arma::mat(k * p, iterations);
    }

    int unstationary_sweeps = 0;
    arma::mat shocks_path(shocks, tt);

    for (int draw = 0; draw < draws; draw++)
    {
        reporter.check_interrupt();
        reporter.progress(draw + 1, draws);

        // Update the unobserved path
        const core::UcPath path = core::draw_uc_path(spec, y, coef.phi(),
                                                     core::drift_path(coef.c(), d, k, tt),
                                                     coef.tau0(), q, irregular_inv, arma::mat(),
                                                     presample.value());

        // Update the cycle coefficients, the drift and tau0, jointly given it
        core::ShockRegression reg = core::shock_regression_paths(
            spec, path.trend, path.cycle, d, coef.positions(), presample.value());
        const arma::vec theta_before = coef.theta();
        if (!coef.draw(reg, q, arma::mat()))
        {
            unstationary_sweeps++;
        }
        else if (!presample.accept(coef.phi(), q))
        {
            coef.set_theta(theta_before);
        }

        // Update the cycle before the sample, given its first p periods
        if (presample.active())
        {
            const core::ShockRegression zero = core::shock_regression_paths(
                spec, path.trend, path.cycle, d, coef.positions());
            arma::mat shocks_zero(shocks, std::min<arma::uword>(p, tt));
            for (arma::uword t = 0; t < shocks_zero.n_cols; t++)
            {
                shocks_zero.col(t) = zero.w.col(t) - zero.x.slice(t) * coef.theta();
            }
            presample.draw(coef.phi(), shocks_zero, q);
            reg = core::shock_regression_paths(spec, path.trend, path.cycle, d, coef.positions(),
                                               presample.value());
        }
        for (arma::uword t = 0; t < tt; t++)
        {
            shocks_path.col(t) = reg.w.col(t) - reg.x.slice(t) * coef.theta();
        }

        // Update the shock precision
        const arma::mat q_new = wishart(shocks_path, input.u_sigma_prior.scale, post_df);
        if (presample.accept(coef.phi(), q_new))
        {
            q = q_new;
        }

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
            out.u_sigma_inv.col(pos) = arma::vectorise(q);
            if (presample.active())
            {
                out.cycle0.col(pos) = presample.value();
            }
            if (use_irregular)
            {
                out.cycle.col(pos) = arma::vectorise(path.cycle);
                out.irregular_inv.col(pos) = irregular_inv;
            }
        }
    }

    reporter.finish();
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

ForecastDraws UcNormalWishartSampler::forecast(const UcNormalWishartInput &input,
                                               const UcNormalWishartDraws &draws,
                                               Reporter &reporter) const
{
    return core::uc_constant_forecast(input.spec, input.train, input.forecast, draws, reporter);
}

arma::mat UcNormalWishartSampler::log_likelihood(const UcNormalWishartInput &input,
                                                 const UcNormalWishartDraws &draws) const
{
    return core::uc_constant_log_likelihood(input.spec, input.train, draws);
}

arma::mat UcNormalWishartSampler::predictive_log_density(const UcNormalWishartInput &input,
                                                         const UcNormalWishartDraws &draws) const
{
    return core::uc_constant_predictive_log_density(input.spec, input.train, input.forecast,
                                                    input.test, draws, "UcNormalWishart");
}

} // namespace bayests
