// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "bayests/uc_normal_stochvol.h"

#include "core/algorithms/stochvol_ocsn_2007.h"
#include "core/algorithms/triangular_packing.h"
#include "core/models/forecast_states.h"
#include "core/models/model_support.h"
#include "core/models/noncentred_support.h"
#include "core/models/predictive_score.h"
#include "core/models/uc_constant_support.h"
#include "core/models/uc_support.h"

#include <stdexcept>
#include <string>

namespace bayests
{

namespace
{

/// Psi' diag(omega_t) Psi for every period, stacked one 2k x 2k block per period.
/// `omega` is the precision of each orthogonalised shock, tt x 2k.
arma::mat precision_stack(const arma::mat &psi, const arma::mat &omega)
{
    const arma::uword w = psi.n_rows;
    const arma::uword tt = omega.n_rows;
    arma::mat out(w * tt, w);
    const arma::mat psi_t = arma::trans(psi);
    for (arma::uword t = 0; t < tt; t++)
    {
        out.rows(t * w, (t + 1) * w - 1) = psi_t * arma::diagmat(omega.row(t)) * psi;
    }
    return out;
}

/// Psi^-1 diag(1 / omega_t) Psi^-T for every period, stacked: the covariances
/// the filter reads.
arma::mat covariance_stack(const arma::mat &psi, const arma::vec &omega_inv_path)
{
    const arma::uword w = psi.n_rows;
    const arma::uword periods = omega_inv_path.n_elem / w;
    const arma::mat psi_inv = arma::inv(arma::trimatl(psi));
    arma::mat out(w * periods, w);
    for (arma::uword t = 0; t < periods; t++)
    {
        const arma::vec variance = 1.0 / omega_inv_path.subvec(t * w, (t + 1) * w - 1);
        out.rows(t * w, (t + 1) * w - 1) =
            psi_inv * arma::diagmat(variance) * arma::trans(psi_inv);
    }
    return out;
}

void require_uc_stochvol_draws(const UcNormalStochvolDraws &draws, const VarSpec &spec,
                               const arma::uword tt)
{
    const arma::uword w = static_cast<arma::uword>(spec.n_uc_shocks());
    if (draws.u_omega_inv.n_elem == 0 || draws.u_sigma_inv.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of u_omega_inv and u_sigma_inv are missing");
    }
    if (draws.u_omega_inv.n_rows != w * tt)
    {
        throw std::invalid_argument("posterior draws of u_omega_inv must have " +
                                    std::to_string(w * tt) + " rows, the 2k precisions of every "
                                    "period, got " + std::to_string(draws.u_omega_inv.n_rows));
    }
    if (spec.covar && draws.psi.n_rows != w * w)
    {
        throw std::invalid_argument("posterior draws of psi must have " + std::to_string(w * w) +
                                    " rows, got " + std::to_string(draws.psi.n_rows));
    }
    core::require_uc_coefficient_draws(spec, draws.a, draws.c, draws.tau0, draws.irregular_inv,
                                       draws.trend, draws.cycle, tt);
}

arma::mat psi_at(const VarSpec &spec, const UcNormalStochvolDraws &draws, const arma::uword i)
{
    const arma::uword w = static_cast<arma::uword>(spec.n_uc_shocks());
    if (!spec.covar)
    {
        return arma::eye<arma::mat>(w, w);
    }
    return arma::reshape(draws.psi.col(i), w, w);
}

/// The filter's log densities over the periods of `y` given each draw's
/// precision path, `omega_inv` (2k * periods) x draws.
arma::mat filtered_log_density(const VarSpec &spec, const UcNormalStochvolDraws &draws,
                               const arma::mat &omega_inv, const arma::mat &y, const arma::mat &d)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword n_draws = draws.iterations();
    arma::mat out(n_draws, y.n_cols);
    for (arma::uword i = 0; i < n_draws; i++)
    {
        const core::UcDrawParts one =
            core::uc_draw_parts(spec, draws.a, draws.c, draws.tau0, draws.irregular_inv, i);
        const core::UcStateSpace ss = core::state_space(spec, one.phi, arma::mat());
        core::UcFilterTerms terms;
        terms.sigma_path = covariance_stack(psi_at(spec, draws, i), omega_inv.col(i));
        terms.irregular_var = one.irregular_var;
        out.row(i) = core::filter_log_density(ss, y, core::drift_path(one.c, d, k, y.n_cols),
                                              one.tau0, terms);
    }
    return out;
}

} // namespace

UcNormalStochvolDraws UcNormalStochvolSampler::draw_coefficients(const UcNormalStochvolInput &input,
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
    const bool use_psi = input.use_psi();
    const bool use_irregular = input.use_irregular();

    core::UcCoefficients coef(spec, input.a_prior, input.initial.a, input.c_prior,
                              input.initial.c, input.tau0_prior, input.initial.tau0);

    // Covariance block over the 2k shocks.
    const arma::uword n_psi = use_psi ? static_cast<arma::uword>(spec.n_uc_psi()) : 0;
    arma::vec psi;
    arma::mat Psi = arma::eye<arma::mat>(w, w);
    arma::mat psi_z;
    if (use_psi)
    {
        psi = input.initial.psi;
        psi_z = arma::zeros<arma::mat>(tt * (w - 1), n_psi);
        core::fill_strict_lower_triangle(Psi, psi);
    }

    // The log-volatilities, tt x 2k, and their random walks.
    const StochvolPrior &sv_prior = input.u_sigma_prior;
    const bool noncentred = sv_prior.state.noncentred();
    arma::mat h = input.initial.h;
    arma::vec h_init = input.initial.h_init;
    arma::vec h_sigma = input.initial.h_sigma;
    const arma::vec h_post_shape =
        noncentred ? arma::vec() : arma::vec(sv_prior.state.sigma.shape + static_cast<double>(tt) * 0.5);
    arma::vec h_omega;
    arma::mat h_tilde;
    core::NoncentredCoefficients h_nc;
    if (noncentred)
    {
        h_omega = arma::sqrt(h_sigma);
    }

    arma::vec irregular_inv = use_irregular ? input.initial.irregular_inv : arma::vec();
    arma::mat q = precision_stack(Psi, arma::exp(-h));

    UcNormalStochvolDraws out;
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
    if (use_psi)
    {
        out.psi = arma::mat(w * w, iterations);
    }
    out.u_omega_inv = arma::mat(w * tt, iterations);
    out.u_sigma_inv = arma::mat(w * w * tt, iterations);
    out.h_sigma = arma::mat(w, iterations);
    if (noncentred)
    {
        core::allocate_noncentred(out.h_noncentred, w, static_cast<arma::uword>(iterations));
    }
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

        // Update the unobserved path, under this sweep's precision path
        const core::UcPath path = core::draw_uc_path(spec, y, coef.phi(),
                                                     core::drift_path(coef.c(), d, k, tt),
                                                     coef.tau0(), q, irregular_inv);

        // Update the cycle coefficients, the drift and tau0
        const core::ShockRegression reg =
            core::shock_regression_paths(spec, path.trend, path.cycle, d, coef.positions());
        if (!coef.draw(reg, q, arma::mat()))
        {
            unstationary_sweeps++;
        }
        for (arma::uword t = 0; t < tt; t++)
        {
            shocks_path.col(t) = reg.w.col(t) - reg.x.slice(t) * coef.theta();
        }

        // Update psi: shock i on the shocks before it, each period weighted by
        // the precision of its own orthogonalised shock.
        if (use_psi)
        {
            const arma::vec psi_y = arma::vectorise(shocks_path.rows(1, w - 1));
            core::build_psi_regressors(psi_z, shocks_path);
            const arma::vec weights =
                arma::vectorise(arma::trans(arma::exp(-h.cols(1, w - 1))));
            const arma::mat wz = psi_z.each_col() % weights;
            psi = core::draw_normal_precision(input.psi_prior.v_inv + arma::trans(wz) * psi_z,
                                              input.psi_prior.v_inv * input.psi_prior.mu +
                                                  arma::trans(wz) * psi_y);
            core::fill_strict_lower_triangle(Psi, psi);
        }
        const arma::mat orthogonal = Psi * shocks_path;

        // Update the log-volatilities
        if (noncentred)
        {
            h_nc = core::draw_noncentred_log_volatility(arma::trans(orthogonal), sv_prior.state,
                                                        sv_prior.offset, h, h_init, h_omega,
                                                        h_tilde);
            h_sigma = arma::square(h_omega);
        }
        else
        {
            h = stochvol_ocsn_2007(arma::trans(orthogonal), h, h_sigma, h_init, sv_prior.offset);
            core::draw_stochvol_state(h_sigma, h_init, h, h_post_shape, sv_prior.state.sigma.rate,
                                      sv_prior.state.initial_state);
        }
        q = precision_stack(Psi, arma::exp(-h));

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
            if (use_psi)
            {
                out.psi.col(pos) = arma::vectorise(Psi);
            }
            out.u_omega_inv.col(pos) = arma::vectorise(arma::trans(arma::exp(-h)));
            for (arma::uword t = 0; t < tt; t++)
            {
                out.u_sigma_inv.submat(t * w * w, pos, (t + 1) * w * w - 1, pos) =
                    arma::vectorise(q.rows(t * w, (t + 1) * w - 1));
            }
            out.h_sigma.col(pos) = h_sigma;
            if (noncentred)
            {
                core::store_noncentred(out.h_noncentred, static_cast<arma::uword>(pos), h_omega,
                                       h_nc);
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

ForecastDraws UcNormalStochvolSampler::forecast(const UcNormalStochvolInput &input,
                                                const UcNormalStochvolDraws &draws,
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
    require_uc_stochvol_draws(draws, spec, tt);
    const bool simulate = spec.forecast_states == ForecastStates::simulate;
    if (simulate && draws.h_sigma.n_rows != w)
    {
        throw std::invalid_argument("simulating the log-volatilities forward needs the variance of "
                                    "their innovations, /posterior/u_sigma_inv/sigma, " +
                                    std::to_string(w) + " per draw; forecast with "
                                    "forecast_states = hold, or estimate again");
    }

    const arma::uword h = static_cast<arma::uword>(spec.h);
    const arma::uword n_draws = draws.iterations();
    arma::mat fcst(h * k, n_draws);

    for (arma::uword i = 0; i < n_draws; i++)
    {
        reporter.check_interrupt();
        reporter.progress(static_cast<long long>(i) + 1, static_cast<long long>(n_draws));

        const core::UcDrawParts one =
            core::uc_draw_parts(spec, draws.a, draws.c, draws.tau0, draws.irregular_inv, i);
        const arma::mat psi = psi_at(spec, draws, i);
        const arma::mat tau = arma::reshape(draws.trend.col(i), k, tt);
        const arma::mat cycle = core::uc_cycle_at(spec, y, tau, draws.cycle, i);
        const arma::vec irregular_sd =
            spec.irregular ? arma::vec(arma::sqrt(one.irregular_var)) : arma::vec();

        // The log-volatility of the last sample period, stepped once per horizon
        // before that horizon's shocks are drawn.
        arma::vec log_var = -arma::log(draws.u_omega_inv.submat((tt - 1) * w, i, tt * w - 1, i));
        const arma::vec step_sigma = simulate ? arma::vec(draws.h_sigma.col(i)) : arma::vec();

        fcst.col(i) = core::uc_forecast_path(
            spec, tau, cycle, one.phi, one.c, input.forecast.x, irregular_sd, [&](arma::uword) {
                if (simulate)
                {
                    core::step_random_walk(log_var, step_sigma, arma::vec());
                }
                return arma::vec(core::covariance_root(psi, arma::exp(log_var)) *
                                 arma::randn<arma::vec>(w));
            });
    }

    reporter.finish();
    return ForecastDraws{fcst};
}

arma::mat UcNormalStochvolSampler::log_likelihood(const UcNormalStochvolInput &input,
                                                  const UcNormalStochvolDraws &draws) const
{
    const VarSpec &spec = input.spec;
    if (spec.k <= 0)
    {
        throw std::invalid_argument("model must have at least one series (k)");
    }
    const arma::mat y = core::uc_sample(spec, input.train);
    require_uc_stochvol_draws(draws, spec, y.n_cols);
    return filtered_log_density(spec, draws, draws.u_omega_inv, y, input.train.x);
}

arma::mat UcNormalStochvolSampler::predictive_log_density(const UcNormalStochvolInput &input,
                                                          const UcNormalStochvolDraws &draws) const
{
    const VarSpec &spec = input.spec;
    core::require_scorable(spec, "UcNormalStochvol");
    arma::mat y, d;
    const arma::uword periods =
        core::uc_scored_sample(spec, input.train, input.forecast, input.test, y, d);
    const arma::uword tt = y.n_cols - periods;
    require_uc_stochvol_draws(draws, spec, tt);

    const arma::uword w = static_cast<arma::uword>(spec.n_uc_shocks());
    const bool simulate = spec.forecast_states == ForecastStates::simulate;

    // The sample's precisions, then each draw's carried over the scored
    // periods as the forecast carries them.
    const arma::mat last = draws.u_omega_inv.rows((tt - 1) * w, tt * w - 1);
    const arma::mat carried =
        core::carry_log_volatility_forward(last, draws.h_sigma, periods, simulate);
    const arma::mat omega_inv = arma::join_vert(draws.u_omega_inv, carried);

    const arma::mat all = filtered_log_density(spec, draws, omega_inv, y, d);
    return all.tail_cols(periods);
}

} // namespace bayests
