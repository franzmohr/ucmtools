// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "bayests/uc_normal_gamma.h"

#include "core/algorithms/triangular_packing.h"
#include "core/models/model_support.h"
#include "core/models/uc_constant_support.h"
#include "core/models/uc_support.h"

#include <stdexcept>
#include <string>

namespace bayests
{

namespace
{

using core::build_psi_regressors;
using core::draw_normal_precision;
using core::fill_strict_lower_triangle;
using core::stacked_response;

} // namespace

UcNormalGammaDraws UcNormalGammaSampler::draw_coefficients(const UcNormalGammaInput &input,
                                                           Reporter &reporter) const
{
    input.validate();

    const VarSpec &spec = input.spec;
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword shocks = 2 * k;
    const int iterations = spec.iterations;
    const int draws = spec.draws();

    const arma::vec stacked = stacked_response(input.train);
    const arma::uword tt = stacked.n_elem / k;
    const arma::mat y = arma::reshape(stacked, k, tt);
    const arma::mat &d = input.train.x;

    const bool use_psi = input.use_psi();
    const bool use_irregular = input.use_irregular();

    // The cycle, the drift and tau0, drawn jointly given the path.
    core::UcCoefficients coef(spec, input.a_prior, input.initial.a, input.c_prior,
                              input.initial.c, input.tau0_prior, input.initial.tau0);
    const arma::uword n_tau0 = k * static_cast<arma::uword>(spec.uc_trend_order());

    // Covariance block over the 2k shocks, exactly as VarNormalGamma's over its
    // k errors: Psi e_t ~ N(0, Omega^-1).
    const arma::uword n_psi = use_psi ? static_cast<arma::uword>(spec.n_uc_psi()) : 0;
    arma::vec psi;
    arma::mat Psi = arma::eye<arma::mat>(shocks, shocks);
    arma::mat psi_z;
    if (use_psi)
    {
        psi = input.initial.psi;
        psi_z = arma::zeros<arma::mat>(tt * (shocks - 1), n_psi);
        fill_strict_lower_triangle(Psi, psi);
    }

    const arma::vec omega_post_shape = input.u_sigma_prior.shape + static_cast<double>(tt) * 0.5;
    arma::vec omega = input.initial.u_omega_inv;
    arma::mat q = arma::trans(Psi) * arma::diagmat(omega) * Psi;

    arma::vec irregular_inv = use_irregular ? input.initial.irregular_inv : arma::vec();

    UcNormalGammaDraws out;
    out.trend = arma::mat(k * tt, iterations);
    if (coef.use_a())
    {
        out.a = arma::mat(k * k * p, iterations);
    }
    if (coef.use_c())
    {
        out.c = arma::mat(k * static_cast<arma::uword>(spec.n), iterations);
    }
    out.tau0 = arma::mat(n_tau0, iterations);
    if (use_psi)
    {
        out.psi = arma::mat(shocks * shocks, iterations);
    }
    out.u_omega_inv = arma::mat(shocks, iterations);
    out.u_sigma_inv = arma::mat(shocks * shocks, iterations);
    if (use_irregular)
    {
        out.cycle = arma::mat(k * tt, iterations);
        out.irregular_inv = arma::mat(k, iterations);
    }

    int unstationary_sweeps = 0;
    arma::mat shocks_path(shocks, tt);

    for (int draw = 0; draw < draws; draw++)
    {
        reporter.check_interrupt();
        reporter.progress(draw + 1, draws);

        // Update the unobserved path: the trend, and the cycle with it where an
        // irregular component separates the two from the series.
        const core::UcPath path = core::draw_uc_path(spec, y, coef.phi(),
                                                     core::drift_path(coef.c(), d, k, tt),
                                                     coef.tau0(), q, irregular_inv);

        // Update the cycle coefficients, the drift and tau0, jointly given it
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

        // Update psi
        if (use_psi)
        {
            const arma::vec psi_y = arma::vectorise(shocks_path.rows(1, shocks - 1));
            build_psi_regressors(psi_z, shocks_path);
            const arma::mat omega_lower = arma::diagmat(omega.tail(shocks - 1));
            const arma::sp_mat weight =
                arma::kron(arma::speye<arma::sp_mat>(tt, tt), arma::sp_mat(omega_lower));
            const arma::mat wz = weight * psi_z;
            psi = draw_normal_precision(input.psi_prior.v_inv + arma::trans(wz) * psi_z,
                                        input.psi_prior.v_inv * input.psi_prior.mu +
                                            arma::trans(wz) * psi_y);
            fill_strict_lower_triangle(Psi, psi);
            shocks_path = Psi * shocks_path;
        }

        // Update Omega
        const arma::vec sse = arma::sum(arma::square(shocks_path), 1);
        for (arma::uword i = 0; i < shocks; i++)
        {
            omega(i) = arma::randg<double>(arma::distr_param(
                omega_post_shape(i), 1.0 / (input.u_sigma_prior.rate(i) + sse(i) * 0.5)));
        }
        q = arma::trans(Psi) * arma::diagmat(omega) * Psi;

        // Update the irregular component's precisions
        if (use_irregular)
        {
            irregular_inv = core::draw_irregular_precision(input.irregular_prior, y, path);
        }

        // Store draws
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
            out.u_omega_inv.col(pos) = omega;
            out.u_sigma_inv.col(pos) = arma::vectorise(q);
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

ForecastDraws UcNormalGammaSampler::forecast(const UcNormalGammaInput &input,
                                             const UcNormalGammaDraws &draws,
                                             Reporter &reporter) const
{
    return core::uc_constant_forecast(input.spec, input.train, input.forecast, draws, reporter);
}

arma::mat UcNormalGammaSampler::log_likelihood(const UcNormalGammaInput &input,
                                               const UcNormalGammaDraws &draws) const
{
    return core::uc_constant_log_likelihood(input.spec, input.train, draws);
}

arma::mat UcNormalGammaSampler::predictive_log_density(const UcNormalGammaInput &input,
                                                       const UcNormalGammaDraws &draws) const
{
    return core::uc_constant_predictive_log_density(input.spec, input.train, input.forecast,
                                                    input.test, draws, "UcNormalGamma");
}

} // namespace bayests
