// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_UC_NORMAL_GAMMA_H
#define BAYESTS_UC_NORMAL_GAMMA_H

#include "bayests/inputs.h"
#include "bayests/reporter.h"
#include "bayests/results.h"

namespace bayests
{

/// Unobserved components model: each of k series the sum of a stochastic trend
/// and a stationary cycle, with gamma priors on the precisions of their shocks
/// and, optionally, a constant covariance block over them.
///
///     y_t   = tau_t + c_t,
///     c_t   = sum_{j=1..p} Phi_j c_{t-j} + u^c_t,
///     tau_t = tau_{t-1} + C d_t + u^tau_t        (or the smooth trend; UcTrend)
///
/// It nests the trend-cycle decompositions of Clark (1987), Morley, Nelson and
/// Zivot (2003) and Grant and Chan (2017), each with or without a break in the
/// drift, the local level model at p = 0, the smooth-trend output gap model of
/// Chan, Koop, Poirier and Tobias (2019, ch. 18), and with k > 1 the bivariate
/// model of Sinclair (2009). See UcNormalGammaInput for the data and the
/// priors.
///
/// Four Gibbs blocks: the trend path, then the cycle coefficients, the drift
/// and the trend's starting values jointly, then the covariance block, then the
/// precisions. The trend path is drawn whole, from its precision, as Grant and
/// Chan (2017) do. Stacked period by period it is block banded of bandwidth
/// max(p, trend order) with k x k blocks, whatever the covariance block
/// correlates, so `chan_jeliazkov_2009_precision` draws it at O(tt k^3).
///
/// Values in, values out: no files, no console, no global state beyond the
/// Armadillo RNG.
///
/// Grant, A. L., & Chan, J. C. C. (2017). A Bayesian model comparison for
/// trend-cycle decompositions of output. Journal of Money, Credit and Banking,
/// 49(2-3), 525-552.
class UcNormalGammaSampler
{
public:
    /// Runs the Gibbs sampler. Reports progress once per draw and honours an
    /// interrupt thrown from the reporter.
    ///
    /// Throws std::invalid_argument if `input` is inconsistent.
    UcNormalGammaDraws draw_coefficients(const UcNormalGammaInput &input,
                                         Reporter &reporter) const;

    /// Simulates one forecast path of the series per posterior draw: the trend
    /// and the cycle run forward from the end of the drawn trend path, with a
    /// shock drawn for both at every step, and summed. The one out-of-sample
    /// input is `input.forecast.x`, the deterministic terms of the horizon, which
    /// a model with them adds to the trend's increment.
    ///
    /// Requires `draws.trend`. Throws std::invalid_argument if it is missing or
    /// if spec.h is zero.
    ForecastDraws forecast(const UcNormalGammaInput &input, const UcNormalGammaDraws &draws,
                           Reporter &reporter) const;

    /// Pointwise log likelihood, draws x periods, as expected by WAIC and
    /// PSIS-LOO.
    ///
    /// With the trend integrated out: column t is p(y_t | y_1 .. y_{t-1},
    /// theta), the one step prediction error decomposition of a Kalman filter
    /// started from the draw's tau0 and a zero cycle, where theta is everything
    /// but the trend path. That is the integrated likelihood Grant and Chan
    /// (2017, app. B) compare these models by, and not the density conditional
    /// on the drawn trend that the factor models report: a trend treated as a
    /// parameter fits every period it is allowed to move in, which makes the
    /// conditional density useless for telling a model with a trend shock from
    /// one without.
    ///
    /// Draws nothing. Requires `draws.tau0`.
    arma::mat log_likelihood(const UcNormalGammaInput &input,
                             const UcNormalGammaDraws &draws) const;

    /// The log predictive density of what the horizon realised, draws x scored
    /// periods: the same filter as log_likelihood(), run on over `input.test.y`
    /// after the sample, so each scored period is predicted from everything
    /// before it -- the sample and the scored periods already realised.
    ///
    /// Draws nothing. Requires `input.test.y`, and `input.forecast.x` where the
    /// model has deterministic terms.
    arma::mat predictive_log_density(const UcNormalGammaInput &input,
                                     const UcNormalGammaDraws &draws) const;
};

} // namespace bayests

#endif // BAYESTS_UC_NORMAL_GAMMA_H
