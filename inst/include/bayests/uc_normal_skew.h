// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_UC_NORMAL_SKEW_H
#define BAYESTS_UC_NORMAL_SKEW_H

#include "bayests/inputs.h"
#include "bayests/reporter.h"
#include "bayests/results.h"

namespace bayests
{

/// Unobserved components model with skewed cycle -- or trend -- shocks whose
/// scales move with an index, after Huang, Luo and Wang (2026); see
/// UcNormalSkewInput for the model.
///
/// Gibbs blocks: the unobserved path, given the skewed shocks' latent variates,
/// under which the model is Gaussian with a shock mean and a variance that move
/// with the period; the cycle, drift and tau0; the precision of the shock that
/// is not skewed; the scale coefficients of each series, by a random walk
/// Metropolis-Hastings step against the skewed shocks' marginal density, the
/// latents integrated out; the latents given them; and the irregular
/// precisions where there is an irregular component. The step's proposal
/// covariance adapts to the chain during the burn-in (Haario, Saksman and
/// Tamminen 2001) and is held from then on, so the kept draws come from a fixed
/// kernel. Its acceptance rate is reported through Reporter::message().
///
/// Huang, Y.-F., Luo, S., & Wang, H.-J. (2026). Macro-economic uncertainty and
/// business cycle asymmetry. Journal of Money, Credit and Banking.
class UcNormalSkewSampler
{
public:
    UcNormalSkewDraws draw_coefficients(const UcNormalSkewInput &input, Reporter &reporter) const;

    /// One forecast path per draw: each horizon's skewed shocks drawn from
    /// their distribution at the scales `input.forecast_skew_x` implies, the
    /// other shocks normal.
    ForecastDraws forecast(const UcNormalSkewInput &input, const UcNormalSkewDraws &draws,
                           Reporter &reporter) const;

    /// Pointwise log likelihood, draws x periods: the trend integrated out by
    /// the Kalman filter, conditional on the drawn latent variates of the
    /// skewed shocks, under which the model is Gaussian. The skewed shocks'
    /// density is the latents' mixture over that, and integrating them out as
    /// well is not a Kalman filter any more; so this is a conditional
    /// likelihood, as the factor models' is, and an information criterion built
    /// from it treats the latents as parameters.
    arma::mat log_likelihood(const UcNormalSkewInput &input, const UcNormalSkewDraws &draws) const;
};

} // namespace bayests

#endif // BAYESTS_UC_NORMAL_SKEW_H
