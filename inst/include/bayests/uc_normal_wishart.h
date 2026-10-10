// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_UC_NORMAL_WISHART_H
#define BAYESTS_UC_NORMAL_WISHART_H

#include "bayests/inputs.h"
#include "bayests/reporter.h"
#include "bayests/results.h"

namespace bayests
{

/// Unobserved components model with a Wishart prior on the precision of its 2k
/// trend and cycle shocks: UcNormalGammaSampler's model with the shock
/// covariance unrestricted. This is the bivariate correlated model of Sinclair
/// (2009) that Grant and Chan (2017, sec. 3.3) estimate under an inverse Wishart
/// prior centred at the identity, and at k = 1 the correlated model of Morley,
/// Nelson and Zivot (2003) with the correlation free rather than a loading.
///
/// Three Gibbs blocks: the unobserved path, the cycle, drift and tau0 jointly
/// given it, and the shock precision, Wishart given the shocks -- plus the
/// irregular precisions where there is an irregular component. The path and
/// the coefficients are drawn as UcNormalGamma draws them; only the precision
/// differs.
///
/// Values in, values out: no files, no console, no global state beyond the
/// Armadillo RNG.
class UcNormalWishartSampler
{
public:
    /// Runs the Gibbs sampler. Throws std::invalid_argument if `input` is
    /// inconsistent.
    UcNormalWishartDraws draw_coefficients(const UcNormalWishartInput &input,
                                           Reporter &reporter) const;

    /// One forecast path per draw, as UcNormalGammaSampler::forecast.
    ForecastDraws forecast(const UcNormalWishartInput &input, const UcNormalWishartDraws &draws,
                           Reporter &reporter) const;

    /// Pointwise log likelihood with the trend integrated out, draws x periods;
    /// see UcNormalGammaSampler::log_likelihood.
    arma::mat log_likelihood(const UcNormalWishartInput &input,
                             const UcNormalWishartDraws &draws) const;

    /// The log predictive density of what the horizon realised, draws x scored
    /// periods; see UcNormalGammaSampler::predictive_log_density.
    arma::mat predictive_log_density(const UcNormalWishartInput &input,
                                     const UcNormalWishartDraws &draws) const;
};

} // namespace bayests

#endif // BAYESTS_UC_NORMAL_WISHART_H
