// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_UC_NORMAL_STOCHVOL_H
#define BAYESTS_UC_NORMAL_STOCHVOL_H

#include "bayests/inputs.h"
#include "bayests/reporter.h"
#include "bayests/results.h"

namespace bayests
{

/// Unobserved components model with stochastic volatility on each of its 2k
/// trend and cycle shocks, optionally behind a constant covariance block. At
/// k = 1 with no cycle lags it is the UCSV model of Stock and Watson (2007), as
/// Chan (2018) estimates it with the volatilities non-centred, and in general
/// the bivariate correlated models of Grant and Chan (2017) with the stochastic
/// volatility their conclusion asks for.
///
/// Gibbs blocks: the unobserved path, whose band precision now changes with the
/// period; the cycle, drift and tau0; the covariance block, a regression whose
/// weights move with the volatilities; the log-volatilities by the
/// ten-component mixture of Omori, Chib, Shephard and Nakajima (2007), centred
/// or non-centred as the prior asks; and the irregular precisions where there
/// is an irregular component.
///
/// Values in, values out: no files, no console, no global state beyond the
/// Armadillo RNG.
class UcNormalStochvolSampler
{
public:
    UcNormalStochvolDraws draw_coefficients(const UcNormalStochvolInput &input,
                                            Reporter &reporter) const;

    /// One forecast path per draw. Under ForecastStates::simulate each draw's
    /// log-volatilities step forward a random walk innovation per horizon
    /// before the shocks of that horizon are drawn; under `hold` they stay at
    /// the last sample period. Requires the draws' `h_sigma` under `simulate`.
    ForecastDraws forecast(const UcNormalStochvolInput &input,
                           const UcNormalStochvolDraws &draws, Reporter &reporter) const;

    /// Pointwise log likelihood, draws x periods: the trend integrated out by
    /// the Kalman filter, conditional on the drawn log-volatility path -- the
    /// same conditioning VarNormalStochvol's likelihood makes.
    arma::mat log_likelihood(const UcNormalStochvolInput &input,
                             const UcNormalStochvolDraws &draws) const;

    /// The log predictive density of what the horizon realised, draws x scored
    /// periods: the filter run on over the realised values, with each draw's
    /// log-volatilities carried over the scored periods as the forecast carries
    /// them. Under `simulate` that is a draw, so it repeats only under a seed.
    arma::mat predictive_log_density(const UcNormalStochvolInput &input,
                                     const UcNormalStochvolDraws &draws) const;
};

} // namespace bayests

#endif // BAYESTS_UC_NORMAL_STOCHVOL_H
