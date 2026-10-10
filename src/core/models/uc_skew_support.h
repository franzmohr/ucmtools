// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_CORE_MODELS_UC_SKEW_SUPPORT_H
#define BAYESTS_CORE_MODELS_UC_SKEW_SUPPORT_H

// The skewed shocks of UcNormalSkew: their two closed form marginal densities,
// the normal mixture each is, and the latent variate's conditional given the
// shock. Here rather than in the sampler's source so that unit.uc_normal_gamma
// can put each density against its mixture integrated numerically and each
// latent draw against its conditional mean.

#include "bayests/arma.h"
#include "bayests/spec.h"
#include "core/algorithms/gig_hormann_leydold_2014.h"
#include "core/algorithms/truncated_normal.h"

#include <algorithm>
#include <cmath>

namespace bayests::core
{

inline double log_2pi() { return std::log(2.0 * arma::datum::pi); }

/// log Phi(z), accurate where Phi itself underflows: below -30 by the
/// asymptotic expansion of the Mills ratio, Phi(z) ~ phi(z) / |z| (1 - 1/z^2 +
/// 3/z^4), whose error there is under 1e-8 in relative terms.
inline double log_normal_cdf(const double z)
{
    if (z > -30.0)
    {
        return std::log(0.5 * std::erfc(-z / std::sqrt(2.0)));
    }
    const double z2 = z * z;
    return -0.5 * z2 - 0.5 * log_2pi() - std::log(-z) + std::log(1.0 - 1.0 / z2 + 3.0 / (z2 * z2));
}

/// The two scales of a series' skewed shock, period by period: sigma_v and
/// sigma_u for the skew normal, lambda_r and lambda_w for the asymmetric
/// Laplace, each exp(design * coefficients).
struct SkewScales
{
    arma::vec a;
    arma::vec b;
};

inline SkewScales skew_scales(const arma::mat &design, const arma::vec &theta)
{
    const arma::uword m = design.n_cols;
    SkewScales out;
    out.a = arma::exp(design * theta.head(m));
    out.b = arma::exp(design * theta.tail(m));
    return out;
}

/// sum_t log f(x_t), the skewed shock's marginal density with its latent
/// variate integrated out:
///
///   skew normal:          f(x) = 2/s phi(x/s) Phi(-sigma_u x / (sigma_v s)),
///                         s^2 = sigma_v^2 + sigma_u^2 (Azzalini 1985);
///   asymmetric Laplace:   f(x) = exp(x / lambda_w) / (lambda_r + lambda_w), x <= 0,
///                                exp(-x / lambda_r) / (lambda_r + lambda_w), x > 0.
inline double log_marginal(const UcSkew skew, const arma::rowvec &x, const SkewScales &s)
{
    double out = 0.0;
    for (arma::uword t = 0; t < x.n_elem; t++)
    {
        const double a = s.a(t);
        const double b = s.b(t);
        if (skew == UcSkew::snd)
        {
            const double sd = std::sqrt(a * a + b * b);
            const double u = x(t) / sd;
            out += std::log(2.0) - std::log(sd) - 0.5 * log_2pi() - 0.5 * u * u +
                   log_normal_cdf(-b * x(t) / (a * sd));
        }
        else
        {
            out += (x(t) <= 0.0 ? x(t) / b : -x(t) / a) - std::log(a + b);
        }
    }
    return out;
}

/// The skewed shock's mean and variance given its latent variate L: -sigma_u L
/// and sigma_v^2 for the skew normal, (lambda_r - lambda_w) L and
/// 2 lambda_r lambda_w L for the asymmetric Laplace -- whose normal mixture is
/// over a unit exponential L, so that r - w = (lambda_r - lambda_w) L +
/// sqrt(2 lambda_r lambda_w L) Z has the characteristic function
/// 1 / ((1 - i lambda_r u)(1 + i lambda_w u)) of the difference of the two
/// exponentials.
inline void conditional_moments(const UcSkew skew, const double a, const double b, const double latent,
                         double &mean, double &var)
{
    if (skew == UcSkew::snd)
    {
        mean = -b * latent;
        var = a * a;
    }
    else
    {
        mean = (a - b) * latent;
        var = 2.0 * a * b * latent;
    }
}

/// The latent variate given the shock: a normal truncated to the positive half
/// line for the skew normal, mean -sigma_u x / s^2 and variance sigma_v^2 / s^2;
/// a generalised inverse Gaussian for the asymmetric Laplace,
/// GIG(1/2, x^2 / delta^2, 2 + mu^2 / delta^2) with mu and delta^2 the mean
/// and variance factors above.
inline double draw_latent(const UcSkew skew, const double a, const double b, const double x)
{
    if (skew == UcSkew::snd)
    {
        // truncated_normal() takes a finite interval. Forty standard deviations
        // above the larger of the mean and zero leaves out less than 1e-300 of
        // the mass, so the draw is the half line's in all but name.
        const double s2 = a * a + b * b;
        const double mean = -b * x / s2;
        const double sd = a / std::sqrt(s2);
        return truncated_normal(mean, sd, 0.0, std::max(mean, 0.0) + 40.0 * sd);
    }
    const double mu = a - b;
    const double delta2 = 2.0 * a * b;
    return gig_hormann_leydold_2014(0.5, x * x / delta2, 2.0 + mu * mu / delta2);
}

/// The latent variate's mean, where the chain starts it: sqrt(2/pi) for the
/// half normal, one for the unit exponential.
inline double latent_mean(const UcSkew skew)
{
    return skew == UcSkew::snd ? std::sqrt(2.0 / arma::datum::pi) : 1.0;
}

} // namespace bayests::core

#endif // BAYESTS_CORE_MODELS_UC_SKEW_SUPPORT_H
