// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_CORE_MODELS_UC_SUPPORT_H
#define BAYESTS_CORE_MODELS_UC_SUPPORT_H

// The pieces of an unobserved components model that do not depend on how its
// shocks are distributed: the cycle's coefficients, the posterior precision of
// the unobserved path, the shocks as a regression on the coefficients, the
// Kalman filter the likelihood integrates the path out with, and the forecast
// recursion. They are here rather than in one model's source so that
// unit.uc_normal_gamma can put them against the dense Gaussian they factorise,
// and so that every member of the family reads the same ones.
//
// What the error block contributes is, per period, a precision of the 2k shocks
// and a mean: a constant precision and a zero mean for the gamma and Wishart
// models, a precision that moves for stochastic volatility, and both for the
// skewed shocks, whose normal mixture representation conditions each shock on
// a latent scale and a latent shift. Every function below takes the precision
// as one 2k x 2k matrix or as a stack of one per period, and the mean as a 2k x
// tt matrix or empty, so the constant case is the one it always was.
//
// With an irregular component the series is the trend plus the cycle plus
// measurement noise, the cycle is no longer y minus the trend, and the path
// drawn is the trend and the cycle together; see uc_state_band().

#include "bayests/priors.h"
#include "bayests/spec.h"
#include "core/algorithms/chan_jeliazkov_2009.h"
#include "core/models/model_support.h"
#include "core/models/shrinkage_support.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bayests::core
{

/// The k x kp cycle matrix [Phi_1 .. Phi_p] the free coefficients describe.
/// Under UcCycle::own the free ones are the diagonals, lag by lag, and the rest
/// is zero.
inline arma::mat cycle_matrix(const VarSpec &spec, const arma::vec &a)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    if (spec.cycle == UcCycle::var)
    {
        return arma::reshape(a, k, k * p);
    }
    arma::mat phi(k, k * p, arma::fill::zeros);
    for (arma::uword j = 0; j < p; j++)
    {
        for (arma::uword i = 0; i < k; i++)
        {
            phi(i, j * k + i) = a(j * k + i);
        }
    }
    return phi;
}

/// Where each free cycle coefficient sits in vec([Phi_1 .. Phi_p]): every
/// position under UcCycle::var, the diagonals alone under UcCycle::own.
inline arma::uvec cycle_positions(const VarSpec &spec)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    if (spec.cycle == UcCycle::var)
    {
        return arma::regspace<arma::uvec>(0, k * k * p - 1);
    }
    arma::uvec positions(k * p);
    for (arma::uword j = 0; j < p; j++)
    {
        for (arma::uword i = 0; i < k; i++)
        {
            positions(j * k + i) = (j * k + i) * k + i;
        }
    }
    return positions;
}

/// C d_t for every period, k x periods. Zero without deterministic terms.
inline arma::mat drift_path(const arma::mat &c, const arma::mat &d, const arma::uword k,
                            const arma::uword periods)
{
    if (c.n_elem == 0)
    {
        return arma::zeros<arma::mat>(k, periods);
    }
    return c * arma::trans(d);
}

/// The shock precision of period t: `q` itself when it is one 2k x 2k matrix,
/// its t-th block when it is a stack of one per period.
inline arma::mat shock_precision_at(const arma::mat &q, const arma::uword t)
{
    const arma::uword s = q.n_cols;
    if (q.n_rows == s)
    {
        return q;
    }
    return q.rows(t * s, (t + 1) * s - 1);
}

/// The trend's own lag polynomial, as the trend half of the M_l below: -I at
/// lag one for a random walk, -2I at lag one and I at lag two for a smooth
/// trend; I at lag zero.
inline double trend_lag_weight(const arma::uword order, const arma::uword lag)
{
    if (lag == 0)
    {
        return 1.0;
    }
    if (order == 1)
    {
        return lag == 1 ? -1.0 : 0.0;
    }
    return lag == 1 ? -2.0 : (lag == 2 ? 1.0 : 0.0);
}

/// The trend half of r_t: minus the drift, and the trend before the sample,
/// which given tau0 is a constant.
inline arma::vec trend_offset(const arma::vec &drift_t, const arma::vec &tau0, const arma::uword k,
                              const arma::uword order, const arma::uword t)
{
    arma::vec rt = -drift_t;
    const arma::vec tau_a = tau0.head(k);
    if (order == 1)
    {
        if (t == 0)
        {
            rt -= tau_a;
        }
    }
    else if (t == 0)
    {
        rt += -2.0 * tau_a + arma::vec(tau0.subvec(k, 2 * k - 1));
    }
    else if (t == 1)
    {
        rt += tau_a;
    }
    return rt;
}

/// The posterior precision of an unobserved path given everything else, as the
/// upper blocks of a band, and its right hand side.
struct UcTrendBand
{
    std::vector<arma::mat> blocks; ///< tt (band + 1) blocks, as chan_jeliazkov_2009_precision reads them.
    arma::mat rhs;                 ///< block size x tt.
    arma::uword band = 0;
};

/// The posterior precision of the k x tt trend path given everything else,
/// without an irregular component, where the cycle is y minus the trend.
///
/// Stack the 2k shocks of period t, e_t = (u^c_t', u^tau_t')'. Both halves are
/// linear in the trend at t and at most L = max(p, trend order) periods before
/// it,
///
///     e_t = r_t + sum_{l=0..L} M_l tau_{t-l},
///
/// with the terms before the sample moved into r_t: the cycle is zero there,
/// so a lag of it that reaches back past the sample contributes nothing at all,
/// and the trend is tau0 there, which is a constant given the rest. Each M_l is
/// 2k x k, the cycle's part on top -- M_0 = [-I; I], M_j = [Phi_j; 0] for the
/// cycle's lags -- and the trend's below: -I at l = 1 for a random walk, -2I at
/// l = 1 and I at l = 2 for a smooth trend.
///
/// The shocks are N(m_t, Q_t^-1), so the log density is
/// -sum_t (e_t - m_t)' Q_t (e_t - m_t) / 2, the trend's posterior precision
/// collects M_a' Q_t M_b in block (t-a, t-b) and its right hand side
/// -M_a' Q_t (r_t - m_t) in block t-a. That is block banded of bandwidth L
/// whatever Q_t correlates -- a trend shock correlated with the cycle shock of
/// the same period couples nothing further apart than the period itself -- and
/// it is the whole of the prior: the trend equation is a term of the joint
/// density like the cycle's, and needs no separate initialisation.
///
/// `q` is the shock precision, one 2k x 2k matrix or a stack of one per period;
/// `mean` the 2k x tt shock means, or empty for zero.
inline UcTrendBand uc_trend_band(const VarSpec &spec, const arma::mat &y, const arma::mat &phi,
                                 const arma::mat &drift, const arma::vec &tau0, const arma::mat &q,
                                 const arma::mat &mean = arma::mat())
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = y.n_cols;
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword band = std::max(p, order);
    const arma::mat eye_k = arma::eye<arma::mat>(k, k);
    const bool constant = q.n_rows == q.n_cols;
    const bool has_mean = mean.n_elem > 0;

    std::vector<arma::mat> m(band + 1, arma::zeros<arma::mat>(2 * k, k));
    m[0].rows(0, k - 1) = -eye_k;
    m[0].rows(k, 2 * k - 1) = eye_k;
    for (arma::uword j = 1; j <= p; j++)
    {
        m[j].rows(0, k - 1) = phi.cols((j - 1) * k, j * k - 1);
    }
    if (order == 1)
    {
        m[1].rows(k, 2 * k - 1) = -eye_k;
    }
    else
    {
        m[1].rows(k, 2 * k - 1) = -2.0 * eye_k;
        m[2].rows(k, 2 * k - 1) = eye_k;
    }

    // M_a' Q, the same in every period where Q is, and redone per period
    // where it is not.
    std::vector<arma::mat> mq(band + 1);
    const auto fill_mq = [&](const arma::mat &q_t) {
        for (arma::uword a = 0; a <= band; a++)
        {
            mq[a] = arma::trans(m[a]) * q_t;
        }
    };
    if (constant)
    {
        fill_mq(q);
    }

    std::vector<arma::mat> blocks(tt * (band + 1), arma::zeros<arma::mat>(k, k));
    arma::mat rhs(k, tt, arma::fill::zeros);
    arma::vec r(2 * k);

    for (arma::uword t = 0; t < tt; t++)
    {
        if (!constant)
        {
            fill_mq(shock_precision_at(q, t));
        }

        // The cycle half of r_t: the data part of c_t - sum_j Phi_j c_{t-j}.
        arma::vec ry = y.col(t);
        for (arma::uword j = 1; j <= p && j <= t; j++)
        {
            ry -= phi.cols((j - 1) * k, j * k - 1) * y.col(t - j);
        }
        r.head(k) = ry;
        r.tail(k) = trend_offset(drift.col(t), tau0, k, order, t);
        if (has_mean)
        {
            r -= mean.col(t);
        }

        for (arma::uword a = 0; a <= band && a <= t; a++)
        {
            rhs.col(t - a) -= mq[a] * r;
            for (arma::uword b = 0; b <= a; b++)
            {
                blocks[(t - a) * (band + 1) + (a - b)] += mq[a] * m[b];
            }
        }
    }

    UcTrendBand out;
    out.blocks = std::move(blocks);
    out.rhs = std::move(rhs);
    out.band = band;
    return out;
}

/// Draws the k x tt trend path given everything else; see uc_trend_band().
inline arma::mat draw_uc_trend(const VarSpec &spec, const arma::mat &y, const arma::mat &phi,
                               const arma::mat &drift, const arma::vec &tau0, const arma::mat &q,
                               const arma::mat &mean = arma::mat())
{
    UcTrendBand sys = uc_trend_band(spec, y, phi, drift, tau0, q, mean);
    return chan_jeliazkov_2009_precision(std::move(sys.blocks), sys.rhs, sys.band);
}

/// The posterior precision of the trend and the cycle together, x_t = (tau_t',
/// c_t')', 2k x tt, where an irregular component separates the two from the
/// series: y_t = tau_t + c_t + eps_t, eps_t ~ N(0, diag(irregular_inv)^-1).
///
/// The shocks are linear in x as they are in the trend alone above, with the
/// cycle now a state rather than y minus the trend: M_0 = [0 I; I 0], the cycle
/// lags in the top right -- [0 -Phi_j] -- and the trend's lag polynomial in the
/// bottom left, and only the trend before the sample left in r_t. The
/// measurement adds G' H G to the diagonal block of every period and G' H y_t to
/// its right hand side, G = [I I] and H the irregular precision. The bandwidth
/// is the same, max(p, trend order), in blocks of twice the width.
inline UcTrendBand uc_state_band(const VarSpec &spec, const arma::mat &y, const arma::mat &phi,
                                 const arma::mat &drift, const arma::vec &tau0, const arma::mat &q,
                                 const arma::vec &irregular_inv, const arma::mat &mean = arma::mat())
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = y.n_cols;
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword band = std::max(p, order);
    const arma::uword w = 2 * k;
    const arma::mat eye_k = arma::eye<arma::mat>(k, k);
    const bool constant = q.n_rows == q.n_cols;
    const bool has_mean = mean.n_elem > 0;

    // Rows: the cycle shock, then the trend shock. Columns: tau, then c.
    std::vector<arma::mat> m(band + 1, arma::zeros<arma::mat>(w, w));
    m[0].submat(0, k, k - 1, w - 1) = eye_k;
    for (arma::uword j = 1; j <= p; j++)
    {
        m[j].submat(0, k, k - 1, w - 1) = -phi.cols((j - 1) * k, j * k - 1);
    }
    for (arma::uword l = 0; l <= order; l++)
    {
        m[l].submat(k, 0, w - 1, k - 1) = trend_lag_weight(order, l) * eye_k;
    }

    std::vector<arma::mat> mq(band + 1);
    const auto fill_mq = [&](const arma::mat &q_t) {
        for (arma::uword a = 0; a <= band; a++)
        {
            mq[a] = arma::trans(m[a]) * q_t;
        }
    };
    if (constant)
    {
        fill_mq(q);
    }

    // G' H G and G' H, G = [I I].
    arma::mat g(k, w);
    g.cols(0, k - 1) = eye_k;
    g.cols(k, w - 1) = eye_k;
    const arma::mat gh = arma::trans(g) * arma::diagmat(irregular_inv);
    const arma::mat ghg = gh * g;

    std::vector<arma::mat> blocks(tt * (band + 1), arma::zeros<arma::mat>(w, w));
    arma::mat rhs(w, tt, arma::fill::zeros);
    arma::vec r(w, arma::fill::zeros);

    for (arma::uword t = 0; t < tt; t++)
    {
        if (!constant)
        {
            fill_mq(shock_precision_at(q, t));
        }

        r.head(k).zeros();
        r.tail(k) = trend_offset(drift.col(t), tau0, k, order, t);
        if (has_mean)
        {
            r -= mean.col(t);
        }

        blocks[t * (band + 1)] += ghg;
        rhs.col(t) += gh * y.col(t);

        for (arma::uword a = 0; a <= band && a <= t; a++)
        {
            rhs.col(t - a) -= mq[a] * r;
            for (arma::uword b = 0; b <= a; b++)
            {
                blocks[(t - a) * (band + 1) + (a - b)] += mq[a] * m[b];
            }
        }
    }

    UcTrendBand out;
    out.blocks = std::move(blocks);
    out.rhs = std::move(rhs);
    out.band = band;
    return out;
}

/// The trend and the cycle, each k x tt, of one draw of the unobserved path.
struct UcPath
{
    arma::mat trend;
    arma::mat cycle;
};

/// Draws the unobserved path given everything else: the trend alone, the cycle
/// following as y minus it, where there is no irregular component -- the
/// model's `irregular_inv` empty -- and the two together where there is.
inline UcPath draw_uc_path(const VarSpec &spec, const arma::mat &y, const arma::mat &phi,
                           const arma::mat &drift, const arma::vec &tau0, const arma::mat &q,
                           const arma::vec &irregular_inv, const arma::mat &mean = arma::mat())
{
    UcPath out;
    if (irregular_inv.n_elem == 0)
    {
        out.trend = draw_uc_trend(spec, y, phi, drift, tau0, q, mean);
        out.cycle = y - out.trend;
        return out;
    }
    const arma::uword k = static_cast<arma::uword>(spec.k);
    UcTrendBand sys = uc_state_band(spec, y, phi, drift, tau0, q, irregular_inv, mean);
    const arma::mat x = chan_jeliazkov_2009_precision(std::move(sys.blocks), sys.rhs, sys.band);
    out.trend = x.rows(0, k - 1);
    out.cycle = x.rows(k, 2 * k - 1);
    return out;
}

/// The 2k shocks of every period as linear in the coefficients: for period t,
/// e_t = w.col(t) - x.slice(t) * theta, with theta the free cycle coefficients,
/// vec(C) and tau0, in that order.
struct ShockRegression
{
    arma::mat w;  ///< 2k x tt.
    arma::cube x; ///< 2k x n_theta x tt.
};

/// The same, given the trend and the cycle as separate paths -- which they are
/// with an irregular component, the cycle then being drawn rather than y minus
/// the trend.
inline ShockRegression shock_regression_paths(const VarSpec &spec, const arma::mat &tau,
                                              const arma::mat &cycle, const arma::mat &d,
                                              const arma::uvec &positions)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = tau.n_cols;
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword n = static_cast<arma::uword>(spec.n);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword n_a = positions.n_elem;
    const arma::uword n_c = k * n;
    const arma::uword n_theta = n_a + n_c + k * order;
    const arma::mat eye_k = arma::eye<arma::mat>(k, k);

    ShockRegression out;
    out.w = arma::mat(2 * k, tt);
    out.x = arma::cube(2 * k, n_theta, tt, arma::fill::zeros);

    for (arma::uword t = 0; t < tt; t++)
    {
        arma::mat x_t(2 * k, n_theta, arma::fill::zeros);

        // The cycle: c_t on its p lags, (x_t' kron I_k) vec(Phi), zero before
        // the sample.
        out.w.col(t).head(k) = cycle.col(t);
        if (n_a > 0)
        {
            arma::vec lags(k * p, arma::fill::zeros);
            for (arma::uword j = 1; j <= p && j <= t; j++)
            {
                lags.subvec((j - 1) * k, j * k - 1) = cycle.col(t - j);
            }
            const arma::mat full = arma::kron(arma::trans(lags), eye_k);
            x_t.submat(0, 0, k - 1, n_a - 1) = full.cols(positions);
        }

        // The trend: its difference on the drift and, near the start, on the
        // trend before the sample.
        arma::vec diff = tau.col(t);
        if (order == 1)
        {
            if (t >= 1)
            {
                diff -= tau.col(t - 1);
            }
        }
        else
        {
            if (t >= 1)
            {
                diff -= 2.0 * tau.col(t - 1);
            }
            if (t >= 2)
            {
                diff += tau.col(t - 2);
            }
        }
        out.w.col(t).tail(k) = diff;

        if (n_c > 0)
        {
            x_t.submat(k, n_a, 2 * k - 1, n_a + n_c - 1) = arma::kron(d.row(t), eye_k);
        }
        const arma::uword at = n_a + n_c;
        if (order == 1)
        {
            if (t == 0)
            {
                x_t.submat(k, at, 2 * k - 1, at + k - 1) = eye_k;
            }
        }
        else if (t == 0)
        {
            x_t.submat(k, at, 2 * k - 1, at + k - 1) = 2.0 * eye_k;
            x_t.submat(k, at + k, 2 * k - 1, at + 2 * k - 1) = -eye_k;
        }
        else if (t == 1)
        {
            x_t.submat(k, at, 2 * k - 1, at + k - 1) = -eye_k;
        }

        out.x.slice(t) = x_t;
    }

    return out;
}

/// Without an irregular component, the cycle is y minus the trend.
inline ShockRegression shock_regression(const VarSpec &spec, const arma::mat &y,
                                        const arma::mat &tau, const arma::mat &d,
                                        const arma::uvec &positions)
{
    return shock_regression_paths(spec, tau, y - tau, d, positions);
}

/// Adds the shocks' likelihood of theta to its prior: sum_t X_t' Q_t X_t to the
/// precision and sum_t X_t' Q_t (w_t - m_t) to the right hand side, with Q the
/// shock precision -- one matrix, or a stack of one per period -- and m the
/// shock means, or empty for zero.
inline void add_shock_likelihood(const ShockRegression &reg, const arma::mat &q,
                                 const arma::mat &mean, arma::mat &post_vinv, arma::vec &post_rhs)
{
    const arma::uword tt = reg.w.n_cols;
    const bool has_mean = mean.n_elem > 0;
    for (arma::uword t = 0; t < tt; t++)
    {
        const arma::mat xq = arma::trans(reg.x.slice(t)) * shock_precision_at(q, t);
        post_vinv += xq * reg.x.slice(t);
        if (has_mean)
        {
            post_rhs += xq * (reg.w.col(t) - mean.col(t));
        }
        else
        {
            post_rhs += xq * reg.w.col(t);
        }
    }
}

/// The innovation structure the Kalman filter below runs the model in, given
/// one draw's parameters.
///
///     s_t = T s_{t-1} + (C d_t, 0)' + R (e_t, with mean m_t),
///     y_t = Z s_t + eps_t,
///
/// with s_t = (tau_t', [tau_{t-1}'], c_t', .., c_{t-q+1}')' and q = max(p, 1),
/// the lagged trend only for a smooth one. Without an irregular component there
/// is no measurement error -- the series is the trend plus the cycle exactly,
/// and what makes F invertible is that both carry a shock every period.
struct UcStateSpace
{
    arma::mat transition; ///< T, m x m.
    arma::mat state_cov;  ///< R Sigma R', m x m, for a shock covariance that does not move.
    arma::mat r;          ///< R, m x 2k: the cycle shock into c_t, the trend shock into tau_t.
    arma::mat z;          ///< k x m.
    arma::uword m = 0;
    arma::uword k = 0;
};

inline UcStateSpace state_space(const VarSpec &spec, const arma::mat &phi, const arma::mat &sigma)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword q = std::max<arma::uword>(p, 1);
    const arma::uword cyc = k * order; // where the cycle block starts
    const arma::mat eye_k = arma::eye<arma::mat>(k, k);

    UcStateSpace ss;
    ss.k = k;
    ss.m = k * order + k * q;
    ss.transition = arma::zeros<arma::mat>(ss.m, ss.m);
    if (order == 1)
    {
        ss.transition.submat(0, 0, k - 1, k - 1) = eye_k;
    }
    else
    {
        ss.transition.submat(0, 0, k - 1, k - 1) = 2.0 * eye_k;
        ss.transition.submat(0, k, k - 1, 2 * k - 1) = -eye_k;
        ss.transition.submat(k, 0, 2 * k - 1, k - 1) = eye_k;
    }
    if (p > 0)
    {
        ss.transition.submat(cyc, cyc, cyc + k - 1, cyc + k * p - 1) = phi;
    }
    for (arma::uword j = 1; j < q; j++)
    {
        ss.transition.submat(cyc + j * k, cyc + (j - 1) * k, cyc + (j + 1) * k - 1,
                             cyc + j * k - 1) = eye_k;
    }

    ss.r = arma::zeros<arma::mat>(ss.m, 2 * k);
    ss.r.submat(cyc, 0, cyc + k - 1, k - 1) = eye_k;   // u^c into c_t
    ss.r.submat(0, k, k - 1, 2 * k - 1) = eye_k;       // u^tau into tau_t
    if (sigma.n_elem > 0)
    {
        ss.state_cov = ss.r * sigma * arma::trans(ss.r);
    }

    ss.z = arma::zeros<arma::mat>(k, ss.m);
    ss.z.cols(0, k - 1) = eye_k;
    ss.z.cols(cyc, cyc + k - 1) = eye_k;
    return ss;
}

/// What moves from period to period in the filter beyond the drift: the shock
/// covariance -- `sigma_path`, one 2k x 2k matrix per period stacked, or empty
/// where ss.state_cov holds for every period -- the shock means, 2k x periods or
/// empty, and the irregular component's variances, k or empty.
struct UcFilterTerms
{
    arma::mat sigma_path;
    arma::mat shock_mean;
    arma::vec irregular_var;
};

/// One step ahead log densities of y.col(0) .. y.col(periods - 1), started from
/// the trend before the sample at tau0 and the cycle at zero, both known: given
/// one draw of everything else, the start is not uncertain, and P_0 is zero.
inline arma::rowvec filter_log_density(const UcStateSpace &ss, const arma::mat &y,
                                       const arma::mat &drift, const arma::vec &tau0,
                                       const UcFilterTerms &terms = UcFilterTerms())
{
    const arma::uword k = ss.k;
    const arma::uword periods = y.n_cols;
    const double constant = -static_cast<double>(k) * std::log(2.0 * arma::datum::pi) / 2.0;
    const bool moving = terms.sigma_path.n_elem > 0;
    const bool has_mean = terms.shock_mean.n_elem > 0;
    const bool irregular = terms.irregular_var.n_elem > 0;
    const arma::mat noise = irregular ? arma::mat(arma::diagmat(terms.irregular_var)) : arma::mat();
    const arma::mat rt = arma::trans(ss.r);

    arma::vec s(ss.m, arma::fill::zeros);
    s.head(tau0.n_elem) = tau0;
    arma::mat p_cov(ss.m, ss.m, arma::fill::zeros);
    const arma::mat zt = arma::trans(ss.z);

    arma::rowvec out(periods);
    for (arma::uword t = 0; t < periods; t++)
    {
        arma::vec a = ss.transition * s;
        a.head(k) += drift.col(t);
        if (has_mean)
        {
            a += ss.r * terms.shock_mean.col(t);
        }
        const arma::mat state_cov =
            moving ? arma::mat(ss.r * shock_precision_at(terms.sigma_path, t) * rt) : ss.state_cov;
        const arma::mat pred = ss.transition * p_cov * arma::trans(ss.transition) + state_cov;

        const arma::vec v = y.col(t) - ss.z * a;
        const arma::mat pz = pred * zt;
        const arma::mat f = irregular ? arma::mat(arma::symmatu(ss.z * pz + noise))
                                      : arma::mat(arma::symmatu(ss.z * pz));

        arma::mat root;
        if (!arma::chol(root, f))
        {
            throw std::runtime_error("unobserved components: the one step prediction variance is "
                                     "not positive definite at period " + std::to_string(t + 1));
        }
        const arma::vec scaled = arma::solve(arma::trimatl(arma::trans(root)), v);
        out(t) = constant - arma::accu(arma::log(root.diag())) - arma::dot(scaled, scaled) / 2.0;

        // Update: K = P Z' F^-1, through the factor rather than an inverse.
        const arma::mat gain = arma::trans(
            arma::solve(arma::trimatu(root), arma::solve(arma::trimatl(arma::trans(root)),
                                                         arma::trans(pz))));
        s = a + gain * v;
        p_cov = arma::symmatu(pred - gain * arma::trans(pz));
    }
    return out;
}

/// Simulates one forecast path, h x k stacked by horizon, from the end of one
/// draw's trend and cycle paths.
///
/// `shock(step)` returns the 2k shocks of a horizon, cycle shocks first, and is
/// where the error block's distribution enters; `irregular_sd`, k or empty,
/// scales a standard normal added to the series at every horizon. The order
/// random numbers are taken in is the shocks of a horizon, then its irregular
/// component.
template <typename Shock>
arma::vec uc_forecast_path(const VarSpec &spec, const arma::mat &tau, const arma::mat &cycle,
                           const arma::mat &phi, const arma::mat &c, const arma::mat &forecast_x,
                           const arma::vec &irregular_sd, Shock shock)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword h = static_cast<arma::uword>(spec.h);
    const arma::uword tt = tau.n_cols;
    const bool smooth = spec.trend == UcTrend::smooth;

    // The cycle's last p values, most recent first, and the trend's last one
    // or two.
    arma::mat cycle_lags(k, p, arma::fill::zeros);
    for (arma::uword j = 0; j < p && j < tt; j++)
    {
        cycle_lags.col(j) = cycle.col(tt - 1 - j);
    }
    arma::vec tau_last = tau.col(tt - 1);
    arma::vec tau_prev = smooth ? arma::vec(tau.col(tt - 2)) : arma::vec();

    arma::vec out(h * k);
    for (arma::uword step = 0; step < h; step++)
    {
        const arma::vec e = shock(step);

        arma::vec cyc = e.head(k);
        for (arma::uword j = 0; j < p; j++)
        {
            cyc += phi.cols(j * k, (j + 1) * k - 1) * cycle_lags.col(j);
        }

        arma::vec next = smooth ? arma::vec(2.0 * tau_last - tau_prev) : tau_last;
        next += e.tail(k);
        if (spec.n > 0)
        {
            next += c * arma::trans(forecast_x.row(step));
        }

        arma::vec level = next + cyc;
        if (irregular_sd.n_elem > 0)
        {
            level += irregular_sd % arma::randn<arma::vec>(k);
        }
        out.subvec(step * k, (step + 1) * k - 1) = level;

        if (p > 0)
        {
            if (p > 1)
            {
                cycle_lags.cols(1, p - 1) = arma::mat(cycle_lags.cols(0, p - 2));
            }
            cycle_lags.col(0) = cyc;
        }
        if (smooth)
        {
            tau_prev = tau_last;
        }
        tau_last = next;
    }
    return out;
}

/// The coefficients every unobserved components model draws jointly given the
/// unobserved path: the free cycle coefficients, vec(C) and tau0, in that
/// order, under a block diagonal normal prior. The error blocks differ between
/// the models and this does not, so it is one object rather than four copies.
class UcCoefficients
{
public:
    UcCoefficients(const VarSpec &spec, const NormalPrior &a_prior, const arma::vec &a_init,
                   const NormalPrior &c_prior, const arma::vec &c_init,
                   const NormalPrior &tau0_prior, const arma::vec &tau0_init)
        : spec_(spec)
    {
        k_ = static_cast<arma::uword>(spec.k);
        p_ = static_cast<arma::uword>(spec.p);
        n_ = static_cast<arma::uword>(spec.n);
        use_a_ = spec.p > 0;
        use_c_ = spec.n > 0;
        positions_ = use_a_ ? cycle_positions(spec) : arma::uvec();
        n_a_ = positions_.n_elem;
        n_c_ = use_c_ ? k_ * n_ : 0;
        n_tau0_ = k_ * static_cast<arma::uword>(spec.uc_trend_order());
        const arma::uword n_theta = n_a_ + n_c_ + n_tau0_;

        prior_vinv_ = arma::zeros<arma::mat>(n_theta, n_theta);
        prior_rhs_ = arma::zeros<arma::vec>(n_theta);
        theta_ = arma::vec(n_theta);
        if (use_a_)
        {
            prior_vinv_.submat(0, 0, n_a_ - 1, n_a_ - 1) = a_prior.v_inv;
            prior_rhs_.head(n_a_) = a_prior.v_inv * a_prior.mu;
            theta_.head(n_a_) = a_init;
        }
        if (use_c_)
        {
            prior_vinv_.submat(n_a_, n_a_, n_a_ + n_c_ - 1, n_a_ + n_c_ - 1) = c_prior.v_inv;
            prior_rhs_.subvec(n_a_, n_a_ + n_c_ - 1) = c_prior.v_inv * c_prior.mu;
            theta_.subvec(n_a_, n_a_ + n_c_ - 1) = c_init;
        }
        prior_vinv_.submat(n_a_ + n_c_, n_a_ + n_c_, n_theta - 1, n_theta - 1) = tau0_prior.v_inv;
        prior_rhs_.tail(n_tau0_) = tau0_prior.v_inv * tau0_prior.mu;
        theta_.tail(n_tau0_) = tau0_init;
    }

    const arma::uvec &positions() const { return positions_; }
    bool use_a() const { return use_a_; }
    bool use_c() const { return use_c_; }

    /// The k x kp cycle matrix, k x 0 without lags.
    arma::mat phi() const { return phi_of(theta_); }
    /// The k x n drift matrix, empty without deterministic terms.
    arma::mat c() const
    {
        return use_c_ ? arma::mat(arma::reshape(theta_.subvec(n_a_, n_a_ + n_c_ - 1), k_, n_))
                      : arma::mat();
    }
    arma::vec c_vec() const { return theta_.subvec(n_a_, n_a_ + n_c_ - 1); }
    arma::vec tau0() const { return theta_.tail(n_tau0_); }
    /// The whole of theta, as ShockRegression::x multiplies it.
    const arma::vec &theta() const { return theta_; }

    /// Draws theta given the shocks' regression on it, under a shock precision
    /// q -- one matrix or a stack of one per period -- and shock means `mean`,
    /// or empty for zero. Under VarSpec::stationary the draw_stationary() rule
    /// over the joint block: up to kStationaryTries proposals, and the previous
    /// draw kept if none has a stationary cycle, in which case it returns false.
    bool draw(const ShockRegression &reg, const arma::mat &q, const arma::mat &mean)
    {
        arma::mat post_vinv = prior_vinv_;
        arma::vec post_rhs = prior_rhs_;
        add_shock_likelihood(reg, q, mean, post_vinv, post_rhs);
        if (spec_.stationary && use_a_)
        {
            for (int attempt = 0; attempt < kStationaryTries; attempt++)
            {
                const arma::vec proposal = draw_normal_precision(post_vinv, post_rhs);
                if (is_stationary(arma::vectorise(phi_of(proposal)), k_, p_))
                {
                    theta_ = proposal;
                    return true;
                }
            }
            return false;
        }
        theta_ = draw_normal_precision(post_vinv, post_rhs);
        return true;
    }

private:
    arma::mat phi_of(const arma::vec &th) const
    {
        return use_a_ ? cycle_matrix(spec_, th.head(n_a_)) : arma::mat(k_, 0);
    }

    const VarSpec &spec_;
    arma::uword k_ = 0, p_ = 0, n_ = 0, n_a_ = 0, n_c_ = 0, n_tau0_ = 0;
    bool use_a_ = false, use_c_ = false;
    arma::uvec positions_;
    arma::mat prior_vinv_;
    arma::vec prior_rhs_;
    arma::vec theta_;
};

/// The irregular component's precisions given the path: independent gamma
/// posteriors on y - tau - c, one per series.
inline arma::vec draw_irregular_precision(const GammaPrior &prior, const arma::mat &y,
                                          const UcPath &path)
{
    const arma::mat noise = y - path.trend - path.cycle;
    const arma::vec sse = arma::sum(arma::square(noise), 1);
    const double half = static_cast<double>(noise.n_cols) * 0.5;
    arma::vec out(noise.n_rows);
    for (arma::uword i = 0; i < noise.n_rows; i++)
    {
        out(i) = arma::randg<double>(
            arma::distr_param(prior.shape(i) + half, 1.0 / (prior.rate(i) + sse(i) * 0.5)));
    }
    return out;
}

/// The covariance a drawn shock precision stands for.
inline arma::mat uc_covariance_of_precision(const arma::mat &q)
{
    arma::mat sigma;
    if (!arma::inv_sympd(sigma, arma::symmatu(q)))
    {
        throw std::runtime_error("unobserved components: a drawn shock precision is not "
                                 "symmetric positive definite");
    }
    return sigma;
}

/// One draw's coefficients in the shapes the filter and the forecast read.
struct UcDrawParts
{
    arma::mat phi;  ///< k x kp, k x 0 without lags.
    arma::mat c;    ///< k x n, empty without deterministic terms.
    arma::vec tau0;
    arma::vec irregular_var; ///< k, empty without an irregular component.
};

/// Draw i of the coefficient datasets every member of the family writes.
inline UcDrawParts uc_draw_parts(const VarSpec &spec, const arma::mat &a, const arma::mat &c,
                                 const arma::mat &tau0, const arma::mat &irregular_inv,
                                 const arma::uword i)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    UcDrawParts out;
    out.phi = spec.p > 0
                  ? arma::mat(arma::reshape(a.col(i), k, k * static_cast<arma::uword>(spec.p)))
                  : arma::mat(k, 0);
    if (spec.n > 0)
    {
        out.c = arma::reshape(c.col(i), k, static_cast<arma::uword>(spec.n));
    }
    out.tau0 = tau0.col(i);
    if (spec.irregular)
    {
        out.irregular_var = 1.0 / irregular_inv.col(i);
    }
    return out;
}

/// Refuses posterior draws of the shared datasets that do not have the heights
/// the spec implies, naming the one that is wrong. `tt` is the sample's length.
inline void require_uc_coefficient_draws(const VarSpec &spec, const arma::mat &a,
                                         const arma::mat &c, const arma::mat &tau0,
                                         const arma::mat &irregular_inv, const arma::mat &trend,
                                         const arma::mat &cycle, const arma::uword tt)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const auto require = [](const bool ok, const std::string &what) {
        if (!ok)
        {
            throw std::invalid_argument(what);
        }
    };
    require(tau0.n_cols > 0 && tau0.n_rows == k * static_cast<arma::uword>(spec.uc_trend_order()),
            "posterior draws of tau0 must have " + std::to_string(k * spec.uc_trend_order()) +
                " rows, got " + std::to_string(tau0.n_rows));
    require(spec.p == 0 || a.n_rows == k * k * static_cast<arma::uword>(spec.p),
            "posterior draws of the cycle coefficients a must have " +
                std::to_string(k * k * spec.p) + " rows, got " + std::to_string(a.n_rows));
    require(spec.n == 0 || c.n_rows == k * static_cast<arma::uword>(spec.n),
            "posterior draws of the drift c must have " + std::to_string(k * spec.n) +
                " rows, got " + std::to_string(c.n_rows));
    require(!spec.irregular || irregular_inv.n_rows == k,
            "posterior draws of the irregular precision must have " + std::to_string(k) +
                " rows, got " + std::to_string(irregular_inv.n_rows));
    require(trend.n_elem == 0 || trend.n_rows == k * tt,
            "posterior draws of the trend path must have " + std::to_string(k * tt) +
                " rows, k per period of the sample, got " + std::to_string(trend.n_rows));
    require(!spec.irregular || cycle.n_elem == 0 || cycle.n_rows == k * tt,
            "posterior draws of the cycle path must have " + std::to_string(k * tt) +
                " rows, got " + std::to_string(cycle.n_rows));
}

/// The k x tt cycle of draw i: stored where there is an irregular component, y
/// minus the trend where there is not.
inline arma::mat uc_cycle_at(const VarSpec &spec, const arma::mat &y, const arma::mat &tau,
                             const arma::mat &cycle_draws, const arma::uword i)
{
    if (spec.irregular)
    {
        return arma::reshape(cycle_draws.col(i), y.n_rows, y.n_cols);
    }
    return y - tau;
}

} // namespace bayests::core

#endif // BAYESTS_CORE_MODELS_UC_SUPPORT_H
