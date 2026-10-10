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
//
// Two refinements a model may add arrive as one UcExtension, empty by default:
// the damped trend's slope coefficients, which change the trend's lag
// polynomial, and the signal equations, further observed series that load on
// the cycle and so add a measurement to the band and rows to the filter.
// Default constructed it changes nothing, number for number.

#include "bayests/priors.h"
#include "bayests/reporter.h"
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

/// The signal equations of one draw, as the band and the filter read them:
///
///     s_t = B x^s_t + sum_{j=0..q} Lambda_j c_{t-j} + v_t,  v_t ~ N(0, H^-1),
///
/// with H = diag(precision) and the cycle zero before the sample, as it is for
/// the cycle's own lags.
struct UcSignalTerms
{
    arma::mat offset;     ///< m x tt: s_t - B x^s_t, the part the cycle does not explain.
    arma::mat lambda;     ///< m x k(q + 1): [Lambda_0 .. Lambda_q].
    arma::vec precision;  ///< m, the diagonal of H.
    arma::uword lags = 0; ///< q.

    bool active() const { return precision.n_elem > 0; }

    /// Lambda_j, m x k.
    arma::mat loading(const arma::uword j, const arma::uword k) const
    {
        return lambda.cols(j * k, (j + 1) * k - 1);
    }
};

/// What a model may add to the trend and the cycle beyond its shocks: the
/// damped trend's slope coefficients and the signal equations. Default
/// constructed, neither.
struct UcExtension
{
    arma::vec rho; ///< k under UcTrend::damped, empty otherwise.
    UcSignalTerms signal;
};

/// The trend's own lag polynomial, k x (order + 1): column l is the weight
/// each series' trend shock puts on tau_{t-l}. One at lag zero; -1 at lag one
/// for a random walk; -2 and 1 at lags one and two for a smooth trend, and
/// -(1 + rho) and rho for a damped one.
inline arma::mat trend_lag_weights(const VarSpec &spec, const arma::vec &rho)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    arma::mat w(k, order + 1);
    w.col(0).ones();
    if (order == 1)
    {
        w.col(1).fill(-1.0);
        return w;
    }
    if (spec.trend == UcTrend::damped)
    {
        if (rho.n_elem != k)
        {
            throw std::invalid_argument("a damped trend needs one slope coefficient rho per "
                                        "series, " + std::to_string(k) + ", got " +
                                        std::to_string(rho.n_elem));
        }
        w.col(1) = -(1.0 + rho);
        w.col(2) = rho;
        return w;
    }
    w.col(1).fill(-2.0);
    w.col(2).fill(1.0);
    return w;
}

/// How far back the signals reach into the cycle: q with signals, zero
/// without.
inline arma::uword signal_reach(const UcExtension &ext)
{
    return ext.signal.active() ? ext.signal.lags : 0;
}

/// The trend half of r_t: minus the drift, and the trend before the sample,
/// which given tau0 is a constant. `weights` is trend_lag_weights().
inline arma::vec trend_offset(const arma::vec &drift_t, const arma::vec &tau0, const arma::uword k,
                              const arma::mat &weights, const arma::uword t)
{
    const arma::uword order = weights.n_cols - 1;
    arma::vec rt = -drift_t;
    const arma::vec tau_a = tau0.head(k);
    if (order == 1)
    {
        if (t == 0)
        {
            rt += weights.col(1) % tau_a;
        }
    }
    else if (t == 0)
    {
        rt += weights.col(1) % tau_a + weights.col(2) % arma::vec(tau0.subvec(k, 2 * k - 1));
    }
    else if (t == 1)
    {
        rt += weights.col(2) % tau_a;
    }
    return rt;
}

/// Adds the signal equations' measurement to a band. With the signal noise
/// v_t = r_t + sum_l N_l x_{t-l}, N_l the m x width loading of x_{t-l}, it
/// collects N_a' H N_b in block (t-a, t-b) and -N_a' H r_t in the right hand
/// side of t-a, exactly as the shocks do. `r` is m x tt.
inline void add_signal_measurement(const std::vector<arma::mat> &n, const arma::mat &r,
                                   const arma::vec &precision, const arma::uword band,
                                   std::vector<arma::mat> &blocks, arma::mat &rhs)
{
    const arma::uword tt = r.n_cols;
    const arma::uword reach = n.size() - 1;
    const arma::mat h = arma::diagmat(precision);
    std::vector<arma::mat> nh(reach + 1);
    for (arma::uword a = 0; a <= reach; a++)
    {
        nh[a] = arma::trans(n[a]) * h;
    }
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword a = 0; a <= reach && a <= t; a++)
        {
            rhs.col(t - a) -= nh[a] * r.col(t);
            for (arma::uword b = 0; b <= a; b++)
            {
                blocks[(t - a) * (band + 1) + (a - b)] += nh[a] * n[b];
            }
        }
    }
}

/// sum_{j > t} Phi_j c_{t-j}: the part of period t's cycle equation that
/// reaches back before the sample, k long. `cycle_pre` stacks the cycle before
/// it, (c_0', c_-1', .., c_1-p')'; empty means zero, and so does a t past p.
inline arma::vec cycle_presample_term(const arma::mat &phi, const arma::vec &cycle_pre,
                                      const arma::uword k, const arma::uword p, const arma::uword t)
{
    arma::vec out(k, arma::fill::zeros);
    if (cycle_pre.n_elem == 0)
    {
        return out;
    }
    for (arma::uword j = t + 1; j <= p; j++)
    {
        const arma::uword i = j - t - 1;
        out += phi.cols((j - 1) * k, j * k - 1) * cycle_pre.subvec(i * k, (i + 1) * k - 1);
    }
    return out;
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
/// `mean` the 2k x tt shock means, or empty for zero; `cycle_pre` the cycle
/// before the sample, kp stacked (c_0', c_-1', .., c_1-p')' -- see
/// UcCycleInit -- or empty for zero, which is the r_t above.
///
/// `ext` adds what UcExtension carries. A damped trend changes the trend's lag
/// polynomial and nothing else. Signals add their measurement: v_t is linear in
/// the trend at t .. t-q once the cycle is y minus it, so the bandwidth becomes
/// max(p, trend order, q). They read the cycle before the sample as zero, and
/// are not combined with `cycle_pre` (validate() refuses the pair).
inline UcTrendBand uc_trend_band(const VarSpec &spec, const arma::mat &y, const arma::mat &phi,
                                 const arma::mat &drift, const arma::vec &tau0, const arma::mat &q,
                                 const arma::mat &mean = arma::mat(),
                                 const arma::vec &cycle_pre = arma::vec(),
                                 const UcExtension &ext = UcExtension())
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = y.n_cols;
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword band = std::max({p, order, signal_reach(ext)});
    const arma::mat eye_k = arma::eye<arma::mat>(k, k);
    const arma::mat weights = trend_lag_weights(spec, ext.rho);
    const bool constant = q.n_rows == q.n_cols;
    const bool has_mean = mean.n_elem > 0;

    std::vector<arma::mat> m(band + 1, arma::zeros<arma::mat>(2 * k, k));
    m[0].rows(0, k - 1) = -eye_k;
    m[0].rows(k, 2 * k - 1) = eye_k;
    for (arma::uword j = 1; j <= p; j++)
    {
        m[j].rows(0, k - 1) = phi.cols((j - 1) * k, j * k - 1);
    }
    for (arma::uword l = 1; l <= order; l++)
    {
        m[l].rows(k, 2 * k - 1) = arma::diagmat(weights.col(l));
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

        // The cycle half of r_t: the data part of c_t - sum_j Phi_j c_{t-j},
        // and the lags that reach back past the sample, which are constants.
        arma::vec ry = y.col(t);
        for (arma::uword j = 1; j <= p && j <= t; j++)
        {
            ry -= phi.cols((j - 1) * k, j * k - 1) * y.col(t - j);
        }
        ry -= cycle_presample_term(phi, cycle_pre, k, p, t);
        r.head(k) = ry;
        r.tail(k) = trend_offset(drift.col(t), tau0, k, weights, t);
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

    // The signals: v_t = s_t - B x^s_t - sum_j Lambda_j (y_{t-j} - tau_{t-j}),
    // so N_l = Lambda_l, and r_t is the offset less the loadings on y.
    if (ext.signal.active())
    {
        const UcSignalTerms &sig = ext.signal;
        std::vector<arma::mat> n(sig.lags + 1);
        for (arma::uword j = 0; j <= sig.lags; j++)
        {
            n[j] = sig.loading(j, k);
        }
        arma::mat rs = sig.offset;
        for (arma::uword t = 0; t < tt; t++)
        {
            for (arma::uword j = 0; j <= sig.lags && j <= t; j++)
            {
                rs.col(t) -= n[j] * y.col(t - j);
            }
        }
        add_signal_measurement(n, rs, sig.precision, band, blocks, rhs);
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
                               const arma::mat &mean = arma::mat(),
                               const arma::vec &cycle_pre = arma::vec(),
                               const UcExtension &ext = UcExtension())
{
    UcTrendBand sys = uc_trend_band(spec, y, phi, drift, tau0, q, mean, cycle_pre, ext);
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
/// is the same, max(p, trend order), in blocks of twice the width -- or q,
/// the signals' reach, where that is longer: a signal is a measurement of the
/// cycle at t .. t-q, with N_l = [0 -Lambda_l] and r_t = s_t - B x^s_t.
inline UcTrendBand uc_state_band(const VarSpec &spec, const arma::mat &y, const arma::mat &phi,
                                 const arma::mat &drift, const arma::vec &tau0, const arma::mat &q,
                                 const arma::vec &irregular_inv, const arma::mat &mean = arma::mat(),
                                 const arma::vec &cycle_pre = arma::vec(),
                                 const UcExtension &ext = UcExtension())
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = y.n_cols;
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword band = std::max({p, order, signal_reach(ext)});
    const arma::uword w = 2 * k;
    const arma::mat eye_k = arma::eye<arma::mat>(k, k);
    const arma::mat weights = trend_lag_weights(spec, ext.rho);
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
        m[l].submat(k, 0, w - 1, k - 1) = arma::diagmat(weights.col(l));
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

        // The cycle's lags before the sample, constants given cycle_pre.
        r.head(k) = -cycle_presample_term(phi, cycle_pre, k, p, t);
        r.tail(k) = trend_offset(drift.col(t), tau0, k, weights, t);
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

    if (ext.signal.active())
    {
        const UcSignalTerms &sig = ext.signal;
        std::vector<arma::mat> n(sig.lags + 1);
        for (arma::uword j = 0; j <= sig.lags; j++)
        {
            n[j] = arma::zeros<arma::mat>(sig.precision.n_elem, w);
            n[j].cols(k, w - 1) = -sig.loading(j, k);
        }
        add_signal_measurement(n, sig.offset, sig.precision, band, blocks, rhs);
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
                           const arma::vec &irregular_inv, const arma::mat &mean = arma::mat(),
                           const arma::vec &cycle_pre = arma::vec(),
                           const UcExtension &ext = UcExtension())
{
    UcPath out;
    if (irregular_inv.n_elem == 0)
    {
        out.trend = draw_uc_trend(spec, y, phi, drift, tau0, q, mean, cycle_pre, ext);
        out.cycle = y - out.trend;
        return out;
    }
    const arma::uword k = static_cast<arma::uword>(spec.k);
    UcTrendBand sys =
        uc_state_band(spec, y, phi, drift, tau0, q, irregular_inv, mean, cycle_pre, ext);
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
                                              const arma::uvec &positions,
                                              const arma::vec &cycle_pre = arma::vec(),
                                              const arma::vec &rho = arma::vec())
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
    const arma::mat weights = trend_lag_weights(spec, rho);

    ShockRegression out;
    out.w = arma::mat(2 * k, tt);
    out.x = arma::cube(2 * k, n_theta, tt, arma::fill::zeros);

    for (arma::uword t = 0; t < tt; t++)
    {
        arma::mat x_t(2 * k, n_theta, arma::fill::zeros);

        // The cycle: c_t on its p lags, (x_t' kron I_k) vec(Phi), zero before
        // the sample unless cycle_pre says what it was.
        out.w.col(t).head(k) = cycle.col(t);
        if (n_a > 0)
        {
            arma::vec lags(k * p, arma::fill::zeros);
            for (arma::uword j = 1; j <= p; j++)
            {
                if (j <= t)
                {
                    lags.subvec((j - 1) * k, j * k - 1) = cycle.col(t - j);
                }
                else if (cycle_pre.n_elem > 0)
                {
                    const arma::uword i = j - t - 1;
                    lags.subvec((j - 1) * k, j * k - 1) = cycle_pre.subvec(i * k, (i + 1) * k - 1);
                }
            }
            const arma::mat full = arma::kron(arma::trans(lags), eye_k);
            x_t.submat(0, 0, k - 1, n_a - 1) = full.cols(positions);
        }

        // The trend: its lag polynomial on the drift and, near the start, on
        // the trend before the sample.
        arma::vec diff = tau.col(t);
        for (arma::uword l = 1; l <= order && l <= t; l++)
        {
            diff += weights.col(l) % tau.col(t - l);
        }
        out.w.col(t).tail(k) = diff;

        if (n_c > 0)
        {
            x_t.submat(k, n_a, 2 * k - 1, n_a + n_c - 1) = arma::kron(d.row(t), eye_k);
        }
        const arma::uword at = n_a + n_c;
        if (t == 0)
        {
            for (arma::uword l = 1; l <= order; l++)
            {
                x_t.submat(k, at + (l - 1) * k, 2 * k - 1, at + l * k - 1) =
                    arma::diagmat(-weights.col(l));
            }
        }
        else if (t == 1 && order == 2)
        {
            x_t.submat(k, at, 2 * k - 1, at + k - 1) = arma::diagmat(-weights.col(2));
        }

        out.x.slice(t) = x_t;
    }

    return out;
}

/// Without an irregular component, the cycle is y minus the trend.
inline ShockRegression shock_regression(const VarSpec &spec, const arma::mat &y,
                                        const arma::mat &tau, const arma::mat &d,
                                        const arma::uvec &positions,
                                        const arma::vec &rho = arma::vec())
{
    return shock_regression_paths(spec, tau, y - tau, d, positions, arma::vec(), rho);
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
    arma::mat z_signal;   ///< n_signals x m, [Lambda_0 .. Lambda_q] on the cycle block; empty without signals.
    arma::uword m = 0;
    arma::uword k = 0;
};

/// With `ext`, the damped trend's coefficients in the trend rows of T and the
/// signals' loadings in `z_signal`. Signals reaching back to c_{t-q} need q + 1
/// cycle values in the state, so the cycle block is max(p, q + 1) long there.
inline UcStateSpace state_space(const VarSpec &spec, const arma::mat &phi, const arma::mat &sigma,
                                const UcExtension &ext = UcExtension())
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword q =
        ext.signal.active() ? std::max<arma::uword>(p, ext.signal.lags + 1)
                            : std::max<arma::uword>(p, 1);
    const arma::uword cyc = k * order; // where the cycle block starts
    const arma::mat eye_k = arma::eye<arma::mat>(k, k);
    const arma::mat weights = trend_lag_weights(spec, ext.rho);

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
        ss.transition.submat(0, 0, k - 1, k - 1) = arma::diagmat(-weights.col(1));
        ss.transition.submat(0, k, k - 1, 2 * k - 1) = arma::diagmat(-weights.col(2));
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

    if (ext.signal.active())
    {
        ss.z_signal = arma::zeros<arma::mat>(ext.signal.precision.n_elem, ss.m);
        for (arma::uword j = 0; j <= ext.signal.lags; j++)
        {
            ss.z_signal.cols(cyc + j * k, cyc + (j + 1) * k - 1) = ext.signal.loading(j, k);
        }
    }
    return ss;
}

/// What moves from period to period in the filter beyond the drift: the shock
/// covariance -- `sigma_path`, one 2k x 2k matrix per period stacked, or empty
/// where ss.state_cov holds for every period -- the shock means, 2k x periods or
/// empty, and the irregular component's variances, k or empty.
///
/// With signals, `signal_offset` (n_signals x periods, s_t - B x^s_t) and
/// `signal_var` (the noise variances) stack them under the series for the
/// periods `signal_offset` covers; a period past those -- a scored one -- sees
/// the series alone, so its density is that of y given everything before it.
struct UcFilterTerms
{
    arma::mat sigma_path;
    arma::mat shock_mean;
    arma::vec irregular_var;
    arma::mat signal_offset;
    arma::vec signal_var;
};

/// One step ahead log densities of y.col(0) .. y.col(periods - 1), started from
/// the trend before the sample at tau0 and the cycle at zero, both known: given
/// one draw of everything else, the start is not uncertain, and P_0 is zero.
/// Under UcCycleInit::stationary the cycle before the sample is not a parameter
/// but a state integrated out like the path: `initial_cycle_cov`, kp x kp, is
/// its stationary covariance and becomes the cycle block of P_0.
inline arma::rowvec filter_log_density(const UcStateSpace &ss, const arma::mat &y,
                                       const arma::mat &drift, const arma::vec &tau0,
                                       const UcFilterTerms &terms = UcFilterTerms(),
                                       const arma::mat &initial_cycle_cov = arma::mat())
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
    if (initial_cycle_cov.n_elem > 0)
    {
        const arma::uword cyc = tau0.n_elem;
        p_cov.submat(cyc, cyc, cyc + initial_cycle_cov.n_rows - 1,
                     cyc + initial_cycle_cov.n_cols - 1) = initial_cycle_cov;
    }
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

        arma::vec v;
        arma::mat pz, f;
        double norm = constant;
        if (t < terms.signal_offset.n_cols)
        {
            // The series and the signals stacked: Z on top of Z_signal, the
            // irregular variances (or zero) and the signal noise on the diagonal.
            const arma::uword ms = ss.z_signal.n_rows;
            const arma::mat zz = arma::join_vert(ss.z, ss.z_signal);
            v = arma::join_vert(arma::vec(y.col(t) - ss.z * a),
                                arma::vec(terms.signal_offset.col(t) - ss.z_signal * a));
            pz = pred * arma::trans(zz);
            arma::vec diag(k + ms, arma::fill::zeros);
            if (irregular)
            {
                diag.head(k) = terms.irregular_var;
            }
            diag.tail(ms) = terms.signal_var;
            f = arma::symmatu(zz * pz + arma::mat(arma::diagmat(diag)));
            norm = -static_cast<double>(k + ms) * std::log(2.0 * arma::datum::pi) / 2.0;
        }
        else
        {
            v = y.col(t) - ss.z * a;
            pz = pred * zt;
            f = irregular ? arma::mat(arma::symmatu(ss.z * pz + noise))
                          : arma::mat(arma::symmatu(ss.z * pz));
        }

        arma::mat root;
        if (!arma::chol(root, f))
        {
            throw std::runtime_error("unobserved components: the one step prediction variance is "
                                     "not positive definite at period " + std::to_string(t + 1));
        }
        const arma::vec scaled = arma::solve(arma::trimatl(arma::trans(root)), v);
        out(t) = norm - arma::accu(arma::log(root.diag())) - arma::dot(scaled, scaled) / 2.0;

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
/// component. `rho` is the damped trend's slope coefficients, empty otherwise;
/// signals are not forecast, the series never reading them.
template <typename Shock>
arma::vec uc_forecast_path(const VarSpec &spec, const arma::mat &tau, const arma::mat &cycle,
                           const arma::mat &phi, const arma::mat &c, const arma::mat &forecast_x,
                           const arma::vec &irregular_sd, const arma::vec &rho, Shock shock)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword h = static_cast<arma::uword>(spec.h);
    const arma::uword tt = tau.n_cols;
    const bool smooth = spec.uc_trend_order() == 2;
    const arma::mat weights = trend_lag_weights(spec, rho);

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

        arma::vec next = smooth ? arma::vec(-weights.col(1) % tau_last - weights.col(2) % tau_prev)
                                : tau_last;
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

/// The same without a damped trend.
template <typename Shock>
arma::vec uc_forecast_path(const VarSpec &spec, const arma::mat &tau, const arma::mat &cycle,
                           const arma::mat &phi, const arma::mat &c, const arma::mat &forecast_x,
                           const arma::vec &irregular_sd, Shock shock)
{
    return uc_forecast_path(spec, tau, cycle, phi, c, forecast_x, irregular_sd, arma::vec(), shock);
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
    /// Puts theta back, for a Metropolis-Hastings step that rejects a draw.
    void set_theta(const arma::vec &theta) { theta_ = theta; }

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

/// The damped trend's slope coefficients given everything else. With the trend
/// before the sample at tau0, each trend shock is
///
///     u^tau_t = (Delta tau_t - C d_t) - diag(rho) Delta tau_{t-1},
///
/// so the 2k shocks are a regression on rho, e_t = w_t - X_t rho, with the
/// cycle shocks -- `cycle_shocks`, k x tt -- held where they are. Drawn under
/// the whole shock precision (draw_damped_rho()), so that a trend shock
/// correlated with a cycle shock is informed by it.
inline ShockRegression damped_rho_regression(const VarSpec &spec, const arma::mat &tau,
                                             const arma::mat &cycle_shocks,
                                             const arma::mat &drift, const arma::vec &tau0)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = tau.n_cols;
    // tau at period s, s = -1 and -2 being tau0's two halves.
    const auto level = [&](const long long s) -> arma::vec {
        if (s >= 0)
        {
            return tau.col(static_cast<arma::uword>(s));
        }
        const arma::uword i = static_cast<arma::uword>(-s - 1);
        return tau0.subvec(i * k, (i + 1) * k - 1);
    };

    ShockRegression reg;
    reg.w = arma::mat(2 * k, tt);
    reg.x = arma::cube(2 * k, k, tt, arma::fill::zeros);
    for (arma::uword t = 0; t < tt; t++)
    {
        const long long s = static_cast<long long>(t);
        const arma::vec now = level(s) - level(s - 1);
        const arma::vec before = level(s - 1) - level(s - 2);
        reg.w.col(t).head(k) = cycle_shocks.col(t);
        reg.w.col(t).tail(k) = now - drift.col(t);
        reg.x.slice(t).submat(k, 0, 2 * k - 1, k - 1) = arma::diagmat(before);
    }
    return reg;
}

/// Draws rho from the regression damped_rho_regression() sets up, under the
/// shock precision `q` -- one matrix or a stack of one per period -- and means
/// `mean`, or empty. The normal prior is truncated to |rho_i| < 1 by the rule
/// draw_stationary() uses: up to kStationaryTries proposals, and `rho` left as
/// it was, returning false, if none is inside.
inline bool draw_damped_rho(const VarSpec &spec, const arma::mat &tau,
                            const arma::mat &cycle_shocks, const arma::mat &drift,
                            const arma::vec &tau0, const arma::mat &q, const arma::mat &mean,
                            const NormalPrior &prior, arma::vec &rho)
{
    const ShockRegression reg = damped_rho_regression(spec, tau, cycle_shocks, drift, tau0);
    arma::mat post_vinv = prior.v_inv;
    arma::vec post_rhs = prior.v_inv * prior.mu;
    add_shock_likelihood(reg, q, mean, post_vinv, post_rhs);
    for (int attempt = 0; attempt < kStationaryTries; attempt++)
    {
        const arma::vec proposal = draw_normal_precision(post_vinv, post_rhs);
        if (arma::all(arma::abs(proposal) < 1.0))
        {
            rho = proposal;
            return true;
        }
    }
    return false;
}

/// Says how many sweeps kept the previous rho for want of a draw inside the
/// unit interval, if any did.
inline void report_undamped_sweeps(Reporter &reporter, const int sweeps, const int draws)
{
    if (sweeps > 0)
    {
        reporter.message(std::to_string(sweeps) + " of " + std::to_string(draws) +
                         " sweeps found no draw of the damped trend's rho inside (-1, 1) in " +
                         std::to_string(kStationaryTries) +
                         " tries and kept the one before: the posterior may put much of its mass "
                         "on a slope with a unit root, which is the smooth trend");
    }
}

/// The regressors of the signal equations, one column per period: x^s_t, then
/// c_t, c_{t-1}, .., c_{t-q}, the cycle zero before the sample. `x` is r x tt,
/// or empty without regressors of the signals' own.
inline arma::mat signal_regressors(const arma::mat &x, const arma::mat &cycle, const arma::uword q)
{
    const arma::uword k = cycle.n_rows;
    const arma::uword tt = cycle.n_cols;
    const arma::uword r = x.n_rows;
    arma::mat z(r + k * (q + 1), tt, arma::fill::zeros);
    if (r > 0)
    {
        z.rows(0, r - 1) = x;
    }
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword j = 0; j <= q && j <= t; j++)
        {
            z.col(t).subvec(r + j * k, r + (j + 1) * k - 1) = cycle.col(t - j);
        }
    }
    return z;
}

/// The signal equations' coefficients and noise precisions, drawn given the
/// cycle path: m regressions on the same regressors with independent noise, so
/// vec(Gamma), Gamma = [B Lambda_0 .. Lambda_q], is normal with precision
/// V^-1 + (Z Z' kron H) and right hand side V^-1 mu + vec(H S Z'), whatever the
/// prior couples, and each precision gamma given the residuals.
class UcSignalBlock
{
public:
    /// `s` and `x` as the file holds them, tt x m and tt x r (or empty).
    UcSignalBlock(const VarSpec &spec, const arma::mat &s, const arma::mat &x,
                  const NormalPrior &prior, const arma::vec &initial, const GammaPrior &noise_prior,
                  const arma::vec &initial_noise_inv)
        : prior_(prior), noise_prior_(noise_prior)
    {
        if (spec.n_signals <= 0)
        {
            return;
        }
        k_ = static_cast<arma::uword>(spec.k);
        s_ = arma::trans(s);
        x_ = x.n_elem > 0 ? arma::mat(arma::trans(x)) : arma::mat();
        terms_.lags = static_cast<arma::uword>(spec.signal_lags);
        gamma_ = arma::reshape(initial, static_cast<arma::uword>(spec.n_signals),
                               static_cast<arma::uword>(spec.n_uc_signal_regressors()));
        precision_ = initial_noise_inv;
        refresh();
    }

    bool active() const { return precision_.n_elem > 0; }
    const UcSignalTerms &terms() const { return terms_; }
    arma::vec coefficients() const { return arma::vectorise(gamma_); }
    const arma::vec &precision() const { return precision_; }

    /// The coefficients given the precisions, then the precisions given them.
    void draw(const arma::mat &cycle)
    {
        if (!active())
        {
            return;
        }
        const arma::mat z = signal_regressors(x_, cycle, terms_.lags);
        const arma::mat h = arma::diagmat(precision_);
        const arma::mat post_vinv = prior_.v_inv + arma::kron(z * arma::trans(z), h);
        const arma::vec post_rhs = prior_.v_inv * prior_.mu + arma::vectorise(h * s_ * arma::trans(z));
        gamma_ = arma::reshape(draw_normal_precision(post_vinv, post_rhs), gamma_.n_rows,
                               gamma_.n_cols);

        const arma::mat resid = s_ - gamma_ * z;
        const arma::vec sse = arma::sum(arma::square(resid), 1);
        const double half = static_cast<double>(resid.n_cols) * 0.5;
        for (arma::uword i = 0; i < precision_.n_elem; i++)
        {
            precision_(i) = arma::randg<double>(arma::distr_param(
                noise_prior_.shape(i) + half, 1.0 / (noise_prior_.rate(i) + sse(i) * 0.5)));
        }
        refresh();
    }

private:
    void refresh()
    {
        const arma::uword r = x_.n_rows;
        terms_.offset = r > 0 ? arma::mat(s_ - gamma_.cols(0, r - 1) * x_) : s_;
        terms_.lambda = gamma_.cols(r, gamma_.n_cols - 1);
        terms_.precision = precision_;
    }

    NormalPrior prior_;
    GammaPrior noise_prior_;
    arma::uword k_ = 0;
    arma::mat s_, x_;  ///< m x tt and r x tt.
    arma::mat gamma_;  ///< m x (r + k(q + 1)).
    arma::vec precision_;
    UcSignalTerms terms_;
};

/// The signal terms of posterior draw i: vec(Gamma) in column i of `signal`,
/// the precisions in column i of `noise_inv`, over the signals `s` (tt x m) and
/// their regressors `x` (tt x r, or empty) of the sample.
inline UcSignalTerms uc_signal_terms_at(const VarSpec &spec, const arma::mat &signal,
                                        const arma::mat &noise_inv, const arma::mat &s,
                                        const arma::mat &x, const arma::uword i)
{
    UcSignalTerms out;
    if (spec.n_signals <= 0)
    {
        return out;
    }
    const arma::mat gamma =
        arma::reshape(signal.col(i), static_cast<arma::uword>(spec.n_signals),
                      static_cast<arma::uword>(spec.n_uc_signal_regressors()));
    const arma::uword r = static_cast<arma::uword>(spec.n_signal_x);
    out.offset = arma::trans(s);
    if (r > 0)
    {
        out.offset -= gamma.cols(0, r - 1) * arma::trans(x);
    }
    out.lambda = gamma.cols(r, gamma.n_cols - 1);
    out.precision = noise_inv.col(i);
    out.lags = static_cast<arma::uword>(spec.signal_lags);
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

/// The same for what UcExtension adds: the damped trend's rho and the signal
/// equations' coefficients and noise precisions, each where the spec asks for
/// it, with the same number of draws as `tau0`.
inline void require_uc_extension_draws(const VarSpec &spec, const arma::mat &rho,
                                       const arma::mat &signal, const arma::mat &noise_inv,
                                       const arma::uword n_draws)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const auto require = [n_draws](const arma::mat &draws, const arma::uword rows,
                                   const std::string &what) {
        if (draws.n_rows != rows || draws.n_cols != n_draws)
        {
            throw std::invalid_argument("posterior draws of " + what + " must be " +
                                        std::to_string(rows) + " x " + std::to_string(n_draws) +
                                        ", got " + std::to_string(draws.n_rows) + " x " +
                                        std::to_string(draws.n_cols));
        }
    };
    if (spec.trend == UcTrend::damped)
    {
        require(rho, k, "the damped trend's rho");
    }
    if (spec.n_signals > 0)
    {
        require(signal, static_cast<arma::uword>(spec.n_uc_signal()),
                "the signal coefficients");
        require(noise_inv, static_cast<arma::uword>(spec.n_signals),
                "the signal noise precision");
    }
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

// ---------------------------------------------------------------------------
// The cycle before the sample, under UcCycleInit::stationary.
//
// The p values the first lags reach back to, s_0 = (c_0', c_-1', .., c_1-p')',
// are a parameter of their own with the stationary distribution of the cycle as
// its prior, N(0, Gamma), Gamma the covariance of p consecutive values of a
// VAR(p) with coefficients Phi and shock covariance Sigma_c -- the cycle block
// of the shock covariance, since only the cycle's shocks move the cycle. Given
// everything else s_0 is normal, and given s_0 the rest is drawn as before with
// s_0 in place of the zeros. Gamma depends on Phi and on Sigma_c, so the blocks
// that draw those keep their proposals -- the conditionals without that one
// factor -- and accept with the ratio of N(s_0; 0, Gamma) under the new and the
// old values, which is Metropolis-Hastings with the conditional as proposal.
// ---------------------------------------------------------------------------

/// Gamma = F Gamma F' + E Sigma_c E', the covariance of the companion state of
/// a stationary VAR(p) with coefficients `phi` (k x kp) and shock covariance
/// `sigma_c` (k x k), kp x kp. Solved through vec(Gamma) = (I - F kron F)^-1
/// vec(E Sigma_c E'), which is exact and small at the sizes a cycle has. Throws
/// if F has a root on or outside the unit circle, where there is none.
inline arma::mat cycle_stationary_covariance(const arma::mat &phi, const arma::mat &sigma_c)
{
    const arma::uword k = phi.n_rows;
    const arma::uword n = phi.n_cols;
    arma::mat f(n, n, arma::fill::zeros);
    f.rows(0, k - 1) = phi;
    if (n > k)
    {
        f.submat(k, 0, n - 1, n - k - 1) = arma::eye<arma::mat>(n - k, n - k);
    }
    arma::mat e_sigma(n, n, arma::fill::zeros);
    e_sigma.submat(0, 0, k - 1, k - 1) = sigma_c;

    const arma::mat lhs = arma::eye<arma::mat>(n * n, n * n) - arma::kron(f, f);
    arma::vec gamma_vec;
    if (!arma::solve(gamma_vec, lhs, arma::vectorise(e_sigma), arma::solve_opts::no_approx))
    {
        throw std::runtime_error("unobserved components: the cycle has no stationary "
                                 "distribution at the drawn coefficients");
    }
    return arma::symmatu(arma::reshape(gamma_vec, n, n));
}

/// log N(cycle_pre; 0, gamma), or -inf where gamma is not positive definite --
/// which a Metropolis-Hastings step then never moves to.
inline double cycle_presample_log_density(const arma::vec &cycle_pre, const arma::mat &gamma)
{
    arma::mat root;
    if (!arma::chol(root, gamma))
    {
        return -arma::datum::inf;
    }
    const arma::vec z = arma::solve(arma::trimatl(arma::trans(root)), cycle_pre);
    return -0.5 * static_cast<double>(cycle_pre.n_elem) * std::log(2.0 * arma::datum::pi) -
           arma::accu(arma::log(root.diag())) - 0.5 * arma::dot(z, z);
}

/// The cycle block, k x k, of the covariance a shock precision stands for.
inline arma::mat cycle_shock_covariance(const arma::mat &q, const arma::uword k)
{
    return uc_covariance_of_precision(q).submat(0, 0, k - 1, k - 1);
}

/// The posterior precision and right hand side of s_0 given everything else.
/// The prior is N(0, gamma); the first p periods' shocks are e_t = b_t - G_t s_0,
/// b_t their value with s_0 at zero and G_t holding Phi_j in the cycle rows of
/// block column j - t - 1 for every lag j that reaches before the sample.
/// `shocks_zero` is b, 2k x tt (only its first p columns are read), `q` the
/// constant shock precision.
inline void cycle_presample_posterior(const arma::mat &phi, const arma::mat &gamma,
                                      const arma::mat &shocks_zero, const arma::mat &q,
                                      arma::mat &post_vinv, arma::vec &post_rhs)
{
    const arma::uword k = phi.n_rows;
    const arma::uword p = phi.n_cols / k;
    const arma::uword n = k * p;
    if (!arma::inv_sympd(post_vinv, gamma))
    {
        throw std::runtime_error("unobserved components: the stationary covariance of the cycle "
                                 "is not positive definite");
    }
    post_rhs = arma::zeros<arma::vec>(n);
    for (arma::uword t = 0; t < p && t < shocks_zero.n_cols; t++)
    {
        arma::mat g(2 * k, n, arma::fill::zeros);
        for (arma::uword j = t + 1; j <= p; j++)
        {
            const arma::uword i = j - t - 1;
            g.submat(0, i * k, k - 1, (i + 1) * k - 1) = phi.cols((j - 1) * k, j * k - 1);
        }
        const arma::mat gq = arma::trans(g) * q;
        post_vinv += gq * g;
        post_rhs += gq * shocks_zero.col(t);
    }
}

/// Draws s_0 given everything else; see cycle_presample_posterior().
inline arma::vec draw_cycle_presample(const arma::mat &phi, const arma::mat &gamma,
                                      const arma::mat &shocks_zero, const arma::mat &q)
{
    arma::mat post_vinv;
    arma::vec post_rhs;
    cycle_presample_posterior(phi, gamma, shocks_zero, q, post_vinv, post_rhs);
    return draw_normal_precision(post_vinv, post_rhs);
}

/// The state of the cycle before the sample in a sampler with one shock
/// precision per draw: s_0, the Gamma it is drawn under and its log density
/// there, kept so that a block that moves Phi or Sigma_c can accept or reject.
/// Inactive -- every member empty, accept() always true and consuming no random
/// numbers -- under UcCycleInit::zero, which is what keeps those draws as they
/// were.
class UcCyclePresample
{
public:
    UcCyclePresample(const VarSpec &spec, const arma::mat &phi, const arma::mat &q)
    {
        active_ = spec.cycle_init == UcCycleInit::stationary && spec.p > 0;
        if (!active_)
        {
            return;
        }
        k_ = static_cast<arma::uword>(spec.k);
        value_ = arma::zeros<arma::vec>(k_ * static_cast<arma::uword>(spec.p));
        try
        {
            set(phi, q);
        }
        catch (const std::runtime_error &)
        {
            throw std::invalid_argument(
                "/model/cycle_init stationary needs initial cycle coefficients /initial/a that "
                "describe a stationary cycle, whose stationary distribution the chain starts from");
        }
    }

    bool active() const { return active_; }
    /// s_0, or empty for zero.
    const arma::vec &value() const { return value_; }

    /// Whether to keep a move of Phi or of the shock precision to (phi, q): the
    /// Metropolis-Hastings ratio of N(s_0; 0, Gamma). Remembers the new Gamma if
    /// it does.
    bool accept(const arma::mat &phi, const arma::mat &q)
    {
        if (!active_)
        {
            return true;
        }
        arma::mat gamma;
        try
        {
            gamma = cycle_stationary_covariance(phi, cycle_shock_covariance(q, k_));
        }
        catch (const std::runtime_error &)
        {
            return false;
        }
        const double log_density = cycle_presample_log_density(value_, gamma);
        if (!std::isfinite(log_density) ||
            std::log(arma::randu<double>()) > log_density - log_density_)
        {
            return false;
        }
        gamma_ = gamma;
        log_density_ = log_density;
        return true;
    }

    /// Draws s_0 given the first p periods' shocks with s_0 at zero.
    void draw(const arma::mat &phi, const arma::mat &shocks_zero, const arma::mat &q)
    {
        if (!active_)
        {
            return;
        }
        value_ = draw_cycle_presample(phi, gamma_, shocks_zero, q);
        log_density_ = cycle_presample_log_density(value_, gamma_);
    }

private:
    void set(const arma::mat &phi, const arma::mat &q)
    {
        gamma_ = cycle_stationary_covariance(phi, cycle_shock_covariance(q, k_));
        log_density_ = cycle_presample_log_density(value_, gamma_);
    }

    bool active_ = false;
    arma::uword k_ = 0;
    arma::vec value_;
    arma::mat gamma_;
    double log_density_ = 0.0;
};

} // namespace bayests::core

#endif // BAYESTS_CORE_MODELS_UC_SUPPORT_H
