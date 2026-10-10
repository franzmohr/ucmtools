#' Unobserved Components Model
#'
#' Sets up an unobserved components model, which decomposes each of one or more
#' time series into a stochastic trend and a stationary cycle.
#'
#' @param y a time-series object of one or more series, in columns. Typically
#' the logarithm of a series times 100, so that the trend's increments are
#' growth rates in percent.
#' @param p the order of the cycle, an autoregression of \code{p} lags. Zero
#' leaves it white noise, which is the local level model. A vector gives a list
#' of models, one per order.
#' @param trend \code{"rw"} for a random walk trend, whose first differences are
#' the shocks, \code{"smooth"} for a smooth trend, whose second differences
#' are, or \code{"damped"} for a damped trend, whose slope is a stationary
#' autoregression of order one, as in the European Commission's production
#' function method (Havik et al. 2014). \code{"damped"} is available for
#' \code{error = "gamma"}, \code{"gamma+covar"} and \code{"wishart"}; its
#' coefficients are \code{posterior$rho}.
#' @param cycle \code{"var"} to regress each cycle on the lags of every cycle,
#' or \code{"own"} to regress each on its own lags only. The two are the same
#' for one series.
#' @param drift logical. Should the trend have a drift, a constant in its
#' increment? Defaults to \code{TRUE} for a random walk and a damped trend and
#' \code{FALSE} for a smooth one. Under a damped trend the slope's mean is the
#' drift divided by \eqn{1 - \rho}.
#' @param breaks the periods in which the drift breaks, as a list of
#' \code{c(year, period)} pairs or a vector of times in the units of
#' \code{time(y)}. Each adds a step dummy to the drift, one from that period on.
#' Needs \code{drift = TRUE}.
#' @param irregular logical. Should each series have an irregular component, a
#' white noise beside the trend and the cycle? Needs \code{p > 0}.
#' @param error the distribution of the shocks: \code{"gamma"} for independent
#' normal shocks with gamma priors on their precisions, \code{"gamma+covar"}
#' for correlated ones, \code{"wishart"} for an unrestricted covariance with a
#' Wishart prior on its inverse, \code{"sv"} and \code{"sv+covar"} for
#' stochastic volatility, and \code{"skew"} for skewed shocks.
#' @param stationary logical. Should every draw of the cycle coefficients
#' describe a stationary cycle, as in Grant and Chan (2017)? The default for a
#' cycle with lags. Without the restriction a share of the draws can put the
#' cycle at or past a unit root, where it absorbs persistent movements of the
#' trend and its credible band widens accordingly.
#' @param cycle_init where the cycle starts: \code{"zero"}, the cycle is zero
#' before the sample, as in Grant and Chan (2017), or \code{"stationary"}, the
#' \code{p} values before the sample are drawn from the cycle's stationary
#' distribution. Zero pins the first estimates of the cycle near zero whatever
#' the data say. \code{"stationary"} needs a stationary cycle with lags and is
#' available for \code{error = "gamma"}, \code{"gamma+covar"} and
#' \code{"wishart"}; the draws of the values before the sample are
#' \code{posterior$cycle0}.
#' @param skew,skew_shock,skew_x for \code{error = "skew"}: the distribution of
#' the skewed shock, \code{"snd"} for the skew normal or \code{"ald"} for the
#' asymmetric Laplace; which shock is skewed, \code{"cycle"} or \code{"trend"};
#' and an optional time-series object of the variables the log scales of the
#' skewed shock move with, such as an uncertainty index, covering the periods of
#' \code{y}.
#' @param signal an optional time-series object of further observed series,
#' the signals, each a regression on the cycle at lags 0 to
#' \code{signal_lags}: the capacity utilisation indicator on the cycle of total
#' factor productivity, or a Phillips curve on the unemployment gap. A signal
#' informs the cycle, and through it the trend, but is not forecast. It must
#' cover the periods of \code{y} without missing values. Available for
#' \code{error = "gamma"}, \code{"gamma+covar"} and \code{"wishart"}, and not
#' with \code{cycle_init = "stationary"}, since the signals read the cycle
#' before the sample as zero.
#' @param signal_lags the last lag of the cycle the signals load on; zero for
#' the current cycle only.
#' @param signal_x an optional time-series object of regressors of the signal
#' equations besides the cycle, covering the periods of \code{y}. Every signal
#' regresses on all of them.
#' @param signal_const logical. Should the signal equations have a constant, as
#' the first of their regressors?
#' @param iterations an integer of MCMC draws kept after the burn-in.
#' @param burnin an integer of MCMC draws discarded first.
#' @param thin an integer, the thinning interval: one draw in \code{thin} is
#' kept, \code{iterations * thin} are drawn after the burn-in.
#'
#' @details The model of series \eqn{i} is
#' \deqn{y_t = \tau_t + c_t (+ \epsilon_t),}
#' \deqn{c_t = \Phi_1 c_{t-1} + \dots + \Phi_p c_{t-p} + u^c_t,}
#' \deqn{\tau_t = \tau_{t-1} + C d_t + u^\tau_t}
#' for a random walk trend, and \eqn{\tau_t = 2\tau_{t-1} - \tau_{t-2} + C d_t + u^\tau_t}
#' for a smooth one, and
#' \eqn{\tau_t - \tau_{t-1} = \rho (\tau_{t-1} - \tau_{t-2}) + C d_t + u^\tau_t}
#' with \eqn{|\rho| < 1}, one \eqn{\rho} per series, for a damped one, where
#' \eqn{d_t} holds the drift and its breaks. Signals add the equations
#' \deqn{s_t = B x^s_t + \Lambda_0 c_t + \dots + \Lambda_q c_{t-q} + v_t,}
#' with independent normal noise \eqn{v_t}, where \eqn{x^s_t} holds the
#' constant and \code{signal_x}, and \eqn{q} is \code{signal_lags}. A damped
#' trend and a signal together give the trend-cycle models the European
#' Commission estimates potential output with (Havik et al. 2014). It nests
#' the decompositions of Clark (1987), Morley, Nelson and Zivot (2003) and
#' Grant and Chan (2017), the local level model, the smooth-trend output gap
#' model and, with stochastic volatility, the UCSV model of Stock and Watson
#' (2007). The posterior is simulated by the Gibbs samplers of BayesTS, the
#' trend path drawn whole by the precision sampler of Chan and Jeliazkov (2009).
#'
#' The shocks are ordered cycle shocks first, then trend shocks, in every prior
#' and every draw that has one entry per shock.
#'
#' @return An object of class \code{ucmodel}, or a \code{modellist} of them when
#' \code{p} has more than one element.
#'
#' @references
#' Clark, P. K. (1987). The cyclical component of U.S. economic activity.
#' \emph{The Quarterly Journal of Economics, 102}(4), 797--814.
#'
#' Grant, A. L., & Chan, J. C. C. (2017). A Bayesian model comparison for
#' trend-cycle decompositions of output. \emph{Journal of Money, Credit and
#' Banking, 49}(2--3), 525--552.
#'
#' Havik, K., Mc Morrow, K., Orlandi, F., Planas, C., Raciborski, R., Roeger,
#' W., Rossi, A., Thum-Thysen, A., & Vandermeulen, V. (2014). The production
#' function methodology for calculating potential growth rates and output gaps.
#' \emph{European Economy, Economic Papers 535}. European Commission.
#'
#' Huang, Luo and Wang (2026), on unobserved components models with skewed
#' shocks whose scales move with macroeconomic uncertainty.
#'
#' Morley, J. C., Nelson, C. R., & Zivot, E. (2003). Why are the
#' Beveridge-Nelson and unobserved-components decompositions of GDP so
#' different? \emph{The Review of Economics and Statistics, 85}(2), 235--243.
#'
#' Stock, J. H., & Watson, M. W. (2007). Why has U.S. inflation become harder
#' to forecast? \emph{Journal of Money, Credit and Banking, 39}(s1), 3--33.
#'
#' @examples
#' data("at_gdp")
#' y <- at_gdp[, "y"] * 100
#'
#' model <- create_ucmodel(y, p = 2, iterations = 200, burnin = 100)
#' model <- add_priors(model)
#' model <- add_initial_values(model)
#' model <- add_posterior_coefficients(model)
#' plot(model)
#'
#' @export
create_ucmodel <- function(y, p = 2, trend = "rw", cycle = "var", drift = NULL,
                           breaks = NULL, irregular = FALSE, error = "gamma",
                           stationary = NULL, cycle_init = "zero", skew = "snd",
                           skew_shock = "cycle", skew_x = NULL, signal = NULL,
                           signal_lags = 0, signal_x = NULL, signal_const = TRUE,
                           iterations = 10000, burnin = 2000, thin = 1) {

  if (!stats::is.ts(y)) {
    stop("Argument 'y' must be a time-series object.", call. = FALSE)
  }
  .check_model_integer(p, "p", 0)
  .check_choice(trend, "trend", c("rw", "smooth", "damped"))
  .check_choice(cycle, "cycle", c("var", "own"))
  .check_choice(error, "error", names(.uc_algorithms))
  .check_choice(skew, "skew", c("snd", "ald"))
  .check_choice(skew_shock, "skew_shock", c("cycle", "trend"))
  .check_flag(irregular, "irregular")
  if (is.null(stationary)) {
    stationary <- all(p > 0)
  }
  .check_flag(stationary, "stationary")
  .check_choice(cycle_init, "cycle_init", c("zero", "stationary"))
  .check_whole_number(iterations, "iterations", 1)
  .check_whole_number(burnin, "burnin", 0)
  if (is.null(drift)) {
    drift <- trend != "smooth"
  }
  .check_flag(drift, "drift")
  .check_whole_number(signal_lags, "signal_lags", 0)
  .check_flag(signal_const, "signal_const")

  # The two models with one shock precision per draw are the ones with a Gibbs
  # block for rho and for the signal equations.
  extends <- error %in% c("gamma", "gamma+covar", "wishart")
  if (trend == "damped" && !extends) {
    stop("Argument trend = \"damped\" is available for error = \"gamma\", ",
         "\"gamma+covar\" and \"wishart\".", call. = FALSE)
  }
  if (is.null(signal)) {
    if (!is.null(signal_x) || signal_lags != 0) {
      stop("Arguments 'signal_lags' and 'signal_x' describe signal equations; give ",
           "the signals in argument 'signal'.", call. = FALSE)
    }
  } else {
    if (!extends) {
      stop("Argument 'signal' is available for error = \"gamma\", \"gamma+covar\" ",
           "and \"wishart\".", call. = FALSE)
    }
    if (cycle_init == "stationary") {
      stop("Argument 'signal' is not combined with cycle_init = \"stationary\": the ",
           "signals read the cycle before the sample as zero.", call. = FALSE)
    }
  }

  if (irregular && any(p == 0)) {
    stop("An irregular component needs a cycle with lags: at p = 0 the cycle is ",
         "white noise as well, and the two cannot be told apart.", call. = FALSE)
  }
  if (stationary && any(p == 0)) {
    stop("Argument 'stationary' needs a cycle with lags, p > 0.", call. = FALSE)
  }
  if (cycle_init == "stationary") {
    if (!error %in% c("gamma", "gamma+covar", "wishart")) {
      stop("Argument cycle_init = \"stationary\" is available for error = \"gamma\", ",
           "\"gamma+covar\" and \"wishart\": the stationary distribution is taken under ",
           "one shock covariance per draw.", call. = FALSE)
    }
    if (!stationary || any(p == 0)) {
      stop("Argument cycle_init = \"stationary\" needs a stationary cycle with lags: ",
           "stationary = TRUE and p > 0.", call. = FALSE)
    }
  }
  if (!is.null(breaks) && !drift) {
    stop("Argument 'breaks' needs a drift to break; set drift = TRUE.", call. = FALSE)
  }
  if (!is.null(skew_x) && error != "skew") {
    stop("Argument 'skew_x' is used by error = \"skew\" only.", call. = FALSE)
  }

  y <- .as_named_ts_matrix(y)
  if (anyNA(y)) {
    stop("Argument 'y' contains missing values.", call. = FALSE)
  }
  det <- .uc_deterministic(y, drift, breaks)

  if (!is.null(skew_x)) {
    skew_x <- .align_to_y(skew_x, y, "skew_x")
  }
  if (!is.null(signal)) {
    signal <- .align_to_y(signal, y, "signal", prefix = "s")
    if (signal_lags >= nrow(y)) {
      stop("Argument 'signal_lags' must be smaller than the number of periods of 'y'.",
           call. = FALSE)
    }
    if (!is.null(signal_x)) {
      signal_x <- .align_to_y(signal_x, y, "signal_x", prefix = "x")
    }
    if (signal_const) {
      signal_x <- cbind(matrix(1, nrow(y), 1, dimnames = list(NULL, "const")), signal_x)
    }
  }

  model <- list(type = "UC",
                algorithm = .uc_algorithms[[error]],
                k = ncol(y),
                p = 0L,
                trend = trend,
                cycle = cycle,
                irregular = irregular,
                error = error,
                deterministic = colnames(det),
                iterations = as.integer(iterations),
                burnin = as.integer(burnin))
  model[["thin"]] <- .check_sampler_thin(thin)
  if (stationary) {
    model[["stationary"]] <- TRUE
  }
  if (cycle_init == "stationary") {
    model[["cycle_init"]] <- "stationary"
  }
  if (error == "skew") {
    model[["skew"]] <- skew
    model[["skew_shock"]] <- skew_shock
  }
  if (!is.null(signal)) {
    model[["n_signals"]] <- ncol(signal)
    model[["signal_lags"]] <- as.integer(signal_lags)
    model[["n_signal_x"]] <- if (is.null(signal_x)) 0L else ncol(signal_x)
  }

  data <- list(y = y)
  data[["deterministic"]] <- det
  data[["skew_x"]] <- skew_x
  data[["signal"]] <- signal
  data[["signal_x"]] <- signal_x

  result <- lapply(p, function(p_i) {
    model_i <- model
    model_i[["p"]] <- as.integer(p_i)
    structure(list(data = data, model = model_i), class = c("ucmodel", "list"))
  })

  if (length(result) == 1) {
    return(result[[1]])
  }
  class(result) <- c("modellist", "list")
  result
}

# A time-series matrix with a name for every column.
.as_named_ts_matrix <- function(y) {
  tsp_y <- stats::tsp(y)
  names_y <- colnames(y)
  y <- as.matrix(y)
  if (is.null(names_y)) {
    names_y <- if (ncol(y) == 1) "y" else paste0("y", seq_len(ncol(y)))
  }
  stats::ts(unname(y), start = tsp_y[1], frequency = tsp_y[3], names = names_y)
}

# The deterministic terms of the trend's increment: a constant for the drift and
# a step dummy per break, or NULL without a drift. A plain matrix with one row
# per period and one column per term.
.uc_deterministic <- function(y, drift, breaks) {
  if (!drift) {
    return(NULL)
  }
  times <- stats::time(y)
  freq <- stats::frequency(y)
  det <- matrix(1, nrow(y), 1, dimnames = list(NULL, "const"))
  if (is.null(breaks)) {
    return(det)
  }
  if (!is.list(breaks)) {
    breaks <- as.list(breaks)
  }
  for (b in breaks) {
    if (!is.numeric(b) || !length(b) %in% 1:2 || anyNA(b)) {
      stop("Each element of 'breaks' must be a time or a c(year, period) pair.", call. = FALSE)
    }
    at <- if (length(b) == 2) b[1] + (b[2] - 1) / freq else b
    if (at <= times[1] + 1e-8 || at > times[length(times)] + 1e-8) {
      stop("A break must fall after the first period of 'y' and within the sample.",
           call. = FALSE)
    }
    label <- if (freq > 1) {
      paste0("break", floor(at + 1e-8), "q", round((at - floor(at + 1e-8)) * freq) + 1)
    } else {
      paste0("break", round(at))
    }
    if (freq != 4 && freq > 1) {
      label <- sub("q", "p", label)
    }
    step <- matrix(as.numeric(times >= at - 1e-8), ncol = 1, dimnames = list(NULL, label))
    det <- cbind(det, step)
  }
  det
}

# A series given beside y -- the skew index, the signals or their regressors --
# on the periods of y, as a plain matrix with named columns. Unnamed columns are
# called after `prefix`.
.align_to_y <- function(x, y, name, prefix = "x") {
  if (!stats::is.ts(x)) {
    stop("Argument '", name, "' must be a time-series object.", call. = FALSE)
  }
  if (is.null(colnames(x))) {
    names_x <- if (NCOL(x) == 1) prefix else paste0(prefix, seq_len(NCOL(x)))
    x <- stats::ts(as.matrix(x), start = stats::start(x), frequency = stats::frequency(x),
                   names = names_x)
  }
  x <- .as_named_ts_matrix(x)
  tsp_y <- stats::tsp(y)
  if (stats::frequency(x) != tsp_y[3]) {
    stop("Arguments 'y' and '", name, "' must have the same frequency.", call. = FALSE)
  }
  # window() warns where x ends early, before the check below says so plainly.
  window_x <- tryCatch(suppressWarnings(stats::window(x, start = tsp_y[1], end = tsp_y[2])),
                       error = function(e) NULL)
  if (is.null(window_x) || nrow(window_x) != nrow(y) || anyNA(window_x)) {
    stop("Argument '", name, "' must cover every period of 'y'.", call. = FALSE)
  }
  matrix(unclass(window_x), nrow(y), dimnames = list(NULL, colnames(x)))
}
