#' Add Priors to an Unobserved Components Model
#'
#' Adds the priors of an unobserved components model, each block filled in with
#' a default where it is not given.
#'
#' @param object an object of class \code{ucmodel}, usually the result of
#' \code{\link{create_ucmodel}}.
#' @param a the normal prior on the cycle coefficients, a list with \code{mu},
#' the prior mean, and \code{v_inv}, the prior precision. Each a single number,
#' which applies to every coefficient, or the full vector and matrix.
#' @param c the normal prior on the coefficients of the drift and its breaks,
#' as \code{a}.
#' @param tau0 the normal prior on the trend before the sample, as \code{a}.
#' \code{mu} defaults to the first observation of each series.
#' @param u_sigma the prior on the shocks: for \code{error = "gamma"} and
#' \code{"gamma+covar"} a list of gamma \code{shape} and \code{rate} on the
#' precision of each of the 2k shocks; for \code{"wishart"} the degrees of
#' freedom \code{df} and the \code{scale} matrix of the inverse Wishart prior on
#' their covariance; for \code{"sv"} and \code{"sv+covar"} the stochastic
#' volatility prior, \code{shape} and \code{rate} of the gamma prior on the
#' precision of the log-volatility innovations (or \code{omega_v}, the prior
#' variance of their standard deviation under the non-centred parameterisation),
#' \code{mu} and \code{v_inv} of the log-volatility before the sample,
#' \code{offset} and \code{sigma}, the starting value of the innovation
#' variance; and for \code{"skew"} the gamma prior of the shock that is not
#' skewed, one per series.
#' @param psi for a covariance block, the normal prior on its free elements, as
#' \code{a}.
#' @param irregular for an irregular component, the gamma \code{shape} and
#' \code{rate} on the precision of each series' irregular shock.
#' @param skew for \code{error = "skew"}, the normal prior on the coefficients
#' of the log scales of the skewed shock, as \code{a}. Per series, a constant
#' and one slope per index variable for each of the two scales.
#' @param rho for \code{trend = "damped"}, the normal prior on the slope's
#' autoregressive coefficient of each series, as \code{a}, truncated to
#' \eqn{(-1, 1)}.
#' @param signal for signal equations, the normal prior on their coefficients,
#' as \code{a}: vec\eqn{(B, \Lambda_0, \dots, \Lambda_q)}, column by column, so
#' for each regressor -- the constant, \code{signal_x}, then each series' cycle
#' at lag 0 to \code{signal_lags} -- its coefficient in every signal.
#' @param signal_noise for signal equations, the gamma \code{shape} and
#' \code{rate} on the precision of each signal's noise.
#' @param ... not used.
#'
#' @details The defaults are weakly informative and set on the scale of the
#' data, the variance \eqn{s_i^2} of the first differences of series \eqn{i}:
#' \itemize{
#'   \item \code{a}: mean zero, precision one;
#'   \item \code{c}: mean zero, precision 0.01;
#'   \item \code{tau0}: mean the first observation, precision 0.01;
#'   \item the shock precisions: gamma with shape 3 and rate \eqn{3 s_i^2}, so
#'   that the prior mean precision is \eqn{1 / s_i^2}, for the cycle and the
#'   trend shock of series \eqn{i} alike; the same for the irregular component
#'   and the shock that is not skewed;
#'   \item \code{"wishart"}: \eqn{2k + 2} degrees of freedom and a diagonal scale
#'   of the \eqn{s_i^2}, which is the prior mean of the covariance;
#'   \item \code{"sv"}: shape 3 and rate 0.01 on the innovation precision of
#'   each log-volatility, the log-volatility before the sample centred on
#'   \eqn{\log s_i^2} with precision 0.1, and an offset of \eqn{10^{-6}};
#'   \item \code{psi}: mean zero, precision one;
#'   \item \code{skew}: the constants centred on \eqn{\log s_i / 2}, the slopes
#'   on zero, precision one;
#'   \item \code{rho}: mean zero, precision one;
#'   \item \code{signal}: the constant of signal \eqn{j} centred on its sample
#'   mean, every other coefficient on zero, precision 0.01;
#'   \item \code{signal_noise}: shape 3 and rate \eqn{3 v_j}, where \eqn{v_j}
#'   is the sample variance of signal \eqn{j}.
#' }
#' These are not the priors of Grant and Chan (2017), which are set for 100 times
#' log US GDP; give \code{a}, \code{u_sigma} and \code{tau0} to reproduce them.
#'
#' @return \code{object} with element \code{priors}.
#'
#' @examples
#' data("at_gdp")
#' model <- create_ucmodel(at_gdp[, "y"] * 100, p = 2)
#' model <- add_priors(model, a = list(mu = c(1.3, -0.7), v_inv = 1))
#' model$priors$a
#'
#' @export
add_priors.ucmodel <- function(object, a = list(), c = list(), tau0 = list(),
                               u_sigma = list(), psi = list(), irregular = list(),
                               skew = list(), rho = list(), signal = list(),
                               signal_noise = list(), ...) {

  model <- object[["model"]]
  y <- object[["data"]][["y"]]
  n_det <- NCOL(object[["data"]][["deterministic"]])
  if (is.null(object[["data"]][["deterministic"]])) n_det <- 0
  n_skew <- if (is.null(object[["data"]][["skew_x"]])) 0 else ncol(object[["data"]][["skew_x"]])
  d <- .uc_dims(model, n_det, n_skew)
  s2 <- .uc_scale(y)
  priors <- list()

  if (d$n_a > 0) {
    priors[["a"]] <- .uc_normal_prior(a, d$n_a, 0, 1, "a")
  }
  if (d$n_c > 0) {
    priors[["c"]] <- .uc_normal_prior(c, d$n_c, 0, 0.01, "c")
  }
  priors[["tau0"]] <- .uc_normal_prior(tau0, d$n_tau0, rep(as.numeric(y[1, ]), d$order),
                                       0.01, "tau0")

  algorithm <- model[["algorithm"]]
  shock_scale <- rep(s2, 2)
  priors[["u_sigma"]] <- switch(
    algorithm,
    UcNormalGamma = .uc_gamma_prior(u_sigma, d$shocks, 3, 3 * shock_scale, "u_sigma"),
    UcNormalWishart = .uc_wishart_prior(u_sigma, d$shocks, shock_scale),
    UcNormalStochvol = .uc_sv_prior(u_sigma, d$shocks, shock_scale),
    UcNormalSkew = .uc_gamma_prior(u_sigma, d$k, 3, 3 * s2, "u_sigma"))

  if (.uc_covar(model)) {
    priors[["psi"]] <- .uc_normal_prior(psi, d$n_psi, 0, 1, "psi")
  }
  if (isTRUE(model[["irregular"]])) {
    priors[["irregular"]] <- .uc_gamma_prior(irregular, d$k, 3, 3 * s2, "irregular")
  }
  if (algorithm == "UcNormalSkew") {
    mu_skew <- unlist(lapply(s2, function(v) {
      rep(c(log(v) / 2, rep(0, n_skew)), 2)
    }))
    priors[["skew"]] <- .uc_normal_prior(skew, d$n_skew_coef, mu_skew, 1, "skew")
  }
  if (identical(model[["trend"]], "damped")) {
    priors[["rho"]] <- .uc_normal_prior(rho, d$k, 0, 1, "rho")
  }
  if (d$n_signals > 0) {
    s <- object[["data"]][["signal"]]
    # vec(B, Lambda_0 .. Lambda_q): n_signals rows per regressor, the constant,
    # where there is one, the first regressor.
    mu_signal <- matrix(0, d$n_signals, d$n_signal_reg)
    signal_x <- object[["data"]][["signal_x"]]
    if (!is.null(signal_x) && identical(colnames(signal_x)[1], "const")) {
      mu_signal[, 1] <- colMeans(s)
    }
    priors[["signal"]] <- .uc_normal_prior(signal, d$n_signal, as.vector(mu_signal), 0.01,
                                           "signal")
    priors[["signal_noise"]] <- .uc_gamma_prior(signal_noise, d$n_signals, 3,
                                                3 * .uc_level_scale(s), "signal_noise")
  }

  object[["priors"]] <- priors
  object
}

# A normal block's prior: `mu` a vector of `n`, `v_inv` an n x n precision. A
# number given for either is spread over the block; the defaults fill in what is
# not given.
.uc_normal_prior <- function(spec, n, mu, v_inv, name) {
  .check_prior_list(spec, name, c("mu", "v_inv"))
  if (!is.null(spec[["mu"]])) mu <- spec[["mu"]]
  if (!is.null(spec[["v_inv"]])) v_inv <- spec[["v_inv"]]
  if (!is.numeric(mu) || anyNA(mu) || !length(mu) %in% c(1, n)) {
    stop("The prior mean of '", name, "' must be a number or a vector of ", n, ".",
         call. = FALSE)
  }
  if (!is.numeric(v_inv) || anyNA(v_inv)) {
    stop("The prior precision of '", name, "' must be numeric.", call. = FALSE)
  }
  v_inv <- if (length(v_inv) == 1) {
    diag(v_inv, n)
  } else if (is.null(dim(v_inv)) && length(v_inv) == n) {
    diag(v_inv, n)
  } else if (all(dim(v_inv) == c(n, n))) {
    as.matrix(v_inv)
  } else {
    stop("The prior precision of '", name, "' must be a number, a vector of ", n,
         " or a ", n, " x ", n, " matrix.", call. = FALSE)
  }
  list(mu = matrix(rep_len(as.numeric(mu), n), ncol = 1), v_inv = v_inv)
}

# Independent gamma priors on `n` precisions.
.uc_gamma_prior <- function(spec, n, shape, rate, name) {
  .check_prior_list(spec, name, c("shape", "rate"))
  if (!is.null(spec[["shape"]])) shape <- spec[["shape"]]
  if (!is.null(spec[["rate"]])) rate <- spec[["rate"]]
  for (i in list(list(shape, "shape"), list(rate, "rate"))) {
    if (!is.numeric(i[[1]]) || anyNA(i[[1]]) || !length(i[[1]]) %in% c(1, n) ||
        any(i[[1]] <= 0)) {
      stop("The gamma prior '", i[[2]], "' of '", name, "' must be positive, one number ",
           "or one per element, ", n, ".", call. = FALSE)
    }
  }
  list(shape = matrix(rep_len(as.numeric(shape), n), ncol = 1),
       rate = matrix(rep_len(as.numeric(rate), n), ncol = 1))
}

.uc_wishart_prior <- function(spec, shocks, shock_scale) {
  .check_prior_list(spec, "u_sigma", c("df", "scale"))
  df <- if (is.null(spec[["df"]])) shocks + 2 else spec[["df"]]
  .check_whole_number(df, "u_sigma$df", 1)
  scale <- spec[["scale"]]
  if (is.null(scale)) {
    scale <- diag(shock_scale * max(df - shocks - 1, 1), shocks)
  } else if (length(scale) == 1) {
    scale <- diag(scale, shocks)
  }
  if (!is.numeric(scale) || !all(dim(as.matrix(scale)) == c(shocks, shocks))) {
    stop("The Wishart scale of 'u_sigma' must be a number or a ", shocks, " x ", shocks,
         " matrix.", call. = FALSE)
  }
  list(df = as.integer(df), scale = as.matrix(scale))
}

.uc_sv_prior <- function(spec, shocks, shock_scale) {
  .check_prior_list(spec, "u_sigma", c("shape", "rate", "omega_v", "mu", "v_inv",
                                       "offset", "sigma"))
  spread <- function(x, name) {
    if (!is.numeric(x) || anyNA(x) || !length(x) %in% c(1, shocks)) {
      stop("Element '", name, "' of 'u_sigma' must be a number or a vector of ", shocks,
           ".", call. = FALSE)
    }
    matrix(rep_len(as.numeric(x), shocks), ncol = 1)
  }
  result <- list(offset = spread(if (is.null(spec[["offset"]])) 1e-6 else spec[["offset"]],
                                 "offset"))
  if (is.null(spec[["omega_v"]])) {
    result[["shape"]] <- spread(if (is.null(spec[["shape"]])) 3 else spec[["shape"]], "shape")
    result[["rate"]] <- spread(if (is.null(spec[["rate"]])) 0.01 else spec[["rate"]], "rate")
  } else {
    result[["omega_v"]] <- spread(spec[["omega_v"]], "omega_v")
  }
  result[["mu"]] <- spread(if (is.null(spec[["mu"]])) log(shock_scale) else spec[["mu"]], "mu")
  v_inv <- if (is.null(spec[["v_inv"]])) 0.1 else spec[["v_inv"]]
  result[["v_inv"]] <- if (length(v_inv) == 1) diag(v_inv, shocks) else as.matrix(v_inv)
  result[["sigma"]] <- spread(if (is.null(spec[["sigma"]])) 0.05 else spec[["sigma"]], "sigma")
  result
}

.check_prior_list <- function(spec, name, allowed) {
  if (!is.list(spec)) {
    stop("Argument '", name, "' must be a list.", call. = FALSE)
  }
  unknown <- setdiff(names(spec), allowed)
  if (length(unknown) > 0) {
    stop("Argument '", name, "' has no element ", paste0("'", unknown, "'", collapse = ", "),
         "; it takes ", paste0("'", allowed, "'", collapse = ", "), ".", call. = FALSE)
  }
  invisible(spec)
}
