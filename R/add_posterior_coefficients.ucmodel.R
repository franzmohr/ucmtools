#' Posterior Simulation of an Unobserved Components Model
#'
#' Draws the posterior of an unobserved components model with the Gibbs sampler
#' of BayesTS its error specification names.
#'
#' @param object an object of class \code{ucmodel} with priors and initial
#' values.
#' @param posterior_function an optional function that takes \code{object} and
#' returns it with its posterior, such as the one
#' \code{\link[bayests]{bayests_posterior}} returns, which runs the BayesTS
#' executable instead.
#' @param verbose logical. Should the sampler report its progress?
#' @param ... not used.
#'
#' @details The internal sampler draws with the seed in \code{model$seed}, see
#' \code{\link[bayests]{add_seed}}: R's generator is set to it for the
#' simulation and put back afterwards, so a model with a seed gives the same
#' draws however R's generator stands. A model without one draws from R's
#' generator as it stands.
#'
#' Element \code{posterior} of the result holds one \code{\link[coda]{mcmc}}
#' object per block under \code{coeffs}, one row per draw:
#' \describe{
#'   \item{\code{trend}}{the trend path, \eqn{k T} columns, vec of the
#'   \eqn{k \times T}{k x T} path, so period by period; see \code{\link{trend}};}
#'   \item{\code{cycle}}{with an irregular component, the cycle path, laid out as
#'   the trend. Without one the cycle is \eqn{y_t - \tau_t} and not stored;}
#'   \item{\code{a}}{the cycle coefficients, vec\eqn{(\Phi_1, \dots, \Phi_p)}, the
#'   whole \eqn{k \times kp}{k x kp} matrix;}
#'   \item{\code{c}}{the drift and its breaks, vec\eqn{(C)};}
#'   \item{\code{tau0}}{the trend before the sample;}
#'   \item{\code{u_omega_inv}, \code{u_sigma_inv}, \code{psi}}{the precision of the
#'   shocks, its diagonal after orthogonalisation and the covariance block, as
#'   the error specification has them; under stochastic volatility one
#'   precision per period, with \code{sigma}, the variance of the log-volatility
#'   innovations;}
#'   \item{\code{irregular_inv}}{the precisions of the irregular component;}
#'   \item{\code{skew}, \code{skew_latent}}{the coefficients of the log scales of
#'   the skewed shock and its latent variates;}
#'   \item{\code{cycle0}}{under \code{cycle_init = "stationary"}, the cycle
#'   before the sample, \eqn{(c_0', \dots, c_{1-p}')'};}
#'   \item{\code{rho}}{under \code{trend = "damped"}, the slope's
#'   autoregressive coefficient of each series;}
#'   \item{\code{signal}, \code{signal_noise_inv}}{with signals, the
#'   coefficients of their equations, vec\eqn{(B, \Lambda_0, \dots, \Lambda_q)},
#'   named \code{<signal>.<regressor>}, and the precisions of their noise.}
#' }
#'
#' @return \code{object} with element \code{posterior}, or with
#' \code{error = TRUE} where the simulation failed.
#'
#' @export
add_posterior_coefficients.ucmodel <- function(object, posterior_function = NULL,
                                               verbose = FALSE, ...) {

  object[["model"]][["verbose"]] <- if (.check_flag(verbose, "verbose")) TRUE
  class_of_object <- class(object)
  object[["posterior"]] <- NULL
  object[["error"]] <- NULL
  model <- object

  if (is.null(posterior_function)) {
    if (is.null(object[["priors"]]) || is.null(object[["initial"]])) {
      stop("Argument 'object' needs priors and initial values. Use add_priors() and ",
           "add_initial_values() first.", call. = FALSE)
    }
    object <- try({
      algorithm <- object[["model"]][["algorithm"]]
      object <- .with_model_seed(object[["model"]][["seed"]], switch(
        algorithm,
        UcNormalGamma = .UcNormalGammaCoefficients(object),
        UcNormalWishart = .UcNormalWishartCoefficients(object),
        UcNormalStochvol = .UcNormalStochvolCoefficients(object),
        UcNormalSkew = .UcNormalSkewCoefficients(object),
        stop("Algorithm '", algorithm, "' not supported.")))
      .label_draws(object)
    })
  } else {
    object <- try(posterior_function(object))
  }

  if (inherits(object, "try-error")) {
    object <- c(model, list(error = TRUE))
  }
  class(object) <- class_of_object
  object
}

# Every block of the posterior as mcmc objects labelled with the draws they are,
# and the columns of the small blocks named.
.label_draws <- function(object) {
  model <- object[["model"]]
  posterior <- object[["posterior"]]
  y <- object[["data"]][["y"]]
  series <- colnames(y)
  det <- object[["model"]][["deterministic"]]

  if (!is.null(posterior[["c"]][["coeffs"]])) {
    colnames(posterior[["c"]][["coeffs"]]) <- paste0(rep(series, length(det)), ".",
                                                     rep(det, each = length(series)))
  }
  if (!is.null(posterior[["tau0"]][["coeffs"]])) {
    order <- ncol(posterior[["tau0"]][["coeffs"]]) / length(series)
    colnames(posterior[["tau0"]][["coeffs"]]) <- paste0(rep(series, order), ".tau",
                                                        rep(-(seq_len(order) - 1), each = length(series)))
  }
  if (!is.null(posterior[["cycle0"]][["coeffs"]])) {
    p0 <- ncol(posterior[["cycle0"]][["coeffs"]]) / length(series)
    colnames(posterior[["cycle0"]][["coeffs"]]) <- paste0(rep(series, p0), ".c",
                                                          rep(-(seq_len(p0) - 1), each = length(series)))
  }
  if (!is.null(posterior[["rho"]][["coeffs"]])) {
    colnames(posterior[["rho"]][["coeffs"]]) <- paste0(series, ".rho")
  }
  if (!is.null(posterior[["signal"]][["coeffs"]])) {
    # A model read back from a file may have lost the column names.
    signals <- colnames(object[["data"]][["signal"]])
    if (is.null(signals)) signals <- paste0("s", seq_len(model[["n_signals"]]))
    signal_x <- colnames(object[["data"]][["signal_x"]])
    if (is.null(signal_x) && model[["n_signal_x"]] > 0) {
      signal_x <- paste0("x", seq_len(model[["n_signal_x"]]))
    }
    regressors <- c(signal_x,
                    paste0(rep(series, model[["signal_lags"]] + 1), ".l",
                           rep(seq(0, model[["signal_lags"]]), each = length(series))))
    colnames(posterior[["signal"]][["coeffs"]]) <-
      paste0(rep(signals, length(regressors)), ".", rep(regressors, each = length(signals)))
    colnames(posterior[["signal_noise_inv"]][["coeffs"]]) <- signals
  }
  if (!is.null(posterior[["a"]][["coeffs"]])) {
    k <- length(series)
    p <- model[["p"]]
    colnames(posterior[["a"]][["coeffs"]]) <- paste0(
      rep(series, k * p), ".", rep(paste0(rep(series, p), ".l", rep(seq_len(p), each = k)),
                                   each = k))
  }

  for (i in names(posterior)) {
    if (i == "forecast") next
    block <- posterior[[i]]
    if (is.list(block)) {
      for (j in names(block)) {
        block[[j]] <- .mcmc_draws(model, block[[j]])
      }
      posterior[[i]] <- block
    }
  }
  object[["posterior"]] <- posterior
  object
}
