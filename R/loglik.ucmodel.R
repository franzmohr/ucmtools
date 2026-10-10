#' Log Likelihood of an Unobserved Components Model
#'
#' \code{add_posterior_loglik} adds the pointwise log likelihood of each draw,
#' on which WAIC and PSIS-LOO in \code{\link[bayests]{selection_criteria}} are
#' based. \code{add_predictive_loglik} scores a forecast against what its
#' horizon realised.
#'
#' @param object an object of class \code{ucmodel} with posterior draws, and for
#' \code{add_predictive_loglik} with forecasts.
#' @param test_sample a time-series object holding the realised values of the
#' series after the sample, at least for the first period of the horizon.
#' @param ... not used.
#'
#' @details The log likelihood is the integrated one: column \eqn{t} is
#' \eqn{p(y_t | y_1, \dots, y_{t-1}, \theta)}, the one-step prediction error
#' decomposition of a Kalman filter with the trend integrated out, which is what
#' Grant and Chan (2017) compare these models by. A density conditional on the
#' drawn trend would favour any model whose trend is allowed to move. For
#' skewed shocks it conditions on their drawn latent variates.
#'
#' The score runs the same filter on over the realised periods. It is not
#' available for skewed shocks, whose predictive density would need their
#' latent variates integrated out of the filter.
#'
#' @return \code{object} with \code{posterior$loglik}, draws by periods, or with
#' \code{posterior$forecast$loglik}, draws by scored periods.
#'
#' @name ucmodel-loglik
NULL

#' @rdname ucmodel-loglik
#' @export
add_posterior_loglik.ucmodel <- function(object, ...) {
  .require_posterior(object)
  class_of_object <- class(object)
  algorithm <- object[["model"]][["algorithm"]]
  object <- switch(algorithm,
                   UcNormalGamma = .UcNormalGammaLogLik(object),
                   UcNormalWishart = .UcNormalWishartLogLik(object),
                   UcNormalStochvol = .UcNormalStochvolLogLik(object),
                   UcNormalSkew = .UcNormalSkewLogLik(object))
  object[["posterior"]][["loglik"]] <- .mcmc_draws(object[["model"]],
                                                   object[["posterior"]][["loglik"]])
  class(object) <- class_of_object
  object
}

#' @rdname ucmodel-loglik
#' @export
add_predictive_loglik.ucmodel <- function(object, test_sample = NULL, ...) {
  .require_posterior(object)
  algorithm <- object[["model"]][["algorithm"]]
  if (algorithm == "UcNormalSkew") {
    stop("A model with skewed shocks cannot be scored: its predictive density ",
         "integrates the shocks' latent variates out of the filter, which is not ",
         "written.", call. = FALSE)
  }
  h <- object[["model"]][["h"]]
  if (is.null(h)) {
    stop("The model has no forecast horizon. Use add_forecast_input() first.", call. = FALSE)
  }
  if (!is.null(test_sample)) {
    object[["data"]][["test"]] <- list(y = .realised(object, test_sample, h))
  }
  if (is.null(object[["data"]][["test"]][["y"]])) {
    stop("Argument 'test_sample' is missing, and the model carries no realised values.",
         call. = FALSE)
  }
  class_of_object <- class(object)
  object <- .with_stage_seed(object, "predictive", switch(
    algorithm,
    UcNormalGamma = .UcNormalGammaScore(object),
    UcNormalWishart = .UcNormalWishartScore(object),
    UcNormalStochvol = .UcNormalStochvolScore(object)))
  object[["posterior"]][["forecast"]][["loglik"]] <-
    .mcmc_draws(object[["model"]], object[["posterior"]][["forecast"]][["loglik"]])
  class(object) <- class_of_object
  object
}

# The realised values of the forecast periods: the rows of `test_sample` after
# the end of the sample, up to the horizon and to the first missing value.
.realised <- function(object, test_sample, h) {
  if (!stats::is.ts(test_sample)) {
    stop("Argument 'test_sample' must be a time-series object.", call. = FALSE)
  }
  y <- object[["data"]][["y"]]
  tsp_y <- stats::tsp(y)
  test_sample <- .as_named_ts_matrix(test_sample)
  if (!all(colnames(y) %in% colnames(test_sample))) {
    if (ncol(test_sample) != ncol(y)) {
      stop("Argument 'test_sample' must hold the series of the model.", call. = FALSE)
    }
    colnames(test_sample) <- colnames(y)
  }
  start <- tsp_y[2] + 1 / tsp_y[3]
  realised <- tryCatch(stats::window(test_sample[, colnames(y), drop = FALSE], start = start,
                                     end = min(start + (h - 1) / tsp_y[3],
                                               stats::tsp(test_sample)[2])),
                       error = function(e) NULL)
  if (is.null(realised)) {
    stop("Argument 'test_sample' holds no period after the end of the sample.", call. = FALSE)
  }
  realised <- as.matrix(realised)
  complete <- which(!stats::complete.cases(realised))
  if (length(complete) > 0) {
    realised <- realised[seq_len(complete[1] - 1), , drop = FALSE]
  }
  if (nrow(realised) == 0) {
    stop("Argument 'test_sample' holds no complete period after the end of the sample.",
         call. = FALSE)
  }
  unname(realised)
}
