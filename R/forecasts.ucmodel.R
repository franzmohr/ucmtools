#' Forecasts of an Unobserved Components Model
#'
#' \code{add_forecast_input} sets the horizon of a forecast and the values over
#' it of what the model takes from outside: the drift's terms and, for skewed
#' shocks with moving scales, the index. \code{add_posterior_forecasts}
#' simulates one forecast path per posterior draw, and \code{predict} summarises
#' them.
#'
#' @param object an object of class \code{ucmodel}.
#' @param n_ahead the forecast horizon.
#' @param skew_x for a model whose skewed shock's scales move with an index, its
#' values over the horizon, a matrix or time-series object of \code{n_ahead}
#' rows, or a time-series object that covers them.
#' @param forecast_states for stochastic volatility, \code{"simulate"} to carry
#' the log-volatilities forward along their random walks, or \code{"hold"} to
#' keep them at their values in the last period.
#' @param ... not used.
#'
#' @details The drift and its breaks continue at their values in the last
#' period: a constant stays one, a break that has happened stays in effect.
#'
#' The forecast draws with a seed derived from the model's own, see
#' \code{\link[bayests]{add_seed}}, so two forecasts of a model with a seed are
#' the same. A model without one draws from R's generator as it stands.
#'
#' @return \code{add_forecast_input} and \code{add_posterior_forecasts} return
#' \code{object}, the latter with \code{posterior$forecast$forecasts}, one row
#' per draw and \eqn{k h} columns stacked by period. \code{predict} returns an
#' object of class \code{bvarprd}, which bayests plots.
#'
#' @examples
#' data("at_gdp")
#' model <- create_ucmodel(at_gdp[, "y"] * 100, p = 2, iterations = 200, burnin = 100)
#' model <- add_initial_values(add_priors(model))
#' model <- add_posterior_coefficients(model)
#' model <- add_forecast_input(model, n_ahead = 8)
#' model <- add_posterior_forecasts(model)
#' plot(predict(model))
#'
#' @name ucmodel-forecasts
NULL

#' @rdname ucmodel-forecasts
#' @export
prepare_forecast_input.ucmodel <- function(object, n_ahead = 8, skew_x = NULL, ...) {
  .check_whole_number(n_ahead, "n_ahead", 1)
  result <- list(h = as.integer(n_ahead))
  det <- object[["data"]][["deterministic"]]
  if (!is.null(det)) {
    result[["x"]] <- matrix(det[nrow(det), ], n_ahead, ncol(det), byrow = TRUE,
                            dimnames = list(NULL, colnames(det)))
  }
  train_skew <- object[["data"]][["skew_x"]]
  if (!is.null(train_skew)) {
    if (is.null(skew_x)) {
      stop("This model's skewed shock moves with an index; give its values over the ",
           "horizon in argument 'skew_x'.", call. = FALSE)
    }
    if (stats::is.ts(skew_x)) {
      tsp_y <- stats::tsp(object[["data"]][["y"]])
      start <- tsp_y[2] + 1 / tsp_y[3]
      skew_x <- tryCatch(stats::window(skew_x, start = start,
                                       end = start + (n_ahead - 1) / tsp_y[3]),
                         error = function(e) NULL)
    }
    skew_x <- as.matrix(skew_x)
    if (is.null(skew_x) || nrow(skew_x) != n_ahead || ncol(skew_x) != ncol(train_skew) ||
        anyNA(skew_x)) {
      stop("Argument 'skew_x' must hold ", ncol(train_skew), " column(s) over the ",
           n_ahead, " periods of the horizon.", call. = FALSE)
    }
    result[["skew_x"]] <- matrix(as.numeric(skew_x), n_ahead,
                                 dimnames = list(NULL, colnames(train_skew)))
  }
  result
}

#' @rdname ucmodel-forecasts
#' @export
add_forecast_input.ucmodel <- function(object, n_ahead = 8, skew_x = NULL, ...) {
  input <- prepare_forecast_input(object, n_ahead = n_ahead, skew_x = skew_x)
  object[["model"]][["h"]] <- input[["h"]]
  input[["h"]] <- NULL
  object[["data"]][["forecast"]] <- if (length(input) > 0) input
  object
}

#' @rdname ucmodel-forecasts
#' @export
add_posterior_forecasts.ucmodel <- function(object, forecast_states = NULL, ...) {
  .require_posterior(object)
  if (is.null(object[["model"]][["h"]])) {
    stop("The model has no forecast horizon. Use add_forecast_input() first.", call. = FALSE)
  }
  if (!is.null(forecast_states)) {
    object[["model"]][["forecast_states"]] <- .check_choice(forecast_states, "forecast_states",
                                                            c("simulate", "hold"))
  }
  class_of_object <- class(object)
  algorithm <- object[["model"]][["algorithm"]]
  object <- .with_stage_seed(object, "forecast", switch(
    algorithm,
    UcNormalGamma = .UcNormalGammaForecasts(object),
    UcNormalWishart = .UcNormalWishartForecasts(object),
    UcNormalStochvol = .UcNormalStochvolForecasts(object),
    UcNormalSkew = .UcNormalSkewForecasts(object)))
  object[["posterior"]][["forecast"]][["forecasts"]] <-
    .mcmc_draws(object[["model"]], object[["posterior"]][["forecast"]][["forecasts"]])
  class(object) <- class_of_object
  object
}

#' @rdname ucmodel-forecasts
#' @export
predict.ucmodel <- function(object, n_ahead = NULL, ...) {
  h <- object[["model"]][["h"]]
  forecasts <- object[["posterior"]][["forecast"]][["forecasts"]]
  if (is.null(h) || is.null(forecasts)) {
    stop("The model has no forecasts. Use add_forecast_input() and ",
         "add_posterior_forecasts() first.", call. = FALSE)
  }
  if (is.null(n_ahead)) {
    n_ahead <- h
  }
  .check_whole_number(n_ahead, "n_ahead", 1)
  if (n_ahead > h) {
    warning("Argument 'n_ahead' is larger than the horizon of the forecast, ", h,
            "; limiting the output to it.", call. = FALSE)
    n_ahead <- h
  }
  y <- object[["data"]][["y"]]
  tsp_y <- stats::tsp(y)
  k <- ncol(y)
  forecasts <- unclass(as.matrix(forecasts))
  draws <- nrow(forecasts)
  times <- tsp_y[2] + seq_len(n_ahead) / tsp_y[3]
  result <- array(NA_real_, dim = c(n_ahead, k, draws),
                  dimnames = list(as.character(times), colnames(y), NULL))
  for (i in seq_len(draws)) {
    result[, , i] <- t(matrix(forecasts[i, ], k)[, seq_len(n_ahead), drop = FALSE])
  }
  attr(result, "tsp") <- c(min(times), max(times), tsp_y[3])
  structure(list(fcst = result, y = y), class = c("bvarprd", "list"))
}

.require_posterior <- function(object) {
  if (is.null(object[["posterior"]][["trend"]][["coeffs"]])) {
    stop("Argument 'object' has no posterior draws of the trend. Use ",
         "add_posterior_coefficients() first.", call. = FALSE)
  }
  invisible(object)
}
