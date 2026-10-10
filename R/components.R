#' Seed of the Posterior Simulation of an Unobserved Components Model
#'
#' The method of \code{\link[bayests]{add_seed}} for unobserved components
#' models: it sets the seed of the posterior simulation in \code{model$seed}.
#' The forecast and the score draw with seeds derived from it.
#'
#' @param object an object of class \code{ucmodel}.
#' @param seed a non-negative whole number no larger than
#' \code{.Machine$integer.max}.
#' @param ... not used.
#'
#' @return \code{object} with the seed set.
#'
#' @export
add_seed.ucmodel <- function(object, seed, ...) {
  object[["model"]][["seed"]] <- .check_seed(seed)
  object
}

#' Trend and Cycle of an Unobserved Components Model
#'
#' The posterior of the trend or the cycle of each series, summarised period by
#' period, or all of its draws.
#'
#' @param object an object of class \code{ucmodel} with posterior draws.
#' @param ci the probability mass of the credible band, 0.9 by default.
#' @param draws logical. Return every draw rather than a summary?
#' @param ... not used.
#'
#' @details Without an irregular component the cycle is \eqn{y_t - \tau_t},
#' draw by draw; with one it is a drawn state of its own.
#'
#' @return For \code{draws = FALSE}, a time-series object per series with
#' columns \code{lower}, \code{median} and \code{upper} -- a list of them for
#' more than one series, named after the series. For \code{draws = TRUE}, an
#' array of periods by series by draws.
#'
#' @examples
#' data("at_gdp")
#' model <- create_ucmodel(at_gdp[, "y"] * 100, p = 2, iterations = 200, burnin = 100)
#' model <- add_posterior_coefficients(add_initial_values(add_priors(model)))
#' head(trend(model))
#' head(cycle(model))
#'
#' @export
trend <- function(object, ...) {
  UseMethod("trend")
}

#' @rdname trend
#' @export
cycle <- function(object, ...) {
  UseMethod("cycle")
}

#' @rdname trend
#' @export
trend.ucmodel <- function(object, ci = 0.9, draws = FALSE, ...) {
  .summarise_component(object, .component_draws(object, "trend"), ci, draws)
}

#' @rdname trend
#' @export
cycle.ucmodel <- function(object, ci = 0.9, draws = FALSE, ...) {
  .summarise_component(object, .component_draws(object, "cycle"), ci, draws)
}

#' @rdname trend
#' @export
cycle.default <- function(object, ...) {
  stats::cycle(object, ...)
}

# A component's draws as an array of periods x series x draws.
.component_draws <- function(object, component) {
  .require_posterior(object)
  y <- unclass(as.matrix(object[["data"]][["y"]]))
  tt <- nrow(y)
  k <- ncol(y)
  as_array <- function(block) {
    block <- unclass(as.matrix(block))
    result <- array(NA_real_, dim = c(tt, k, nrow(block)))
    for (i in seq_len(nrow(block))) {
      result[, , i] <- t(matrix(block[i, ], k, tt))
    }
    result
  }
  trend <- as_array(object[["posterior"]][["trend"]][["coeffs"]])
  if (component == "trend") {
    return(trend)
  }
  cycle <- object[["posterior"]][["cycle"]][["coeffs"]]
  if (!is.null(cycle)) {
    return(as_array(cycle))
  }
  sweep(-trend, c(1, 2), y, "+")
}

.summarise_component <- function(object, values, ci, draws) {
  y <- object[["data"]][["y"]]
  tsp_y <- stats::tsp(y)
  dimnames(values) <- list(NULL, colnames(y), NULL)
  if (.check_flag(draws, "draws")) {
    return(values)
  }
  if (!is.numeric(ci) || length(ci) != 1 || ci <= 0 || ci >= 1) {
    stop("Argument 'ci' must be a single number between 0 and 1.", call. = FALSE)
  }
  probs <- c((1 - ci) / 2, 0.5, 1 - (1 - ci) / 2)
  result <- lapply(seq_len(ncol(y)), function(i) {
    bands <- t(apply(values[, i, , drop = FALSE], 1, stats::quantile, probs = probs))
    stats::ts(bands, start = tsp_y[1], frequency = tsp_y[3],
              names = c("lower", "median", "upper"))
  })
  names(result) <- colnames(y)
  if (length(result) == 1) result[[1]] else result
}

#' Plot an Unobserved Components Model
#'
#' Plots, for each series, the data with the posterior median of the trend and
#' its credible band, and below it the cycle with its band.
#'
#' @param x an object of class \code{ucmodel} with posterior draws.
#' @param ci the probability mass of the credible bands.
#' @param ... further arguments passed to \code{\link[graphics]{plot}}.
#'
#' @return \code{x}, invisibly.
#'
#' @export
plot.ucmodel <- function(x, ci = 0.9, ...) {
  trend_bands <- trend(x, ci = ci)
  cycle_bands <- cycle(x, ci = ci)
  y <- x[["data"]][["y"]]
  series <- colnames(y)
  if (length(series) == 1) {
    trend_bands <- list(trend_bands)
    cycle_bands <- list(cycle_bands)
  }
  old <- graphics::par(mfrow = c(2, length(series)), mar = c(4, 4, 2, 1))
  on.exit(graphics::par(old), add = TRUE)
  draw <- function(bands, main, data = NULL) {
    range_y <- range(c(bands, data), na.rm = TRUE)
    graphics::plot(bands[, "median"], ylim = range_y, main = main, ylab = "",
                   xlab = "", type = "n", ...)
    times <- as.numeric(stats::time(bands))
    graphics::polygon(c(times, rev(times)), c(bands[, "lower"], rev(bands[, "upper"])),
                      col = "grey85", border = NA)
    if (!is.null(data)) {
      graphics::lines(data, col = "grey40")
    } else {
      graphics::abline(h = 0, lty = "dotted")
    }
    graphics::lines(bands[, "median"], lwd = 2)
  }
  for (i in seq_along(series)) {
    draw(trend_bands[[i]], paste0(series[i], ": trend"), y[, i])
  }
  for (i in seq_along(series)) {
    draw(cycle_bands[[i]], paste0(series[i], ": cycle"))
  }
  invisible(x)
}

#' @export
print.ucmodel <- function(x, ...) {
  model <- x[["model"]]
  cat("Unobserved components model (", model[["algorithm"]], ")\n", sep = "")
  cat("  Series:     ", paste(colnames(x[["data"]][["y"]]), collapse = ", "), "\n", sep = "")
  cat("  Trend:      ", model[["trend"]],
      if (length(model[["deterministic"]]) > 0)
        paste0(", drift on ", paste(model[["deterministic"]], collapse = ", ")),
      "\n", sep = "")
  cat("  Cycle:      ", if (model[["p"]] == 0) "white noise" else
    paste0(model[["cycle"]], "(", model[["p"]], ")"), "\n", sep = "")
  if (isTRUE(model[["irregular"]])) cat("  Irregular:  yes\n")
  if (!is.null(x[["data"]][["signal"]])) {
    cat("  Signals:    ", paste(colnames(x[["data"]][["signal"]]), collapse = ", "),
        " on the cycle at lags 0 to ", model[["signal_lags"]], "\n", sep = "")
  }
  cat("  Error:      ", model[["error"]], "\n", sep = "")
  cat("  Draws:      ", model[["iterations"]], " after a burn-in of ", model[["burnin"]],
      if (!is.null(model[["thin"]])) paste0(", thinned by ", model[["thin"]]), "\n", sep = "")
  status <- if (isTRUE(x[["error"]])) "failed" else if (is.null(x[["posterior"]])) "not yet" else "done"
  cat("  Simulated:  ", status, "\n", sep = "")
  invisible(x)
}
