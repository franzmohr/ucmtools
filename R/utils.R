# Checks and conversions shared by the methods of this package. The first four
# are dfmtools' own, which this package cannot reach, since they are not
# exported there.

# A vector of whole numbers at or above `minimum`: `p` of create_ucmodel(),
# which indexes a model rather than measuring anything.
.check_model_integer <- function(value, name, minimum) {
  if (!is.numeric(value) || length(value) == 0 || anyNA(value) ||
      any(!is.finite(value)) || any(value != round(value))) {
    stop("Argument '", name, "' must be a vector of whole numbers.", call. = FALSE)
  }
  if (any(value < minimum)) {
    stop("Argument '", name, "' must be at least ", minimum, ".", call. = FALSE)
  }
  invisible(value)
}

# A single whole number at or above `minimum`.
.check_whole_number <- function(value, name, minimum) {
  if (!is.numeric(value) || length(value) != 1 || is.na(value) ||
      !is.finite(value) || value != round(value) || value < minimum) {
    stop("Argument '", name, "' must be a single whole number of at least ",
         minimum, ".", call. = FALSE)
  }
  invisible(value)
}

# The thinning interval, or NULL for one, which is how a model that keeps every
# draw says so.
.check_sampler_thin <- function(thin) {
  .check_whole_number(thin, "thin", 1)
  if (thin == 1) {
    return(NULL)
  }
  as.integer(thin)
}

# A block's draws as an mcmc object whose labels say which draws were kept:
# BayesTS keeps the last of every `thin` draws after the burn-in.
.mcmc_draws <- function(model, draws) {
  thin <- model[["thin"]]
  if (is.null(thin) || thin == 1) {
    return(coda::as.mcmc(draws))
  }
  thin <- as.numeric(thin)
  coda::mcmc(draws, start = thin, end = NROW(draws) * thin, thin = thin)
}

.check_flag <- function(value, name) {
  if (!is.logical(value) || length(value) != 1 || is.na(value)) {
    stop("Argument '", name, "' must be TRUE or FALSE.", call. = FALSE)
  }
  value
}

# One of `choices`, by name.
.check_choice <- function(value, name, choices) {
  if (!is.character(value) || length(value) != 1 || !value %in% choices) {
    stop("Argument '", name, "' must be one of ",
         paste0("\"", choices, "\"", collapse = ", "), ".", call. = FALSE)
  }
  value
}

# The four samplers and the error specifications that pick them.
.uc_algorithms <- c("gamma" = "UcNormalGamma", "gamma+covar" = "UcNormalGamma",
                    "wishart" = "UcNormalWishart", "sv" = "UcNormalStochvol",
                    "sv+covar" = "UcNormalStochvol", "skew" = "UcNormalSkew")

# Dimensions the core derives from the specification, written out once here so
# that the priors, the starting values and the accessors agree on them.
.uc_dims <- function(model, n_det = 0, n_skew = 0) {
  k <- model[["k"]]
  p <- model[["p"]]
  order <- if (identical(model[["trend"]], "rw") || is.null(model[["trend"]])) 1 else 2
  n_signals <- if (is.null(model[["n_signals"]])) 0 else model[["n_signals"]]
  n_signal_reg <- if (n_signals > 0) {
    model[["n_signal_x"]] + k * (model[["signal_lags"]] + 1)
  } else {
    0
  }
  list(k = k, p = p, n = n_det, order = order,
       n_signals = n_signals, n_signal_reg = n_signal_reg,
       n_signal = n_signals * n_signal_reg,
       n_a = if (p == 0) 0 else if (identical(model[["cycle"]], "own")) k * p else k * k * p,
       n_c = k * n_det,
       n_tau0 = k * order,
       shocks = 2 * k,
       n_psi = (2 * k) * (2 * k - 1) / 2,
       n_skew_coef = k * 2 * (1 + n_skew))
}

# The variance of each series' first differences: the scale the default priors
# and starting values are set on, so that they suit data in logs and in levels
# alike. One where a series has no variation to measure.
.uc_scale <- function(y) {
  y <- as.matrix(y)
  s2 <- apply(y, 2, function(x) stats::var(diff(x), na.rm = TRUE))
  s2[!is.finite(s2) | s2 <= 0] <- 1
  unname(s2)
}

# The variance of each column in levels, for series that do not trend: the
# signals, whose noise priors and starts are set on it. One where a column has no
# variation.
.uc_level_scale <- function(x) {
  v <- apply(as.matrix(x), 2, stats::var)
  v[!is.finite(v) | v <= 0] <- 1
  unname(v)
}

# Whether a model carries a covariance block over its shocks.
.uc_covar <- function(model) {
  grepl("\\+covar$", model[["error"]])
}
