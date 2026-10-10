# The R layer of the four samplers: what a model is set up as, what each step
# adds, what is refused, and that a seeded model's draws repeat.

test_that("create_ucmodel() names the sampler its error specification picks", {
  y <- uc_data()[, "y"]
  algorithms <- c("gamma" = "UcNormalGamma", "gamma+covar" = "UcNormalGamma",
                  "wishart" = "UcNormalWishart", "sv" = "UcNormalStochvol",
                  "sv+covar" = "UcNormalStochvol", "skew" = "UcNormalSkew")
  for (error in names(algorithms)) {
    model <- create_ucmodel(y, error = error)
    expect_s3_class(model, "ucmodel")
    expect_identical(model$model$algorithm, algorithms[[error]])
  }
  expect_s3_class(create_ucmodel(y, p = 0:2), "modellist")
  expect_length(create_ucmodel(y, p = 0:2), 3)
})

test_that("the drift and its breaks are the trend's deterministic terms", {
  y <- uc_data()[, "y"]
  model <- create_ucmodel(y, breaks = list(c(2008, 3), 2015))
  det <- model$data$deterministic
  expect_identical(colnames(det), c("const", "break2008q3", "break2015q1"))
  expect_identical(det[, "break2008q3"],
                   as.numeric(stats::time(y) >= 2008.5 - 1e-8))
  expect_null(create_ucmodel(y, trend = "smooth")$data$deterministic)
  expect_null(create_ucmodel(y, drift = FALSE)$data$deterministic)
})

test_that("what cannot be estimated is refused where it is specified", {
  y <- uc_data()[, "y"]
  expect_error(create_ucmodel(as.numeric(y)), "time-series")
  expect_error(create_ucmodel(y, error = "normal"), "error")
  expect_error(create_ucmodel(y, p = 0, irregular = TRUE), "irregular")
  expect_error(create_ucmodel(y, error = "sv", stationary = TRUE), "stationary")
  expect_error(create_ucmodel(y, drift = FALSE, breaks = list(2008)), "drift")
  expect_error(create_ucmodel(y, breaks = list(1950)), "break")
  expect_error(create_ucmodel(y, skew_x = y), "skew")
  expect_error(add_priors(create_ucmodel(y), a = list(scale = 1)), "no element")
  expect_error(add_initial_values(create_ucmodel(y)), "add_priors")
})

test_that("the priors and starting values have the sizes the samplers expect", {
  y <- uc_data()
  model <- add_initial_values(add_priors(create_ucmodel(y, p = 2, error = "gamma+covar",
                                                        irregular = TRUE)))
  k <- 2L
  expect_identical(dim(model$priors$a$v_inv), c(k * k * 2L, k * k * 2L))
  expect_identical(dim(model$priors$tau0$mu), c(k, 1L))
  expect_identical(dim(model$priors$u_sigma$shape), c(2L * k, 1L))
  expect_identical(dim(model$priors$psi$v_inv), c(6L, 6L))
  expect_identical(dim(model$initial$irregular_inv), c(k, 1L))
  expect_type(model$model$seed, "integer")

  own <- add_priors(create_ucmodel(y, p = 2, cycle = "own"))
  expect_identical(dim(own$priors$a$v_inv), c(k * 2L, k * 2L))

  smooth <- add_priors(create_ucmodel(y[, "y"], trend = "smooth"))
  expect_identical(dim(smooth$priors$tau0$mu), c(2L, 1L))
})

test_that("every sampler returns the trend path and the blocks of its model", {
  y <- uc_data()[, "y"]
  tt <- length(y)
  cases <- list(list(error = "gamma"), list(error = "gamma+covar"),
                list(error = "wishart"), list(error = "sv"),
                list(error = "skew"), list(error = "gamma", irregular = TRUE))
  for (case in cases) {
    fit <- do.call(uc_fit, c(case, list(p = 2)))
    label <- paste(unlist(case), collapse = " ")
    expect_null(fit$error, label = label)
    expect_identical(dim(fit$posterior$trend$coeffs), c(60L, tt), label = label)
    expect_s3_class(fit$posterior$trend$coeffs, "mcmc")
    expect_identical(colnames(fit$posterior$a$coeffs), c("y.y.l1", "y.y.l2"))
    expect_identical(colnames(fit$posterior$c$coeffs), "y.const")
  }
  expect_identical(dim(uc_fit(p = 2, irregular = TRUE)$posterior$cycle$coeffs), c(60L, tt))
  expect_null(uc_fit(p = 2)$posterior$cycle)
  expect_false(is.null(uc_fit(p = 2, error = "sv")$posterior$u_sigma_inv$sigma))
})

test_that("trend() and cycle() add up to the data without an irregular component", {
  fit <- uc_fit(p = 2)
  values <- trend(fit, draws = TRUE) + cycle(fit, draws = TRUE)
  expect_equal(values[, 1, 1], as.numeric(uc_data()[, "y"]))

  bands <- trend(fit, ci = 0.68)
  expect_s3_class(bands, "ts")
  expect_identical(colnames(bands), c("lower", "median", "upper"))
  expect_true(all(bands[, "lower"] <= bands[, "upper"]))
  expect_named(cycle(uc_fit(y = uc_data(), p = 1)), c("y", "y.s"))
  expect_identical(cycle(stats::ts(1:8, frequency = 4)), stats::cycle(stats::ts(1:8, frequency = 4)))
})

test_that("a seeded model draws the same posterior and forecast however R's generator stands", {
  model <- create_ucmodel(uc_data()[, "y"], p = 2, iterations = 50, burnin = 30)
  model <- add_seed(add_initial_values(add_priors(model)), 42)
  set.seed(1)
  state <- get(".Random.seed", envir = globalenv())
  one <- add_posterior_coefficients(model)
  expect_identical(get(".Random.seed", envir = globalenv()), state)
  set.seed(2)
  two <- add_posterior_coefficients(model)
  expect_identical(as.matrix(one$posterior$trend$coeffs), as.matrix(two$posterior$trend$coeffs))

  one <- add_forecast_input(one, n_ahead = 4)
  forecast <- function(state) {
    set.seed(state)
    add_posterior_forecasts(one)$posterior$forecast$forecasts
  }
  expect_identical(forecast(1), forecast(2))
  # The forecast does not reuse the stream the chain started from.
  expect_false(identical(add_seed(one, 43)$model$seed, one$model$seed))
})

test_that("a forecast has one row per draw and k columns per period", {
  fit <- add_forecast_input(uc_fit(y = uc_data(), p = 1), n_ahead = 3)
  expect_identical(dim(fit$data$forecast$x), c(3L, 1L))
  fit <- add_posterior_forecasts(fit)
  expect_identical(dim(fit$posterior$forecast$forecasts), c(60L, 6L))
  prd <- predict(fit)
  expect_s3_class(prd, "bvarprd")
  expect_identical(dim(prd$fcst), c(3L, 2L, 60L))
  expect_warning(predict(fit, n_ahead = 5), "horizon")
  expect_error(add_posterior_forecasts(uc_fit(p = 1)), "add_forecast_input")
})

test_that("a skewed shock whose scale moves needs the index over the horizon", {
  y <- uc_data()[, "y"]
  index <- stats::ts(seq(-1, 1, length.out = length(y) + 4),
                     start = stats::start(y), frequency = 4)
  fit <- uc_fit(p = 2, error = "skew", skew = "ald", skew_x = index)
  expect_identical(ncol(fit$posterior$skew$coeffs), 4L)
  expect_error(add_forecast_input(fit, n_ahead = 4), "skew_x")
  fit <- add_posterior_forecasts(add_forecast_input(fit, n_ahead = 4, skew_x = index))
  expect_identical(dim(fit$posterior$forecast$forecasts), c(60L, 4L))
  expect_error(add_predictive_loglik(fit, test_sample = y), "skewed")
})

test_that("the log likelihood is draws by periods and reaches selection_criteria()", {
  fit <- add_posterior_loglik(uc_fit(p = 2))
  expect_identical(dim(fit$posterior$loglik), c(60L, length(uc_data()[, "y"])))
  expect_true(all(is.finite(as.matrix(fit$posterior$loglik))))
  criteria <- suppressWarnings(bayests::selection_criteria(fit))
  expect_false(is.null(criteria))
})

test_that("a forecast is scored against what its horizon realised", {
  fit <- add_posterior_forecasts(add_forecast_input(uc_fit(p = 2), n_ahead = 4))
  scored <- add_predictive_loglik(fit, test_sample = at_gdp[, "y"] * 100)
  expect_identical(dim(scored$posterior$forecast$loglik), c(60L, 4L))
  expect_identical(nrow(scored$data$test$y), 4L)
  expect_error(add_predictive_loglik(fit), "test_sample")
})

test_that("a model survives a round trip through its model file", {
  fit <- add_posterior_loglik(uc_fit(p = 2, breaks = list(c(2008, 3))))
  file <- tempfile(fileext = ".h5")
  on.exit(unlink(file), add = TRUE)
  write_to_hdf5(fit, file)
  back <- bayests::read_model_from_hdf5(file)
  expect_s3_class(back, "ucmodel")
  expect_identical(back$model$algorithm, fit$model$algorithm)
  expect_identical(colnames(back$data$deterministic), colnames(fit$data$deterministic))
  expect_equal(unclass(as.matrix(back$posterior$trend$coeffs)),
               unclass(as.matrix(fit$posterior$trend$coeffs)), ignore_attr = TRUE)
  expect_equal(back$priors$tau0$mu, fit$priors$tau0$mu, ignore_attr = TRUE)
})

test_that("a failed simulation is reported on the object", {
  model <- add_initial_values(add_priors(create_ucmodel(uc_data()[, "y"], p = 2,
                                                         iterations = 10, burnin = 5)))
  model$initial$tau0 <- matrix(0, 3)
  fit <- suppressWarnings(add_posterior_coefficients(model))
  expect_true(isTRUE(fit$error))
  expect_null(fit$posterior)
})

test_that("the cycle is drawn stationary by default where the sampler can", {
  y <- uc_data()[, "y"]
  expect_true(create_ucmodel(y)$model$stationary)
  expect_true(create_ucmodel(y, error = "gamma+covar")$model$stationary)
  expect_null(create_ucmodel(y, stationary = FALSE)$model$stationary)
  expect_null(create_ucmodel(y, error = "sv")$model$stationary)
  expect_null(create_ucmodel(y, p = 0)$model$stationary)
})
