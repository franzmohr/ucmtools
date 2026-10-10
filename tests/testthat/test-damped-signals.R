# The damped trend and the signal equations: the European Commission's
# trend-cycle models, nested in UcNormalGamma and UcNormalWishart.

# A signal that moves with the deviation of y from a straight line, around a
# level of 80, like a capacity utilisation indicator: plumbing, not a model.
uc_signal <- function(y = uc_data()[, "y"], n = 1) {
  set.seed(7)
  gap <- as.numeric(stats::residuals(stats::lm(y ~ seq_along(y))))
  s <- sapply(seq_len(n), function(i) 80 + 0.5 * gap + stats::rnorm(length(y), sd = 0.5))
  stats::ts(s, start = stats::start(y), frequency = stats::frequency(y),
            names = paste0("cu", seq_len(n)))
}

test_that("a damped trend draws one rho per series, inside (-1, 1)", {
  for (error in c("gamma", "gamma+covar", "wishart")) {
    fit <- uc_fit(p = 2, trend = "damped", error = error)
    expect_null(fit$error, label = error)
    rho <- as.matrix(fit$posterior$rho$coeffs)
    expect_identical(dim(rho), c(60L, 1L))
    expect_identical(colnames(rho), "y.rho")
    expect_true(all(abs(rho) < 1))
    expect_identical(ncol(fit$posterior$tau0$coeffs), 2L)
  }
  model <- add_initial_values(add_priors(create_ucmodel(uc_data(), trend = "damped"),
                                         rho = list(mu = 1.2)))
  expect_identical(dim(model$priors$rho$v_inv), c(2L, 2L))
  expect_equal(as.numeric(model$initial$rho), c(0.95, 0.95))
  expect_true(create_ucmodel(uc_data()[, "y"], trend = "damped")$model$deterministic == "const")

  fit <- add_posterior_loglik(uc_fit(p = 2, trend = "damped"))
  expect_true(all(is.finite(as.matrix(fit$posterior$loglik))))
  fit <- add_posterior_forecasts(add_forecast_input(fit, n_ahead = 4))
  expect_identical(dim(fit$posterior$forecast$forecasts), c(60L, 4L))
})

test_that("signals load on the cycle and are labelled by regressor", {
  y <- uc_data()[, "y"]
  index <- stats::ts(seq(-1, 1, length.out = length(y)), start = stats::start(y),
                     frequency = 4)
  for (error in c("gamma", "wishart")) {
    fit <- uc_fit(p = 2, error = error, signal = uc_signal(n = 2), signal_lags = 1,
                  signal_x = index)
    expect_null(fit$error, label = error)
    expect_identical(fit$model$n_signals, 2L)
    expect_identical(fit$model$n_signal_x, 2L)
    expect_identical(colnames(fit$data$signal_x), c("const", "x"))
    expect_identical(colnames(fit$posterior$signal$coeffs),
                     c("cu1.const", "cu2.const", "cu1.x", "cu2.x",
                       "cu1.y.l0", "cu2.y.l0", "cu1.y.l1", "cu2.y.l1"))
    expect_identical(colnames(fit$posterior$signal_noise_inv$coeffs), c("cu1", "cu2"))
    expect_true(all(as.matrix(fit$posterior$signal_noise_inv$coeffs) > 0))
  }
  # The constant's prior is centred on the signal's mean.
  model <- add_priors(create_ucmodel(y, signal = uc_signal()))
  expect_equal(as.numeric(model$priors$signal$mu), c(mean(uc_signal()), 0))

  fit <- uc_fit(p = 2, trend = "damped", signal = uc_signal(), signal_const = TRUE)
  expect_null(fit$error)
  fit <- add_posterior_loglik(fit)
  expect_true(all(is.finite(as.matrix(fit$posterior$loglik))))
  fit <- add_posterior_forecasts(add_forecast_input(fit, n_ahead = 4))
  expect_identical(dim(fit$posterior$forecast$forecasts), c(60L, 4L))
  scored <- add_predictive_loglik(fit, test_sample = at_gdp[, "y"] * 100)
  expect_identical(dim(scored$posterior$forecast$loglik), c(60L, 4L))
})

test_that("a model with signals survives a round trip through its model file", {
  fit <- uc_fit(p = 2, trend = "damped", signal = uc_signal(), signal_lags = 1)
  file <- tempfile(fileext = ".h5")
  on.exit(unlink(file), add = TRUE)
  write_to_hdf5(fit, file)
  back <- bayests::read_model_from_hdf5(file)
  expect_identical(back$model$trend, "damped")
  expect_identical(as.integer(back$model$n_signals), 1L)
  expect_equal(unclass(back$data$signal), unclass(fit$data$signal), ignore_attr = TRUE)
  expect_equal(unclass(back$data$signal_x), unclass(fit$data$signal_x), ignore_attr = TRUE)
  expect_equal(unclass(as.matrix(back$posterior$signal$coeffs)),
               unclass(as.matrix(fit$posterior$signal$coeffs)), ignore_attr = TRUE)
  # Drawn again from the file, the posterior is the one drawn from the object.
  again <- add_posterior_coefficients(back)
  expect_equal(unclass(as.matrix(again$posterior$rho$coeffs)),
               unclass(as.matrix(fit$posterior$rho$coeffs)), ignore_attr = TRUE)
})

test_that("the damped trend and signals are refused where no sampler reads them", {
  y <- uc_data()[, "y"]
  expect_error(create_ucmodel(y, trend = "damped", error = "sv"), "damped")
  expect_error(create_ucmodel(y, trend = "damped", error = "skew"), "damped")
  expect_error(create_ucmodel(y, signal = uc_signal(), error = "sv+covar"), "signal")
  expect_error(create_ucmodel(y, signal = uc_signal(), cycle_init = "stationary"), "signal")
  expect_error(create_ucmodel(y, signal_lags = 1), "signal")
  expect_error(create_ucmodel(y, signal = stats::window(uc_signal(), end = c(2015, 4))), "cover")
  expect_error(create_ucmodel(y, signal = as.numeric(uc_signal())), "time-series")
})

test_that("a model without the new arguments is the model it was", {
  model <- create_ucmodel(uc_data()[, "y"], p = 2)
  expect_null(model$model$n_signals)
  expect_null(model$data$signal)
  model <- add_initial_values(add_priors(model))
  expect_null(model$priors$rho)
  expect_null(model$priors$signal)
  expect_null(model$initial$rho)
})
