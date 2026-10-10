#' Add Initial Values to an Unobserved Components Model
#'
#' Sets where the Gibbs sampler of an unobserved components model starts, and
#' gives the model a seed for its posterior simulation unless it has one.
#'
#' @param object an object of class \code{ucmodel} with priors, see
#' \code{\link{add_priors.ucmodel}}.
#' @param ... not used.
#'
#' @details The chain starts from cycle coefficients of zero, the drift at the
#' mean growth of each series and its breaks at zero, the trend before the
#' sample at the prior mean, no correlation between the shocks, and shock
#' precisions of \eqn{1 / s_i^2}, the variance of the first differences of
#' series \eqn{i}. The trend path is not among them: every draw starts from it.
#'
#' The seed is drawn from R's random number generator, so \code{set.seed()}
#' before this call makes it reproducible; \code{\link[bayests]{add_seed}}
#' replaces it.
#'
#' @return \code{object} with element \code{initial}, and \code{model$seed}.
#'
#' @export
add_initial_values.ucmodel <- function(object, ...) {

  if (is.null(object[["priors"]])) {
    stop("Argument 'object' has no priors. Use add_priors() first.", call. = FALSE)
  }
  model <- object[["model"]]
  y <- object[["data"]][["y"]]
  det <- object[["data"]][["deterministic"]]
  n_det <- if (is.null(det)) 0 else ncol(det)
  n_skew <- if (is.null(object[["data"]][["skew_x"]])) 0 else ncol(object[["data"]][["skew_x"]])
  d <- .uc_dims(model, n_det, n_skew)
  s2 <- .uc_scale(y)
  priors <- object[["priors"]]
  initial <- list()

  if (d$n_a > 0) {
    initial[["a"]] <- matrix(0, d$n_a)
  }
  if (n_det > 0) {
    # vec(C), k x n, column by column: the constant first, at each series'
    # mean growth, the breaks at zero.
    c0 <- matrix(0, d$k, n_det)
    growth <- colMeans(diff(as.matrix(y)))
    if (d$order == 2) growth <- 0 * growth
    c0[, colnames(det) == "const"] <- growth
    initial[["c"]] <- matrix(as.vector(c0))
  }
  initial[["tau0"]] <- priors[["tau0"]][["mu"]]

  shock_scale <- rep(s2, 2)
  switch(model[["algorithm"]],
         UcNormalGamma = {
           initial[["u_omega_inv"]] <- matrix(1 / shock_scale)
         },
         UcNormalWishart = {
           initial[["u_sigma_inv"]] <- diag(1 / shock_scale, d$shocks)
         },
         UcNormalStochvol = {
           tt <- nrow(y)
           initial[["h"]] <- matrix(log(shock_scale), tt, d$shocks, byrow = TRUE)
           initial[["h_init"]] <- matrix(log(shock_scale))
         },
         UcNormalSkew = {
           initial[["u_omega_inv"]] <- matrix(1 / s2)
           initial[["skew"]] <- priors[["skew"]][["mu"]]
         })

  if (.uc_covar(model)) {
    initial[["psi"]] <- matrix(0, d$n_psi)
  }
  if (isTRUE(model[["irregular"]])) {
    initial[["irregular_inv"]] <- matrix(1 / s2)
  }

  object[["initial"]] <- initial
  if (is.null(object[["model"]][["seed"]])) {
    object[["model"]][["seed"]] <- bayests::.draw_model_seed()
  }
  object
}
