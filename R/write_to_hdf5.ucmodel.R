#' Write an Unobserved Components Model to a BayesTS Model File
#'
#' Writes an unobserved components model to an HDF5 file in the layout the
#' BayesTS executable reads, and reads one back with
#' \code{\link[bayests]{read_model_from_hdf5}}.
#'
#' @param object an object of class \code{ucmodel}.
#' @param filename the path of the file.
#' @param group the group in the file the model is written to, the root by
#' default.
#' @param tree the tree of a model file, as \code{\link[bayests]{read_bayests_tree}}
#' returns it.
#' @param ... not used.
#'
#' @details The R object names its priors, starting values and draws as the
#' file does, so the two hold the same elements under the same names. The
#' series are \code{/data/train/y}, the drift's terms \code{/data/train/x} and
#' over the horizon \code{/data/forecast/x}, the skew index
#' \code{/data/train/skew_x} and \code{/data/forecast/skew_x}, and the realised
#' values \code{/data/test/y}.
#'
#' @return \code{write_to_hdf5} returns \code{filename} invisibly;
#' \code{from_bayests_tree} the model.
#'
#' @export
write_to_hdf5.ucmodel <- function(object, filename, group = "", ...) {
  bayests::write_bayests_tree(.uc_model_tree(object), filename = filename, group = group)
}

#' @rdname write_to_hdf5.ucmodel
#' @export
from_bayests_tree.ucmodel <- function(tree, ...) {
  .uc_model_from_tree(tree)
}

# The specification elements BayesTS reads as integers.
.uc_spec_counts <- c("k", "p", "n", "iterations", "burnin", "thin", "h", "seed")

.uc_model_tree <- function(object) {
  model <- object[["model"]]
  if (is.null(model[["algorithm"]])) {
    stop("Element 'model$algorithm' is missing. Was the object produced by ",
         "create_ucmodel()?", call. = FALSE)
  }
  attributes <- list()
  nested <- list()
  for (i in names(model)) {
    value <- model[[i]]
    if (is.null(value)) next
    if (is.list(value)) {
      nested[[i]] <- value
    } else {
      if (i %in% .uc_spec_counts && is.numeric(value) && all(value == round(value))) {
        value <- as.integer(value)
      }
      attributes[[i]] <- value
    }
  }
  det <- object[["data"]][["deterministic"]]
  attributes[["n"]] <- if (is.null(det)) 0L else ncol(det)
  attributes[["rclass"]] <- class(object)
  attributes[["rpackage"]] <- "ucmtools"

  data <- object[["data"]]
  tree_data <- list(
    train = list(y = data[["y"]], x = det, skew_x = data[["skew_x"]]),
    forecast = data[["forecast"]],
    test = data[["test"]])

  list(model = c(nested, list(.attributes = attributes)),
       data = tree_data,
       priors = object[["priors"]],
       initial = object[["initial"]],
       posterior = object[["posterior"]])
}

.uc_model_from_tree <- function(tree) {
  attributes <- tree[["model"]][[".attributes"]]
  model_class <- attributes[["rclass"]]
  if (is.null(model_class)) {
    model_class <- c("ucmodel", "list")
  }
  attributes[["rclass"]] <- NULL
  attributes[["rpackage"]] <- NULL
  # n is BayesTS's count of the drift's terms, which the terms themselves say;
  # m and s are always zero here.
  for (i in c("n", "m", "s")) attributes[[i]] <- NULL
  model <- attributes
  nested <- tree[["model"]]
  nested[[".attributes"]] <- NULL
  for (i in names(nested)) model[[i]] <- nested[[i]]
  if (!is.null(model[["deterministic"]])) {
    model[["deterministic"]] <- as.character(model[["deterministic"]])
  }

  train <- tree[["data"]][["train"]]
  data <- list(y = train[["y"]])
  if (!is.null(train[["x"]])) {
    data[["deterministic"]] <- train[["x"]]
    if (ncol(data[["deterministic"]]) == length(model[["deterministic"]])) {
      colnames(data[["deterministic"]]) <- model[["deterministic"]]
    }
  }
  data[["skew_x"]] <- train[["skew_x"]]
  data[["forecast"]] <- tree[["data"]][["forecast"]]
  data[["test"]] <- tree[["data"]][["test"]]

  result <- list(data = data, model = model)
  for (i in c("priors", "initial", "posterior")) {
    result[[i]] <- tree[[i]]
  }
  class(result) <- model_class
  result
}
