# The claims src/core/ and inst/COPYRIGHTS make about each other.
#
# tools/update-bayests-core.R checks both when it runs, and refuses to finish
# if either is broken -- but it needs an upstream BayesTS checkout, which no
# check machine has, so between refreshes nothing verified them. Everything
# below is answerable from the vendored tree alone, which is why it can live
# here.
#
# What is *not* here is the Armadillo redirection that makes the samplers draw
# from R's generator: src/bayests_rng_guard.cpp asserts that at compile time,
# on the configuration that was actually built, which is a stronger statement
# than reading the headers. The grep below is the weaker second question --
# whether any vendored file still asks for <armadillo> by name.

# Where the sources are, from wherever the tests were started.
#
# devtools::test() runs from tests/testthat inside the checkout; R CMD check
# runs from <pkg>.Rcheck/tests/testthat, with the unpacked tarball beside it as
# <pkg>.Rcheck/ucmtools. Both have to be found, or this skips in exactly the
# run it exists for. The installed package is not a candidate: src/core is not
# installed, only compiled.
vendored_root <- function() {
  candidates <- c(".", "..", "../..", "../../..",
                  "../ucmtools", "../../ucmtools", "../../../ucmtools",
                  "../00_pkg_src/ucmtools", "../../00_pkg_src/ucmtools")
  for (root in candidates) {
    if (dir.exists(file.path(root, "src", "core")) &&
        file.exists(file.path(root, "inst", "COPYRIGHTS"))) {
      return(root)
    }
  }
  NULL
}

root <- vendored_root()

test_that("inst/COPYRIGHTS lists exactly the files that are vendored", {

  skip_if(is.null(root), "no source tree beside the tests")

  copyrights <- file.path(root, "inst", "COPYRIGHTS")

  # The same rule the refresh script applies: an indented path under inst/ or
  # src/ is a claim that the file is vendored.
  listed <- sort(trimws(grep("^[[:space:]]+(inst|src)/[^[:space:]]+$",
                             readLines(copyrights, warn = FALSE), value = TRUE)))

  present <- sort(c(
    file.path("inst/include/bayests",
              list.files(file.path(root, "inst/include/bayests"), "[.]h$",
                         recursive = TRUE)),
    file.path("src/core",
              list.files(file.path(root, "src/core"), "[.](h|cpp)$",
                         recursive = TRUE))))

  # Named rather than counted, so a failure says which file moved.
  expect_equal(setdiff(present, listed), character(0),
               label = "vendored files missing from inst/COPYRIGHTS")
  expect_equal(setdiff(listed, present), character(0),
               label = "inst/COPYRIGHTS entries with no file")
  expect_gt(length(present), 0)
})

test_that("no vendored file reaches for Armadillo directly", {

  skip_if(is.null(root), "not run from a source checkout")

  sources <- c(list.files(file.path(root, "inst/include/bayests"), "[.]h$",
                          recursive = TRUE, full.names = TRUE),
               list.files(file.path(root, "src/core"), "[.](h|cpp)$",
                          recursive = TRUE, full.names = TRUE))
  skip_if(length(sources) == 0, "not run from a source checkout")

  unpatched <- Filter(function(f) {
    any(grepl("^#include <armadillo>$", readLines(f, warn = FALSE)))
  }, sources)

  expect_equal(basename(unpatched), character(0),
               label = "files a refresh left pointing at <armadillo>")
})

test_that("the compiled samplers draw from R's generator", {

  # The question src/bayests_rng_guard.cpp answers at compile time, asked again
  # of the built object: an unseeded model draws from R's generator, so
  # set.seed() fixes its chain and a different seed moves it. If Armadillo had
  # its own generator, neither would hold.
  build <- function() {
    model <- create_ucmodel(uc_data()[, "y"], p = 1, iterations = 20, burnin = 10)
    model <- add_initial_values(add_priors(model))
    model$model$seed <- NULL
    model
  }
  draw <- function(state) {
    set.seed(state)
    as.matrix(add_posterior_coefficients(build())$posterior$trend$coeffs)
  }
  expect_identical(draw(808), draw(808))
  expect_false(identical(draw(808), draw(809)))
})
