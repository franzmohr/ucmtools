#' Austrian and Foreign Real GDP
#'
#' Quarterly log real GDP of Austria and of its trading partners, from 1979Q2
#' to 2023Q3.
#'
#' @usage data("at_gdp")
#'
#' @format A quarterly time-series object with 178 rows and two columns:
#' \describe{
#'   \item{\code{y}}{log real GDP of Austria.}
#'   \item{\code{y.s}}{log real GDP of Austria's trading partners, a trade
#'   weighted average with weights from Austria's trade over 2014 to 2016.}
#' }
#'
#' @details Taken from data set \code{at_macrodata} of package bvartools, which
#' compiles it from the GVAR database. Multiply by 100 to have the trend's
#' increments in percent.
#'
#' @references
#' Mohaddes, K., & Raissi, M. (2024). \emph{Compilation, revision and updating
#' of the global VAR (GVAR) database, 1979Q2--2023Q3} (mimeo). University of
#' Cambridge: Judge Business School.
"at_gdp"
