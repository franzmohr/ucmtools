#!/usr/bin/env Rscript

## Refresh the vendored BayesTS core.
##
##   Rscript tools/update-bayests-core.R [path/to/BayesTS]
##
## Copies the part of the core this package needs into the two places it lives
## here and reapplies the one local patch: <armadillo> becomes "bayests/arma.h",
## so that RcppArmadillo is seen first and the sampler draws from R's RNG.
## Everything this script knows is written up in src/core/VENDORED.md; if you
## change one, change the other.
##
## Unlike bvartools, which mirrors the whole core, this package takes only what
## the four unobserved components models reach -- they are the samplers here,
## and compiling the other twenty-two would be minutes of build time and a
## larger shared object for code that is never called. The script is dfmtools'
## with its entry points changed. The set is *computed* rather than listed: the script
## starts from the entry points below and follows every #include of a
## "bayests/..." or "core/..." header until nothing new turns up, taking each
## header's .cpp alongside it. A list would go stale the first time upstream
## added an include; this cannot.

args <- commandArgs(trailingOnly = TRUE)
upstream <- if (length(args) > 0) {
  args[1]
} else {
  Sys.getenv("BAYESTS_SOURCE", "D:/projects/BayesTS")
}

if (!dir.exists(file.path(upstream, "src", "core"))) {
  stop("no BayesTS source tree at '", upstream, "'.\n",
       "  Pass the path as an argument or set BAYESTS_SOURCE.", call. = FALSE)
}

## Where a path of each kind lives upstream and here. A "bayests/x.h" include
## resolves against include/ upstream and inst/include/ here; a "core/x.cpp"
## include resolves against src/ upstream and src/ here.
from_of <- function(rel) {
  if (startsWith(rel, "bayests/")) file.path(upstream, "include", rel)
  else file.path(upstream, "src", rel)
}
to_of <- function(rel) {
  if (startsWith(rel, "bayests/")) file.path("inst/include", rel)
  else file.path("src", rel)
}

## Ours, not upstream's; a refresh must leave them alone. `bayests/arma.h` is
## *not* among them -- it is upstream's own file, and the redirection to
## RcppArmadillo it provides for is applied from src/Makevars rather than by
## editing it.
keep <- c("core/VENDORED.md")

## What the package actually calls. Everything else is discovered from here.
##
##   - each sampler's declaration and its numerics;
##   - spec.cpp and inputs.cpp, which hold VarSpec::validate() and every
##     Input::validate(). They are reached by no #include -- the samplers call
##     them across translation units -- so they have to be named.
##
## All four unobserved components models are named. They share uc_support.h and
## chan_jeliazkov_2009, so each one after the first costs little more than its
## own two files; an unlisted sampler is not an error here, it is a link error
## later.
entry <- c("bayests/uc_normal_gamma.h",
           "core/models/uc_normal_gamma.cpp",
           "bayests/uc_normal_wishart.h",
           "core/models/uc_normal_wishart.cpp",
           "bayests/uc_normal_stochvol.h",
           "core/models/uc_normal_stochvol.cpp",
           "bayests/uc_normal_skew.h",
           "core/models/uc_normal_skew.cpp",
           "core/spec.cpp",
           "core/inputs.cpp")

patch <- function(lines) {
  lines <- lines[lines != "#define ARMA_DONT_PRINT_FAST_MATH_WARNING"]
  sub("^#include <armadillo>$", "#include \"bayests/arma.h\"", lines)
}

## The "bayests/..." and "core/..." headers one file includes.
includes_of <- function(path) {
  lines <- readLines(path, warn = FALSE)
  hits <- regmatches(lines, regexpr('"(bayests|core)/[^"]+"', lines))
  gsub('"', "", hits)
}

## Transitive closure of `entry` under #include, plus the .cpp beside every .h
## that has one -- a header whose definitions live in a source file is no use
## without it.
closure <- function(seed) {
  seen <- character()
  todo <- seed
  while (length(todo) > 0) {
    rel <- todo[[1]]
    todo <- todo[-1]
    if (rel %in% seen) next
    from <- from_of(rel)
    if (!file.exists(from)) {
      stop("upstream has no '", rel, "', which something here includes.", call. = FALSE)
    }
    seen <- c(seen, rel)

    todo <- c(todo, includes_of(from))
    if (grepl("\\.h$", rel)) {
      companion <- sub("\\.h$", ".cpp", rel)
      ## Only under src/: an include/ header's implementation is in src/core/.
      for (cand in c(companion, sub("^bayests/", "core/", companion))) {
        if (!startsWith(cand, "bayests/") && file.exists(from_of(cand))) {
          todo <- c(todo, cand)
        }
      }
    }
  }
  sort(unique(seen))
}

copy_one <- function(from, to) {
  lines <- patch(readLines(from, warn = FALSE))
  old <- if (file.exists(to)) readLines(to, warn = FALSE) else character()
  if (identical(lines, old)) return("unchanged")
  dir.create(dirname(to), recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, to)
  if (length(old) == 0L) "new" else "updated"
}

vendored <- closure(entry)

cat("upstream <- ", upstream, "\n", sep = "")
status <- vapply(vendored, function(rel) copy_one(from_of(rel), to_of(rel)), character(1))
for (rel in vendored[status != "unchanged"]) {
  cat(sprintf("  %-9s %s\n", status[[rel]], to_of(rel)))
}
cat(sprintf("  %d file(s) reached from the entry points, %d changed\n",
            length(vendored), sum(status != "unchanged")))

## Anything here that the closure no longer reaches is either ours or left over
## from a refresh that shrank. Reported, not deleted: this script does not
## remove files it did not put there.
here <- c(file.path("bayests", list.files("inst/include/bayests", "\\.h$", recursive = TRUE)),
          file.path("core", list.files("src/core", "\\.(h|cpp|md)$", recursive = TRUE)))
extra <- setdiff(here, c(vendored, keep))
if (length(extra) > 0) {
  cat("  no longer reached -- delete by hand if it is not yours:\n")
  cat(paste0("    ", vapply(extra, to_of, character(1)), collapse = "\n"), "\n")
}

## The patch is the whole point, so check it took. A file that still reaches
## for <armadillo> would compile and run and quietly draw from Armadillo's RNG
## instead of R's; src/bayests_rng_guard.cpp is the second line of defence.
left <- unlist(lapply(
  c(list.files("inst/include/bayests", "\\.h$", recursive = TRUE, full.names = TRUE),
    list.files("src/core", "\\.(h|cpp)$", recursive = TRUE, full.names = TRUE)),
  function(f) if (any(grepl("^#include <armadillo>$", readLines(f, warn = FALSE)))) f))
if (length(left) > 0) {
  stop("the Armadillo patch did not take in:\n  ",
       paste(left, collapse = "\n  "),
       "\nFix the rule in this script before building.", call. = FALSE)
}

## inst/COPYRIGHTS names the vendored files one by one and claims to name all of
## them, which is what a CRAN reviewer checks it against. Nothing about copying
## a file adds it to that list, so the list drifts silently. Compare the two
## here, where the set that just changed is still at hand.
listed <- sort(trimws(grep("^\\s+(inst|src)/\\S+$", readLines("inst/COPYRIGHTS", warn = FALSE),
                           value = TRUE)))
expected <- sort(vapply(vendored, to_of, character(1)))

missing <- setdiff(expected, listed)
stale <- setdiff(listed, expected)
if (length(missing) > 0 || length(stale) > 0) {
  stop("inst/COPYRIGHTS does not match what is vendored.\n",
       if (length(missing)) paste0("  add:    ", paste(missing, collapse = "\n  add:    "), "\n"),
       if (length(stale)) paste0("  remove: ", paste(stale, collapse = "\n  remove: "), "\n"),
       "The file claims to list every vendored source, so it has to.",
       call. = FALSE)
}
cat("\ninst/COPYRIGHTS lists all ", length(vendored), " vendored files.\n", sep = "")
cat("\nNow rebuild. src/Makevars globs core/*.cpp and core/*/*.cpp, so a file the\n",
    "closure newly reached is picked up without an edit, and Rcpp::compileAttributes()\n",
    "has nothing to do here -- the core exports nothing to R.\n", sep = "")
