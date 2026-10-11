# Vendored BayesTS core

`src/core/` and `inst/include/bayests/` are a copy of part of the core layer of
[BayesTS](https://github.com/franzmohr/BayesTS): the sampler declarations in
`include/bayests/` and the numerics in `src/core/`. Nothing else of that
project is here -- the core links neither HDF5 nor HighFive, prints nothing and
reads no files, which is what makes it embeddable in an R package.

The copy is **BayesTS `deeebd0`**, BayesTS `main` after merging `c7d9f38`,
which adds the volatility break to VecNormalWishart: the rows of the
differenced response, of the short-run regressors and of the levels the
cointegration term is built from are divided by the scale of their period, and
the forecast and the score are the level VAR's, break included. Neither model
is copied here; of the files here it changes `inputs.cpp`,
`error_scale_checks.h` and the three shared headers. Fixed-seed coefficients,
log likelihoods and forecasts of every error specification are identical to a
clean build of the previous copy.

Before that the copy was **BayesTS `9f470f1`**, BayesTS `main` after merging `ae77301` and
`9f470f1`. `ae77301` lets the volatility break and the Student t errors of
VarNormalWishart read a panel not observed whole -- `/data/train/constraints`,
mixed frequencies and missing values -- through `completion_support.h`;
`9f470f1` moves the four refusals of those two error scales out of the
samplers' headers into the new `core/models/error_scale_checks.h`, so that
`inputs.cpp` no longer includes `student_t_support.h` and
`volatility_break_support.h`. Of the files here it changes `inputs.cpp` and
`completion_support.h` and adds `error_scale_checks.h`; `student_t_support.h`
and `volatility_break_support.h` are no longer copied, while `adaptive_walk.h`
stays, being UcNormalSkew's proposal. No draw moves: fixed-seed coefficients,
log likelihoods and forecasts of every error specification are identical to a
build of the previous copy, and upstream compared the fingerprints of all 215
fixtures, none moved.

Before that the copy was **BayesTS `8c9865f`**, BayesTS `main` after merging `b1e8849`,
`004b000` and `efd18b8`. `b1e8849` adds a volatility break to VarNormalWishart
(Lenza and Primiceri 2022) and `004b000` Student t errors to it, with the
adapted random walk proposal `UcNormalSkew` used moved out into the new
`core/models/adaptive_walk.h` so the two share it; `efd18b8` has `bayests
check` refuse a structural model it would be asked to score, and makes
VecNormalGamma, VecNormalWishart and VecKlgs2010 raise that refusal as "this
VEC" rather than under the name of the level VAR they score through. Of the
files here it changes `spec.h`, `inputs.h`, `priors.h`, `results.h`,
`inputs.cpp`, `completion_support.h` and `uc_normal_skew.cpp`, and adds
`adaptive_walk.h` -- `UcNormalSkew`'s proposal, moved unchanged -- along with
`student_t_support.h` and `volatility_break_support.h`, which the closure
reaches through `inputs.cpp` and no model here uses. No draw moves: fixed-seed
coefficients, log likelihoods and forecasts under gamma, Wishart, stochastic
volatility and skewed errors are identical to a build of the previous copy.

Before that the copy was **BayesTS `02a41de`**, BayesTS `main` after merging `bc27dfe` and `02a41de`. `bc27dfe` adds the damped
trend and the signal equations to UcNormalGamma and UcNormalWishart, the
European Commission's output gap models nested in them; `02a41de` moves
`require_uc_signal_data()` from `uc_support.h` to `model_support.h`, so that
`inputs.cpp` no longer includes the unobserved components support header. It changes `spec.h`,
`inputs.h`, `results.h`, `spec.cpp`, `inputs.cpp`, `model_support.h`,
`uc_support.h`, `uc_constant_support.h` and the two models' sources. Draws of a
file without the new attributes are unchanged; upstream compared the
fingerprints of all 202 fixtures, none moved. Neither addition is reachable from
R yet.

Before that the copy was **BayesTS `4808dff`**, the tip of upstream's `uc-cycle-init`
branch off `main` at `22b3488`: `/model/cycle_init = "stationary"`, the cycle
before the sample drawn from its stationary distribution in UcNormalGamma and
UcNormalWishart (`spec.h`, `results.h`, `spec.cpp`, `inputs.cpp`,
`uc_support.h`, `uc_constant_support.h` and the two models' sources). Draws of
a file without the attribute are unchanged; upstream compared the fingerprints
of all 194 fixtures, none moved.

Before that the copy was **BayesTS `75d35c6`**, upstream `main` after the merge
of the unobserved components models (`7cdd0ac`, UcNormalGamma; `14130e1`,
UcNormalWishart, UcNormalStochvol, UcNormalSkew and the irregular component).

## What is copied

Not the whole core: `tools/update-bayests-core.R` starts from the four
samplers' headers and sources, `spec.cpp` and `inputs.cpp`, and follows every
`#include` of a `"bayests/..."` or `"core/..."` header until nothing new turns
up, taking each header's `.cpp` beside it. That is 44 files, every one of them
listed in `inst/COPYRIGHTS`, which the script checks.

## The one local change

`#include <armadillo>` becomes `#include "bayests/arma.h"`, and
`src/Makevars` defines `BAYESTS_ARMA_HEADER` as `<RcppArmadillo.h>`. That is
what makes the samplers draw from R's random number generator, so that
`set.seed()` and the model's seed reach them. `src/bayests_rng_guard.cpp` turns
a refresh that loses the patch into a compile error.

## Refreshing

    Rscript tools/update-bayests-core.R D:/projects/BayesTS

then compare the trees with `diff --strip-trailing-cr` (the script's count of
changed files is not to be trusted on its own), rebuild with
`R CMD INSTALL --preclean .`, and update the commit named above.
