# Vendored BayesTS core

`src/core/` and `inst/include/bayests/` are a copy of part of the core layer of
[BayesTS](https://github.com/franzmohr/BayesTS): the sampler declarations in
`include/bayests/` and the numerics in `src/core/`. Nothing else of that
project is here -- the core links neither HDF5 nor HighFive, prints nothing and
reads no files, which is what makes it embeddable in an R package.

The copy is **BayesTS `4808dff`**, the tip of upstream's `uc-cycle-init`
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
