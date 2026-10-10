# data/at_gdp.rda: Austrian and foreign log real GDP, from data set at_macrodata
# of bvartools, which compiles the GVAR database of Mohaddes and Raissi (2024).
# Run from the package root with bvartools' source tree at the path below.

load("D:/workspace-r/bvartools/data/at_macrodata.rda")
at_gdp <- cbind(y = at_macrodata[["domestic"]][, "y"],
                y.s = at_macrodata[["foreign"]][, "y.s"])
save(at_gdp, file = "data/at_gdp.rda", compress = "xz")
