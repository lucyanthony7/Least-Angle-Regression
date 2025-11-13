library(lars)
library(ggplot2)
data(diabetes)

lars_fit <- lars(diabetes$x, diabetes$y, type = "lasso")

plot(lars_fit, xvar = "step", main = "LARS Coefficient Paths")