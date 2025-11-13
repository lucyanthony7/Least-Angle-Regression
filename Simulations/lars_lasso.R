library(lars)
library(ggplot2)

data(diabetes)
x <- scale(diabetes$x)
y <- scale(diabetes$y)

#LARS algorithm
lars_fit <- lars(x, y, type="lar")

#LARS with Lasso modification algorithm
lars_lasso_fit <- lars(x, y, type="lasso")

plot(lars_fit, xvar="step", main="LARS algorithm")
plot(lars_lasso_fit, xvar="step", main ="Lasso-modified LARS algorithm")