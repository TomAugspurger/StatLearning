setwd('~/sandbox/StatLearning/r')
load('../data//7.R.RData')
plot(x, y)

mod <- lm(y ~ x)
summary(mod)
coefficients(mod)

x2 <- x^2
mod2 <- lm(y ~ x + x2)
coefficients(mod2)
