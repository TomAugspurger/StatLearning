require(boot)

df <- read.csv('~/sandbox/StatLearning/r/xy.csv')
head(df)

mod <- lm("y ~ X1 + X2", df)
summary(mod)

se = function(y, X1, X2){
  mod <- lm(y ~ X1 + X2)
  var_e <- var(residuals(mod))
  x_bar <- mean(df$X1)
  denom <- sum((df$X1 - x_bar)**2)
  var_e / denom
}
  
se(df$y, df$X1, df$X2)

se.fn = function(data, index){
  with(data[index,], se(y, X1, X2))
}

se.fn(df, 1:1000)
set.seed(1)

boot.out=boot(df, se.fn, R=1000)
boot.out
