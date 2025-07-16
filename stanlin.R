# This is an alternative, Bayesian, approach to smoothing and growth rates.
# It avoids the problems of the approximation in GPC but getting good mixing can be hard
# Here we consider just alpha

library(rstan)
library(tidyverse)

df <- read.csv('./output/Counts.csv')

days <- df$collection_date
positives <- df$All
alpha <- df$Alpha...B.1.1.7

jj <- which((!is.na(alpha) & !is.na(positives)))
ii <- (jj[1]-14):(jj[length(jj)]+14) # This is the 14 observations before and after

myN <- length(ii)
tt <- days[ii]
mm <- positives[ii]
mm[is.na(mm)] <- 0
yy <- alpha[ii]
yy[is.na(yy)] <- 0

# Quick check of part extracted
plot(tt,yy)
plot(tt,yy/mm)

data <- list(
  N = myN,
  t = tt,
  m = mm,
  y = yy
)

fit <- stan(
  file = "logisticGP.stan",
  data = data,
  chains = 10,
  warmup = 1000,
  iter = 3000,
  cores = 10,
  refresh = 100
)

# See how mixing went
traceplot(fit)

# Make dataframes for spaghetti plots with a little thinning
f_out <- rstan::extract(fit)$f
ff <- t(f_out[seq(1,2000,10),])
pp <- plogis(ff)
dr <- diff(ff)
df_pp <- add_column(as.data.frame(pp), time=tt, .before=TRUE)
df_dr <- add_column(as.data.frame(dr), time=tt[1:(length(tt)-1)], .before=TRUE)

ggplot(
  data=df_pp %>% pivot_longer(!time),
  aes(x=time, y=value, group=name)
  ) +
  geom_line(alpha=0.008, linewidth=3) + ylab('Proportion') + xlab('Day')

ggplot(
  data=df_dr %>% pivot_longer(!time),
  aes(x=time, y=value, group=name)
  ) +
  geom_line(alpha=0.01, linewidth=1) + ylab('Relative Growth Rate') + xlab('Day')

