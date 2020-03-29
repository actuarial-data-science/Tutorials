##########################################
#########  Data Analysis French MTPL
#########  Load and Pre-Process Data
#########  Author: Mario Wuthrich
#########  Version March 02, 2020
##########################################


##########################################
#########  load packages and data
##########################################
 
require(MASS)
library(CASdatasets)
require(stats)
library(data.table)
library(plyr)
library(plyr)
library(rpart)
library(xtable)
library(Hmisc)
library(rpart.plot)


data(freMTPL2freq)
dat <- freMTPL2freq
dat$VehGas <- factor(dat$VehGas)
dat$n <- 1

dat$ClaimNb <- pmin(dat$ClaimNb, 4) # correct for unreasonable observations (that might be data error)
dat$Exposure <- pmin(dat$Exposure, 1) # correct for unreasonable observations (that might be data error)

#save(dat, file="./Data/dat.rda")

Poisson.Deviance <- function(pred, obs){200*(sum(pred)-sum(obs)+sum(log((obs/pred)^(obs))))/length(pred)}

##########################################
#########  build learning and test samples
##########################################

dat2 <- dat
dat2$AreaGLM <- as.integer(dat2$Area)
dat2$VehPowerGLM <- as.factor(pmin(dat2$VehPower,9))
VehAgeGLM <- cbind(c(0:110), c(1, rep(2,10), rep(3,100)))
dat2$VehAgeGLM <- as.factor(VehAgeGLM[dat2$VehAge+1,2])
dat2[,"VehAgeGLM"] <-relevel(dat2[,"VehAgeGLM"], ref="2")
DrivAgeGLM <- cbind(c(18:100), c(rep(1,3), rep(2,5), rep(3,5), rep(4,10), rep(5,10), rep(6,20), rep(7,30)))
dat2$DrivAgeGLM <- as.factor(DrivAgeGLM[dat2$DrivAge-17,2])
dat2[,"DrivAgeGLM"] <-relevel(dat2[,"DrivAgeGLM"], ref="5")
dat2$BonusMalusGLM <- as.integer(pmin(dat2$BonusMalus, 150))
dat2$DensityGLM <- as.numeric(log(dat2$Density))
dat2[,"Region"] <-relevel(dat2[,"Region"], ref="R24")

set.seed(100)
ll <- sample(c(1:nrow(dat2)), round(0.9*nrow(dat2)), replace = FALSE)
learn.GLM <- dat2[ll,]
test.GLM <- dat2[-ll,]
n_l <- nrow(learn.GLM)
n_t <- nrow(test.GLM)


