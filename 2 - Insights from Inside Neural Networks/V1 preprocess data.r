###############################################
#########  Data Analysis French MTPL2
#########  Pre process data and other tools
#########  Author: Mario Wuthrich
#########  Version October 8, 2018
###############################################

### install CASdatasets
#install.packages("CASdatasets", repos = "http://dutangc.free.fr/pub/RRepos/", type="source")

###############################################
#########  load packages and data
###############################################
 
require(MASS)
library(CASdatasets)
require(stats)
library(data.table)
library(plyr)
library(keras)

data(freMTPL2freq)
dat <- freMTPL2freq
dat$VehGas <- factor(dat$VehGas)      # consider VehGas as categorical
dat$n <- 1                            # intercept
dat$ClaimNb <- pmin(dat$ClaimNb, 4)   # correct for unreasonable observations (that might be data error)
dat$Exposure <- pmin(dat$Exposure, 1) # correct for unreasonable observations (that might be data error)
str(dat)


###############################################
#########  Poisson deviance statistics
###############################################

Poisson.Deviance <- function(pred, obs){
     2*(sum(pred)-sum(obs)+sum(log((obs/pred)^(obs))))/length(pred)
               }

##########################################
#########  feature pre-processing functions
##########################################

# MinMax scaler
PreProcess.Minimax <- function(var1, dat2){
   names(dat2)[names(dat2) == var1]  <- "V1"
   dat2$X <- as.numeric(dat2$V1)
   dat2$X <- 2*(dat2$X-min(dat2$X))/(max(dat2$X)-min(dat2$X))-1
   names(dat2)[names(dat2) == "V1"]  <- var1
   names(dat2)[names(dat2) == "X"]  <- paste(var1,"X", sep="")
   dat2
   }

# Dummy coding 
PreProcess.CatDummy <- function(var1, short, dat2){
   names(dat2)[names(dat2) == var1]  <- "V1"
   n2 <- ncol(dat2)
   dat2$X <- as.integer(dat2$V1)
   n0 <- length(unique(dat2$X))
   for (n1 in 2:n0){dat2[, paste(short, n1, sep="")] <- as.integer(dat2$X==n1)}
   names(dat2)[names(dat2) == "V1"]  <- var1
   dat2[, c(1:n2,(n2+2):ncol(dat2))]
   }

# Feature pre-processing using MinMax Scaler and Dummy Coding
Features.PreProcess <- function(dat2){
   dat2 <- PreProcess.Minimax("Area", dat2)   
   dat2 <- PreProcess.Minimax("VehPower", dat2)   
   dat2$VehAge <- pmin(dat2$VehAge,20)
   dat2 <- PreProcess.Minimax("VehAge", dat2)   
   dat2$DrivAge <- pmin(dat2$DrivAge,90)
   dat2 <- PreProcess.Minimax("DrivAge", dat2)   
   dat2$BonusMalus <- pmin(dat2$BonusMalus,150)
   dat2 <- PreProcess.Minimax("BonusMalus", dat2)   
   dat2 <- PreProcess.CatDummy("VehBrand", "Br", dat2)
   dat2$VehGasX <- as.integer(dat2$VehGas)-1.5
   dat2$Density <- round(log(dat2$Density),2)
   dat2 <- PreProcess.Minimax("Density", dat2)   
   dat2 <- PreProcess.CatDummy("Region", "R", dat2)
   dat2
    }

##########################################
#########  feature pre-processing and building learning and test samples
##########################################

dat2 <- Features.PreProcess(dat)  
str(dat2)   

set.seed(100)
ll <- sample(c(1:nrow(dat2)), round(0.9*nrow(dat2)), replace = FALSE)
learn <- dat2[ll,]
test <- dat2[-ll,]
(n_l <- nrow(learn))
(n_t <- nrow(test))


