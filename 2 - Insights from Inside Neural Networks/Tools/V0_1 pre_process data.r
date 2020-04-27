##########################################
#########  Data Analysis French MTPL
#########  Insights from Inside Neural Networks
#########  Author: Mario Wuthrich
#########  Version April 22, 2020
##########################################

# Reference
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3226852

##########################################
#########  load packages and data
##########################################
 
require(MASS)
library(CASdatasets)
require(stats)
library(data.table)
library(plyr)
library(keras)


data(freMTPL2freq)
dat <- freMTPL2freq
dat$VehGas <- factor(dat$VehGas)
dat$n <- 1
dat$ClaimNb <- pmin(dat$ClaimNb, 4) # correct for unreasonable observations (that might be data error)
dat$Exposure <- pmin(dat$Exposure, 1) # correct for unreasonable observations (that might be data error)
str(dat)

##########################################
#########  feature pre-processing functions
##########################################

PreProcess.Continuous <- function(var1, dat2){
   names(dat2)[names(dat2) == var1]  <- "V1"
   dat2$X <- as.numeric(dat2$V1)
   dat2$X <- 2*(dat2$X-min(dat2$X))/(max(dat2$X)-min(dat2$X))-1
   names(dat2)[names(dat2) == "V1"]  <- var1
   names(dat2)[names(dat2) == "X"]  <- paste(var1,"X", sep="")
   dat2
   }

PreProcess.CatDummy <- function(var1, short, dat2){
   names(dat2)[names(dat2) == var1]  <- "V1"
   n2 <- ncol(dat2)
   dat2$X <- as.integer(dat2$V1)
   n0 <- length(unique(dat2$X))
   for (n1 in 2:n0){dat2[, paste(short, n1, sep="")] <- as.integer(dat2$X==n1)}
   names(dat2)[names(dat2) == "V1"]  <- var1
   dat2[, c(1:n2,(n2+2):ncol(dat2))]
   }

Features.PreProcess <- function(dat2){
   dat2 <- PreProcess.Continuous("Area", dat2)   
   dat2 <- PreProcess.Continuous("VehPower", dat2)   
   dat2$VehAge <- pmin(dat2$VehAge,20)
   dat2 <- PreProcess.Continuous("VehAge", dat2)   
   dat2$DrivAge <- pmin(dat2$DrivAge,90)
   dat2 <- PreProcess.Continuous("DrivAge", dat2)   
   dat2$BonusMalus <- pmin(dat2$BonusMalus,150)
   dat2 <- PreProcess.Continuous("BonusMalus", dat2)   
   dat2 <- PreProcess.CatDummy("VehBrand", "Br", dat2)
   dat2$VehGasX <- as.integer(dat2$VehGas)-1.5
   dat2$Density <- round(log(dat2$Density),2)
   dat2 <- PreProcess.Continuous("Density", dat2)   
   dat2 <- PreProcess.CatDummy("Region", "R", dat2)
   dat2
    }

##########################################
#########  feature pre-processing and building learning and test samples
##########################################

# note that we apply the MinMaxScaler: we apply it on learning and test data simultaneously
# because the learning data contains all labels.
dat2 <- Features.PreProcess(dat)     

set.seed(100)
ll <- sample(c(1:nrow(dat2)), round(0.9*nrow(dat2)), replace = FALSE)
learn <- dat2[ll,]
test <- dat2[-ll,]
(n_l <- nrow(learn))
(n_t <- nrow(test))

   
