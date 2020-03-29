##########################################
#########  Data Analysis French MTPL
#########  Neural Network Approach
#########  Author: Mario Wuthrich
#########  Version March 23, 2018
##########################################


##########################################
#########  load packages and data
##########################################
 
require(MASS)
library(CASdatasets)
require(stats)
library(data.table)
library(plyr)

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

dat2 <- Features.PreProcess(dat)     

set.seed(100)
ll <- sample(c(1:nrow(dat2)), round(0.9*nrow(dat2)), replace = FALSE)
learn <- dat2[ll,]
test <- dat2[setdiff(c(1:nrow(dat2)),ll),]
(n_l <- nrow(learn))
(n_t <- nrow(test))

   

#####################################################################
######## gradient descent method for 1 hidden layer
#####################################################################

Shallow.Neural.Net <- function(S1, q1, rho, a, beta1, W1){
   DevStat <- array(NA, c(S1+1,1))
   z1 <- array(1, c(q1+1, nt))
   z1[-1,] <- tanh(W1 %*% t(Xt))
   lambda.t <- exp(t(beta1) %*% z1)
   DevStat[1,1] <- 200*(sum(lambda.t*Yt[,1])-sum(Yt[,2])+sum(log((Yt[,2]/(lambda.t*Yt[,1]))^(Yt[,2]))))/nt
   # Initialize the velocity vectors for the Momentum-Based Gradient Descent method
   v.beta <- array(0, dim(beta1))
   v.weights.1 <- array(0, dim(W1))
   # Introduce friction parameters mu.beta, mu.weights.1 and mu.weights.2 in [0,1] (mu.beta = mu.weights.1 = mu.weights.2 = 0 leads to usual gradient descent method)
   mu.beta <- a[2]
   mu.weights.1 <- a[1]
   for (s0 in 1:S1){
      v1 <- 2*(lambda.t*Yt[,1] - Yt[,2])
      # Calculate the gradients
      grad_beta <- z1 %*% t(v1)
      delta.1 <- (beta1[-1,] %*% v1) * (1 - (z1[-1,])^2)
      grad_W1 <- delta.1 %*% as.matrix(Xt)
      # update the parameters
      v.beta <- mu.beta * v.beta - rho[2] * grad_beta/sqrt(sum(grad_beta^2))
      beta1 <- beta1 + v.beta
      v.weights.1 <- mu.weights.1 * v.weights.1 - rho[1] * grad_W1/sqrt(sum(grad_W1^2))
      W1 <- W1 + v.weights.1
      z1 <- array(1, c(q1+1, nt))
      z1[-1,] <- tanh(W1 %*% t(Xt))
      lambda.t <- exp(t(beta1) %*% z1)
      DevStat[s0+1,1] <- 200*(sum(lambda.t*Yt[,1])-sum(Yt[,2])+sum(log((Yt[,2]/(lambda.t*Yt[,1]))^(Yt[,2]))))/nt
       }
    plot(DevStat[,1], type='l', col="magenta", ylim=c(range(DevStat)), ylab="deviance statistics", xlab="iteration", main=paste("shallow net: ",q1, " hidden neurons", sep=""))
    abline(h=c(31.47076), col="red", lty=1)
    abline(h=c(32.17123), col="blue", lty=1)
    legend(x="topright", col=c("magenta", "red", "blue"), lty=c(1,1,1), lwd=c(1,1,1), pch=c(-1,-1,-1), legend=c("(in-sample loss)","Model PBM3 (out-of-sample)", "Model GLM1  (out-of-sample)"))
    list(beta1, W1)
   }

##########################################
#########  initialize weights for start
##########################################

#starting.weights <- function(q1, MLE_hom, seed){
#   beta1 <- array(log(MLE_hom), c(q1+1,1))    
#   beta1[-1,1] <- rnorm(q1, mean=0, sd=1/sqrt(10*d1))
#   W1 <- array(0, c(q1,d1))
#   set.seed(seed)
#   for (d0 in 2:d1){W1[,d0] <- rnorm(q1, mean=1, sd=1/sqrt(10*d1)) }
#   write.table(beta1, file=paste("./Parameters_NN/beta_neurons_",q1,"_V0.csv", sep=""), sep=";", row.names=FALSE, col.names=FALSE)
#   write.table(W1, file=paste("./Parameters_NN/W1_neurons_",q1,"_V0.csv", sep=""), sep=";", row.names=FALSE, col.names=FALSE)    
#  }
#starting.weights(20, MLE_hom, 10000)


NN.lambda.regression <- function(W1, beta1, n1, X){
   z1 <- array(1, c(nrow(W1)+1, n1))
   z1[-1,] <- tanh(W1 %*% t(X))
   exp(t(beta1) %*% z1)
              }


##########################################
#########  neural network calibration
##########################################

features <- c(13:ncol(dat2))
(d1 <- length(features))
q1 <- 20
(MLE_hom <- sum(learn$ClaimNb)/sum(learn$Exposure))
Xlearn <- learn[, features]  # design matrix learning sample
Ylearn <- cbind(learn$Exposure, as.numeric(learn$ClaimNb))
Xtest <- test[, features]    # design matrix test sample
Ytest <- cbind(test$Exposure, as.numeric(test$ClaimNb))


            
batch <- learn
(nt <- nrow(batch))
Xt <- batch[, features]       
Yt <- cbind(batch$Exposure, as.numeric(batch$ClaimNb))


### run 
V0 <- 1
beta.0 <- as.matrix(read.table(file=paste("./Parameters_NN/beta_neurons_",q1,"_V",V0,".csv", sep=""), header=FALSE, sep=";"))
W1.0 <- as.matrix(read.table(file=paste("./Parameters_NN/W1_neurons_",q1,"_V",V0,".csv", sep=""), header=FALSE, sep=";"))

learn$fit <- t(NN.lambda.regression (W1.0, beta.0, n_l, Xlearn))
(Cali1.OOS <- 200*(sum(learn$fit*Ylearn[,1])-sum(Ylearn[,2])+sum(log((Ylearn[,2]/(learn$fit*Ylearn[,1]))^(Ylearn[,2]))))/n_l)       
test$fit <- t(NN.lambda.regression (W1.0, beta.0, n_t, Xtest))
(Cali1.OOS <- 200*(sum(test$fit*Ytest[,1])-sum(Ytest[,2])+sum(log((Ytest[,2]/(test$fit*Ytest[,1]))^(Ytest[,2]))))/n_t)       
 
rho <- c(0.003, 0.0002)       # step size
a <- c(.3,.2)           # momentum factor
S1 <- 10

param <- Shallow.Neural.Net(S1, q1, rho, a, beta.0, W1.0)

beta1 <- param[[1]]
W1 <- param[[2]]

V0 <- 2
#write.table(beta1, file=paste("./Parameters_NN/beta_neurons_",q1,"_V",V0,".csv", sep=""), sep=";", row.names=FALSE, col.names=FALSE)
#write.table(W1, file=paste("./Parameters_NN/W1_neurons_",q1,"_V",V0,".csv", sep=""), sep=";", row.names=FALSE, col.names=FALSE)    

learn$fit <- t(NN.lambda.regression (W1, beta1, n_l, Xlearn))
(Cali1.OOS <- 200*(sum(learn$fit*Ylearn[,1])-sum(Ylearn[,2])+sum(log((Ylearn[,2]/(learn$fit*Ylearn[,1]))^(Ylearn[,2]))))/n_l)       
test$fit <- t(NN.lambda.regression (W1, beta1, n_t, Xtest))
(Cali1.OOS <- 200*(sum(test$fit*Ytest[,1])-sum(Ytest[,2])+sum(log((Ytest[,2]/(test$fit*Ytest[,1]))^(Ytest[,2]))))/n_t)       


