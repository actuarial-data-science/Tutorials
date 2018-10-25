###############################################
#########  Data Analysis French MTPL2
#########  Shallow Neural Network
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


#####################################################################
######## hyperbolic tangent activation shallow neural net
#####################################################################

hyperbolic.tangent.NN <- function(W1, beta1, X){
   z1 <- array(1, c(nrow(W1)+1, nrow(X)))
   z1[-1,] <- tanh(W1 %*% t(X))
   exp(t(beta1) %*% z1)
              }

#####################################################################
######## momentum-based gradient descent method for 1 hidden layer
#####################################################################

Momentum.Based.GDM.Poisson <- function(S1, q1, rho, a, beta1, W1, Xt, Yt){
   DevStat <- array(NA, c(S1+1,1))
   z1 <- array(1, c(q1+1, nrow(Xt)))
   # hyperbolic tangent activation (for sigmoid activation, see notes Listing 10)
   z1[-1,] <- tanh(W1 %*% t(Xt))
   lambda.t <- exp(t(beta1) %*% z1)
   DevStat[1,1] <- 100*Poisson.Deviance(lambda.t*Yt[,1],Yt[,2])
   # initialize the velocity vectors for the momentum-based gradient descent method
   v.beta <- array(0, dim(beta1))
   v.weights.1 <- array(0, dim(W1))
   # introduce friction parameters mu.beta, mu.weights.1 and mu.weights.2 in [0,1] (mu.beta = mu.weights.1 = mu.weights.2 = 0 leads to usual gradient descent method)
   mu.beta <- a[2]            # for a[2]=0 we do not consider any momentum
   mu.weights.1 <- a[1]       # for a[1]=0 we do not consider any momentum
   for (s0 in 1:S1){
      v1 <- 2*(lambda.t*Yt[,1] - Yt[,2])
      # calculate the gradients
      grad_beta <- z1 %*% t(v1)
      delta.1 <- (beta1[-1,] %*% v1) * (1 - (z1[-1,])^2)
      grad_W1 <- delta.1 %*% as.matrix(Xt)
      # update the parameters (momentum-based method for a[i]>0)
      v.beta <- mu.beta * v.beta - rho[2] * grad_beta/sqrt(sum(grad_beta^2))
      beta1 <- beta1 + v.beta
      v.weights.1 <- mu.weights.1 * v.weights.1 - rho[1] * grad_W1/sqrt(sum(grad_W1^2))
      W1 <- W1 + v.weights.1
      z1[-1,] <- tanh(W1 %*% t(Xt))
      lambda.t <- exp(t(beta1) %*% z1)
      DevStat[s0+1,1] <- 100*Poisson.Deviance(lambda.t*Yt[,1],Yt[,2])
       }
    # plot decrease insample loss   
    plot(DevStat[,1], type='l', col="magenta", ylim=c(range(DevStat)), ylab="deviance statistics", xlab="iteration", main=paste("shallow net: ",q1, " hidden neurons", sep=""))
    legend(x="topright", col=c("magenta"), lty=c(1), lwd=c(1), pch=c(-1), legend=c("in-sample loss"))
    # return updated parameters
    list(beta1, W1)
   }


##########################################
#########  run gradient descent algorithm
##########################################

features <- c(13:ncol(dat2))
(d1 <- length(features))                                    # dimension of feature vector (incl. intercept)
q1 <- 20                                                    # number of hidden neurons
Xlearn <- learn[, features]                                 # design matrix learning sample
Ylearn <- cbind(learn$Exposure, as.numeric(learn$ClaimNb))  # responses learning sample
Xtest <- test[, features]                                   # design matrix test sample
Ytest <- cbind(test$Exposure, as.numeric(test$ClaimNb))     # responses test sample



### load starting parameters
V0 <- 1
beta.0 <- as.matrix(read.table(file=paste("./Parameters_NN/beta_neurons_",q1,"_V",V0,".csv", sep=""), header=FALSE, sep=";"))
W.0 <- as.matrix(read.table(file=paste("./Parameters_NN/W1_neurons_",q1,"_V",V0,".csv", sep=""), header=FALSE, sep=";"))
learn$fit <- as.numeric(t(hyperbolic.tangent.NN(W.0, beta.0, Xlearn))*learn$Exposure)
test$fit <- as.numeric(t(hyperbolic.tangent.NN(W.0, beta.0, Xtest))*test$Exposure)
100*Poisson.Deviance(learn$fit, learn$ClaimNb)
100*Poisson.Deviance(test$fit, test$ClaimNb)


### running gradient descent 
rho <- c(0.003, 0.0002)     # learning rate
a <- c(.3,.2)               # momentum factor
epochs <- 10                # number of epochs of the GDM

# run gradient descent algorithm: note mini-batches may still be implemented
{t1 <- proc.time()
param <- Momentum.Based.GDM.Poisson(epochs, q1, rho, a, beta.0, W.0, Xlearn, Ylearn)
(proc.time()-t1)}

# update parameters
beta.1 <- param[[1]]
W.1 <- param[[2]]

learn$fit <- as.numeric(t(hyperbolic.tangent.NN(W.1, beta.1, Xlearn))*learn$Exposure)
test$fit <- as.numeric(t(hyperbolic.tangent.NN(W.1, beta.1, Xtest))*test$Exposure)

100*Poisson.Deviance(learn$fit, learn$ClaimNb)
100*Poisson.Deviance(test$fit, test$ClaimNb)


#####################################################################
#####################################################################
######## implementation in Keras
#####################################################################
#####################################################################

# set parameters and learning/test samples
q1 <- 20                                        # number of hidden neurons
features.keras <- c(14:ncol(dat2))              # chosen features
(q0 <- length(features.keras))                  # dimension of feature vector
Xlearn <- as.matrix(learn[, features.keras])    # design matrix learning sample
Xtest <- as.matrix(test[, features.keras])      # design matrix test sample
Wlearn <- as.matrix(learn$Exposure)             # weights (exposure) learning sample
Wtest <- as.matrix(test$Exposure)               # weights (exposure) test sample
Ylearn <- as.matrix(learn$ClaimNb)              # response learning sample
Ytest <- as.matrix(test$ClaimNb)                # response test sample


# define the neural network model in Keras
features.0 <- layer_input(shape=c(ncol(Xlearn)))         # define network for features
net <- features.0 %>%
     layer_dense(units = q1, activation = 'tanh') %>% 
     layer_dense(units = 1, activation = k_exp)
volumes.0 <- layer_input(shape=c(1))                     # define network for offset
offset <- volumes.0 %>%
     layer_dense(units = 1, activation = 'linear', use_bias=FALSE, trainable=FALSE, weights=list(array(1, dim=c(1,1))))
merged <- list(net, offset) %>%                          # combine the two networks
     layer_multiply() 
model <- keras_model(inputs=list(features.0, volumes.0), outputs=merged)    
summary(model)    
model %>% compile(loss = 'poisson', optimizer = 'rmsprop')   # define loss and optimizer

# initialize the weights as above
weights0 <- list(array(t(W.0[,-1]), dim=c(q0, q1)), array(t(W.0[,1])), array(t(beta.0[-1,1]),dim=c(q1, 1)), array(beta.0[1,1]), array(1, dim=c(1,1)) )
set_weights(model, weights0)
learn$fit.keras <- as.vector(model %>% predict(list(Xlearn, Wlearn)))
test$fit.keras <- as.vector(model %>% predict(list(Xtest, Wtest)))
100*Poisson.Deviance(learn$fit.keras, learn$ClaimNb)
100*Poisson.Deviance(test$fit.keras, test$ClaimNb)


# run gradient descent within Keras
epochs <- 10                       # number of epochs
batchsize <- 100000                # batch size

{t1 <- proc.time()
fit <- model %>% fit(list(Xlearn, Wlearn), Ylearn, epochs=epochs, batch_size=batchsize)
(proc.time()-t1)}
plot(fit)

learn$fit.keras <- as.vector(model %>% predict(list(Xlearn, Wlearn)))
test$fit.keras <- as.vector(model %>% predict(list(Xtest, Wtest)))
100*Poisson.Deviance(learn$fit.keras, learn$ClaimNb)
100*Poisson.Deviance(test$fit.keras, test$ClaimNb)

