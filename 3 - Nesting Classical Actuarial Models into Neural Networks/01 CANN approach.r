##################################################################
#########  Nesting Classical Actuarial Models into Neural Networks
#########  Models GLM2, neural net with embeddings, and CANN
#########  Author: Mario Wuthrich
#########  Version February 5, 2019
##################################################################

###############################################
#########  load packages and clean data
###############################################
 
library(CASdatasets)
library(keras)
library(data.table)


data(freMTPL2freq)
dat <- freMTPL2freq
dat$VehGas <- factor(dat$VehGas)      # consider VehGas as categorical
dat$ClaimNb <- pmin(dat$ClaimNb, 4)   # correct for unreasonable observations (that might be data error)
dat$Exposure <- pmin(dat$Exposure, 1) # correct for unreasonable observations (that might be data error)
str(dat)

###############################################
#########  Poisson deviance statistics
###############################################

Poisson.Deviance <- function(pred, obs){200*(sum(pred)-sum(obs)+sum(log((obs/pred)^(obs))))/length(pred)}

###############################################
#########  feature pre-processing for GLM
###############################################

dat2 <- dat
dat2$AreaGLM <- as.integer(dat2$Area)
dat2$VehPowerGLM <- as.factor(pmin(dat2$VehPower,9))
VehAgeGLM <- cbind(c(0:110), c(1, rep(2,10), rep(3,100)))
dat2$VehAgeGLM <- as.factor(VehAgeGLM[dat2$VehAge+1,2])
dat2[,"VehAgeGLM"] <-relevel(dat2[,"VehAgeGLM"], ref="2")
DrivAgeGLM <- cbind(c(18:100), c(rep(1,21-18), rep(2,26-21), rep(3,31-26), rep(4,41-31), rep(5,51-41), rep(6,71-51), rep(7,101-71)))
dat2$DrivAgeGLM <- as.factor(DrivAgeGLM[dat2$DrivAge-17,2])
dat2[,"DrivAgeGLM"] <-relevel(dat2[,"DrivAgeGLM"], ref="5")
dat2$BonusMalusGLM <- as.integer(pmin(dat2$BonusMalus, 150))
dat2$DensityGLM <- as.numeric(log(dat2$Density))
dat2[,"Region"] <-relevel(dat2[,"Region"], ref="R24")
str(dat2)

###############################################
#########  choosing learning and test sample
###############################################

set.seed(100)
ll <- sample(c(1:nrow(dat2)), round(0.9*nrow(dat2)), replace = FALSE)
learn <- dat2[ll,]
test <- dat2[-ll,]
(n_l <- nrow(learn))
(n_t <- nrow(test))
sum(learn$ClaimNb)/sum(learn$Exposure)

###############################################
#########  GLM2 analysis
###############################################

{t1 <- proc.time()
d.glm2 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + BonusMalusGLM
                        + VehBrand + VehGas + DensityGLM + Region + AreaGLM +
                        DrivAge + log(DrivAge) +  I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4), 
                        data=learn, offset=log(Exposure), family=poisson())
(proc.time()-t1)}

summary(d.glm2)
                  
learn$fitGLM2 <- fitted(d.glm2)
test$fitGLM2 <- predict(d.glm2, newdata=test, type="response")
dat$fitGLM2 <- predict(d.glm2, newdata=dat2, type="response")

# in-sample and out-of-sample losses (in 10^(-2))
Poisson.Deviance(learn$fitGLM2, learn$ClaimNb)
Poisson.Deviance(test$fitGLM2, test$ClaimNb)
round(sum(test$fitGLM2)/sum(test$Exposure),4)

######################################################
#########  feature pre-processing for (CA)NN Embedding
######################################################

PreProcess.Continuous <- function(var1, dat2){
   names(dat2)[names(dat2) == var1]  <- "V1"
   dat2$X <- as.numeric(dat2$V1)
   dat2$X <- 2*(dat2$X-min(dat2$X))/(max(dat2$X)-min(dat2$X))-1
   names(dat2)[names(dat2) == "V1"]  <- var1
   names(dat2)[names(dat2) == "X"]  <- paste(var1,"X", sep="")
   dat2
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
   dat2$VehBrandX <- as.integer(dat2$VehBrand)-1
   dat2$VehGasX <- as.integer(dat2$VehGas)-1.5
   dat2$Density <- round(log(dat2$Density),2)
   dat2 <- PreProcess.Continuous("Density", dat2)   
   dat2$RegionX <- as.integer(dat2$Region)-1
   dat2
    }

dat2 <- Features.PreProcess(dat)     

###############################################
#########  choosing learning and test sample
###############################################

set.seed(100)
ll <- sample(c(1:nrow(dat2)), round(0.9*nrow(dat2)), replace = FALSE)
learn <- dat2[ll,]
test <- dat2[-ll,]
(n_l <- nrow(learn))
(n_t <- nrow(test))
learn0 <- learn
test0 <- test

###############################################
#########  neural network (with embeddings)
###############################################

# definition of feature variables (non-categorical)
features <- c(14:18, 20:21)
(q0 <- length(features))
# learning data
Xlearn <- as.matrix(learn[, features])  # design matrix learning sample
Brlearn <- as.matrix(learn$VehBrandX)
Relearn <- as.matrix(learn$RegionX)
Ylearn <- as.matrix(learn$ClaimNb)
# testing data
Xtest <- as.matrix(test[, features])    # design matrix test sample
Brtest <- as.matrix(test$VehBrandX)
Retest <- as.matrix(test$RegionX)
Ytest <- as.matrix(test$ClaimNb)
# choosing the right volumes for EmbNN and CANN
Vlearn <- as.matrix(log(learn$Exposure))
Vtest <- as.matrix(log(test$Exposure))
lambda.hom <- sum(learn$ClaimNb)/sum(learn$Exposure)

CANN <- 1  # 0=Embedding NN, 1=CANN

if (CANN==1){
     Vlearn <- as.matrix(log(learn$fitGLM2))
     Vtest <- as.matrix(log(test$fitGLM2))
     lambda.hom <- sum(learn$ClaimNb)/sum(learn$fitGLM2)
            }
lambda.hom

# hyperparameters of the neural network architecture
(BrLabel <- length(unique(learn$VehBrandX)))
(ReLabel <- length(unique(learn$RegionX)))
q1 <- 20   
q2 <- 15
q3 <- 10
d <- 2         # dimensions embedding layers for categorical features
# define the network architecture
Design   <- layer_input(shape = c(q0),  dtype = 'float32', name = 'Design')
VehBrand <- layer_input(shape = c(1),   dtype = 'int32', name = 'VehBrand')
Region   <- layer_input(shape = c(1),   dtype = 'int32', name = 'Region')
LogVol   <- layer_input(shape = c(1),   dtype = 'float32', name = 'LogVol')
    #
BrandEmb = VehBrand %>% 
      layer_embedding(input_dim = BrLabel, output_dim = d, input_length = 1, name = 'BrandEmb') %>%
      layer_flatten(name='Brand_flat')
       
RegionEmb = Region %>% 
      layer_embedding(input_dim = ReLabel, output_dim = d, input_length = 1, name = 'RegionEmb') %>%
      layer_flatten(name='Region_flat')

Network = list(Design, BrandEmb, RegionEmb) %>% layer_concatenate(name='concate') %>% 
          layer_dense(units=q1, activation='tanh', name='hidden1') %>%
          layer_dense(units=q2, activation='tanh', name='hidden2') %>%
          layer_dense(units=q3, activation='tanh', name='hidden3') %>%
          layer_dense(units=1, activation='linear', name='Network', 
                    weights=list(array(0, dim=c(q3,1)), array(log(lambda.hom), dim=c(1))))

Response = list(Network, LogVol) %>% layer_add(name='Add') %>% 
           layer_dense(units=1, activation=k_exp, name = 'Response', trainable=FALSE,
                        weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

model <- keras_model(inputs = c(Design, VehBrand, Region, LogVol), outputs = c(Response))
model %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')
summary(model)
# fitting the neural network
{t1 <- proc.time()
fit <- model %>% fit(list(Xlearn, Brlearn, Relearn, Vlearn), Ylearn, epochs=100, 
                                  batch_size=10000, verbose=0, validation_split=0)
(proc.time()-t1)}

plot(fit)

# calculating the predictions
learn0$fitNN <- as.vector(model %>% predict(list(Xlearn, Brlearn, Relearn, Vlearn)))
test0$fitNN <- as.vector(model %>% predict(list(Xtest, Brtest, Retest, Vtest)))

# in-sample and out-of-sample losses (in 10^(-2))
Poisson.Deviance(learn0$fitNN, as.vector(unlist(learn0$ClaimNb)))
Poisson.Deviance(test0$fitNN, as.vector(unlist(test0$ClaimNb)))
sum(test0$fitNN)/sum(test0$Exposure)
