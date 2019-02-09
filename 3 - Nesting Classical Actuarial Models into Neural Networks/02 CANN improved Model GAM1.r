##################################################################
#########  Nesting Classical Actuarial Models into Neural Networks
#########  CANN improved Model GAM1 corresponding to formula (3.11)
#########  Author: Mario Wuthrich          
#########  Version February 5, 2019
##################################################################

library(CASdatasets)
library(keras)
library(data.table)
library(plyr)
library(mgcv)

data(freMTPL2freq)
dat <- freMTPL2freq
dat$VehGas <- factor(dat$VehGas)      # consider VehGas as categorical
dat$ClaimNb <- pmin(dat$ClaimNb, 4)   # correct for unreasonable observations (that might be data error)
dat$Exposure <- pmin(dat$Exposure, 1) # correct for unreasonable observations (that might be data error)
dat$BonusMalus <- as.integer(pmin(dat$BonusMalus, 150))  # we truncate everywhere at 150

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
(inGLM <- Poisson.Deviance(learn$fitGLM2, learn$ClaimNb))
(outGLM <- Poisson.Deviance(test$fitGLM2, test$ClaimNb))
round(sum(test$fitGLM2)/sum(test$Exposure),4)

#############################################################
#########  GAM marginals improvement (VehAge and BonusMalus)
#############################################################
 
{t1 <- proc.time()
dat.GAM2 <- ddply(learn, .(VehAge, BonusMalus), summarize, fitGLM2=sum(fitGLM2), ClaimNb=sum(ClaimNb))
d.gam <- gam(ClaimNb ~ s(VehAge, bs="cr")+s(BonusMalus, bs="cr"), data=dat.GAM2, method="GCV.Cp", offset=log(fitGLM2), family=poisson)
(proc.time()-t1)}

summary(d.gam)

learn$fitGAM1 <- exp(predict(d.gam, newdata=learn))*learn$fitGLM2
test$fitGAM1 <- exp(predict(d.gam, newdata=test))*test$fitGLM2
dat$fitGAM1 <- exp(predict(d.gam, newdata=dat))*dat$fitGLM2

Poisson.Deviance(learn$fitGAM1, learn$ClaimNb)
Poisson.Deviance(test$fitGAM1, test$ClaimNb)
round(sum(test$fitGAM1)/sum(test$Exposure),4)

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

#######################################################
#########  neural network definitions for model (3.11)
#######################################################

learn.x <- list(as.matrix(learn[,c("VehPowerX", "VehAgeX", "VehGasX")]),
                as.matrix(learn[,"VehBrandX"]),
                as.matrix(learn[,c("DrivAgeX", "BonusMalus")]),
                as.matrix(log(learn$fitGAM1)) )

test.x <- list(as.matrix(test[,c("VehPowerX", "VehAgeX", "VehGasX")]),
                as.matrix(test[,"VehBrandX"]),
                as.matrix(test[,c("DrivAgeX", "BonusMalus")]),
                as.matrix(log(test$fitGAM1)) )

neurons <- c(20,15,10)
No.Labels <- length(unique(learn$VehBrandX))

###############################################
#########  definition of neural network (3.11)
###############################################

model.2IA <- function(No.Labels){
   Cont1        <- layer_input(shape = c(3), dtype = 'float32', name='Cont1')
   Cat1         <- layer_input(shape = c(1), dtype = 'int32',   name='Cat1')
   Cont2        <- layer_input(shape = c(2), dtype = 'float32', name='Cont2')
   LogExposure  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogExposure')     
   x.input <- c(Cont1, Cat1, Cont2, LogExposure)
   #
   Cat1_embed = Cat1 %>%  
            layer_embedding(input_dim = No.Labels, output_dim = 2, trainable=TRUE, 
                    input_length = 1, name = 'Cat1_embed') %>%
                    layer_flatten(name='Cat1_flat')
   #
   NNetwork1 = list(Cont1, Cat1_embed) %>% layer_concatenate(name='cont') %>%
            layer_dense(units=neurons[1], activation='tanh', name='hidden1') %>%
            layer_dense(units=neurons[2], activation='tanh', name='hidden2') %>%
            layer_dense(units=neurons[3], activation='tanh', name='hidden3') %>%
            layer_dense(units=1, activation='linear', name='NNetwork1', 
                    weights=list(array(0, dim=c(neurons[3],1)), array(0, dim=c(1))))
   #
   NNetwork2 = Cont2 %>%
            layer_dense(units=neurons[1], activation='tanh', name='hidden4') %>%
            layer_dense(units=neurons[2], activation='tanh', name='hidden5') %>%
            layer_dense(units=neurons[3], activation='tanh', name='hidden6') %>%
            layer_dense(units=1, activation='linear', name='NNetwork2', 
                    weights=list(array(0, dim=c(neurons[3],1)), array(0, dim=c(1))))

   #
   NNoutput = list(NNetwork1, NNetwork2, LogExposure) %>% layer_add(name='Add') %>%
                 layer_dense(units=1, activation=k_exp, name = 'NNoutput', trainable=FALSE,
                       weights=list(array(c(1), dim=c(1,1)), array(0, dim=c(1))))

   model <- keras_model(inputs = x.input, outputs = c(NNoutput))
   model %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')        
   model
   }

model <- model.2IA(No.Labels)
summary(model)

# may take a couple of minutes if epochs is more than 100
{t1 <- proc.time()
     fit <- model %>% fit(learn.x, as.matrix(learn$ClaimNb), epochs=400, batch_size=10000, verbose=0, 
                          validation_data=list(test.x, as.matrix(test$ClaimNb)))
(proc.time()-t1)}

# This plot should not be studied because in a thorough analyis one should not track 
# out-of-sample losses on the epochs, however, it is quite illustrative, here. 
oos <- 200* fit[[2]]$val_loss + 200*(-mean(test$ClaimNb)+mean(log(test$ClaimNb^test$ClaimNb)))
plot(oos, type='l', ylim=c(31.5,32.1), xlab="epochs", ylab="out-of-sample loss", cex=1.5, cex.lab=1.5, main=list(paste("Model GAM+ calibration", sep=""), cex=1.5) )
abline(h=c(32.07597, 31.50136), col="orange", lty=2)
     
learn0 <- learn     
learn0$fitGAMPlus <- as.vector(model %>% predict(learn.x))
test0 <- test
test0$fitGAMPlus <- as.vector(model %>% predict(test.x))


# in-sample and out-of-sample losses (in 10^(-2))
Poisson.Deviance(learn0$fitGAMPlus, as.vector(unlist(learn0$ClaimNb)))
Poisson.Deviance(test0$fitGAMPlus, as.vector(unlist(test0$ClaimNb)))
sum(test0$fitGAMPlus)/sum(test0$Exposure)


