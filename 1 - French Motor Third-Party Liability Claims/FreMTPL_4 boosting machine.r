##########################################
#########  Data Analysis French MTPL
#########  Poisson boosting machine
#########  Author: Mario Wuthrich
#########  Version March 02, 2020
##########################################

source("./Tools/FreMTPL_1b load data.R")

learn <- learn.GLM
test <- test.GLM
str(learn)

##########################################
#########  Poisson boosting machine
##########################################

### Model BM1/2/3

J0 <- 2      #depth of tree
M0 <- 50      #iterations

learn$fit0 <- learn$Exposure
test$fit0  <- test$Exposure

{t1 <- proc.time()
for (m in 1:M0){
    PBM.1 <- rpart(cbind(fit0,ClaimNb) ~ Area + VehPower + VehAge + DrivAge 
                   + BonusMalus + VehBrand + VehGas + Density + Region, 
             data=learn, method="poisson",
             control=rpart.control(maxdepth=J0, maxsurrogate=0, xval=1, minbucket=10000, cp=0.00001))     
             learn$fit0 <- learn$fit0 * predict(PBM.1)
             learn[,paste("PBM_",m, sep="")] <-  learn$fit0
             test$fit0 <- test$fit0 * predict(PBM.1, newdata=test)
             test[,paste("PBM_",m, sep="")] <-  test$fit0
              }
(proc.time()-t1)[3]}


losses <- array(NA, c(2,M0))

for (m in 1:M0){
     losses[1,m] <- 200*(sum(learn[,paste("PBM_",m, sep="")])-sum(learn$ClaimNb)+sum(log((learn$ClaimNb/learn[,paste("PBM_",m, sep="")])^(learn$ClaimNb))))/n_l
     losses[2,m] <- 200*(sum(test[,paste("PBM_",m, sep="")])-sum(test$ClaimNb)+sum(log((test$ClaimNb/test[,paste("PBM_",m, sep="")])^(test$ClaimNb))))/n_t
       }
       
losses[,M0]       

plot(x=c(0:M0), y=c(32.93518, losses[1,]), type='l', col="red", ylim=c(30,33.5), xlab="number of iterations", ylab="average in-sample loss (in 10^(-2))", main=paste("decrease of in-sample loss (depth=", J0,")", sep=""))
points(x=c(0:M0), y=c(32.93518, losses[1,]), pch=19, col="red")
abline(h=c(30.70841), col="blue", lty=2)
abline(h=c(31.26738), col="green", lty=2)
J1 <- J0
legend(x="topright", col=c("red", "blue", "green"), lty=c(1,2,2), lwd=c(1,1,1), pch=c(19,-1,-1), legend=c(paste("Model PBM", J1, sep=""), "Model RT2", "Model GLM1"))


plot(x=c(1:M0), y=losses[2,], type='l', lwd=2, col="red", ylim=c(30.5,33.5), xlab="number of iterations", ylab="average out-of-sample loss (in 10^(-2))", main="decrease of out-of-sample loss")
abline(h=c(32.17123), col="green", lty=2)
legend(x="topright", col=c("red", "green"), lty=c(1,2), lwd=c(1,1), pch=c(19,-1), legend=c(paste("Model PBM", J0, sep=""), "Model GLM1"))



##########################################
#########  GLM Boost
##########################################

### Model GLM1
{t1 <- proc.time()
  d.glm1 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM
                       + VehBrand + VehGas + DensityGLM + Region + AreaGLM, 
                       data=learn.GLM, offset=log(Exposure), family=poisson())
(proc.time()-t1)[3]}
                   
learn.GLM$fit <- fitted(d.glm1)
test.GLM$fit <- predict(d.glm1, newdata=test.GLM, type="response")
c(Poisson.Deviance(learn.GLM$fit, learn.GLM$ClaimNb),Poisson.Deviance(test.GLM$fit, test.GLM$ClaimNb))




### Model GLMBoost

J0 <- 3       #depth of tree
M0 <- 50      #iterations

learn.GLM$fit0 <- learn.GLM$fit
test.GLM$fit0  <- test.GLM$fit

{t1 <- proc.time()
for (m in 1:M0){
    PBM.1 <- rpart(cbind(fit0,ClaimNb) ~ Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Density + Region, 
             data=learn.GLM, method="poisson",
             control=rpart.control(maxdepth=J0, maxsurrogate=0, xval=1, minbucket=10000, cp=0.00001))     
             learn.GLM$fit0 <- learn.GLM$fit0 * predict(PBM.1)
             learn.GLM[,paste("PBM_",m, sep="")] <-  learn.GLM$fit0
             test.GLM$fit0 <- test.GLM$fit0 * predict(PBM.1, newdata=test.GLM)
             test.GLM[,paste("PBM_",m, sep="")] <-  test.GLM$fit0
              }
(proc.time()-t1)[3]}

losses <- array(NA, c(2,M0))

for (m in 1:M0){
     losses[1,m] <- 200*(sum(learn.GLM[,paste("PBM_",m, sep="")])-sum(learn.GLM$ClaimNb)+sum(log((learn.GLM$ClaimNb/learn.GLM[,paste("PBM_",m, sep="")])^(learn.GLM$ClaimNb))))/n_l
     losses[2,m] <- 200*(sum(test.GLM[,paste("PBM_",m, sep="")])-sum(test.GLM$ClaimNb)+sum(log((test.GLM$ClaimNb/test.GLM[,paste("PBM_",m, sep="")])^(test.GLM$ClaimNb))))/n_t
       }
       
losses[,M0]       

plot(x=c(0:M0), y=c(31.26738, losses[1,]), type='l', col="magenta", ylim=c(30,32), xlab="number of iterations", ylab="average in-sample loss (in 10^(-2))", main=paste("GLM Boost: decrease of in-sample loss (depth=", J0,")", sep=""))
points(x=c(0:M0), y=c(31.26738, losses[1,]), pch=19, col="magenta")
abline(h=c(30.13151), col="red", lty=2)
abline(h=c(30.70841), col="blue", lty=2)
abline(h=c(31.26738), col="green", lty=2)
J1 <- J0
legend(x="topright", col=c("magenta", "red", "blue", "green"), lty=c(1,2,2,2), lwd=c(1,1,1,1), pch=c(19,-1,-1,-1), legend=c(paste("Model GLMBoost", sep=""), "Model PBM3","Model RT2", "Model GLM1"))


plot(x=c(1:M0), y=losses[2,], type='l', lwd=2, col="magenta", ylim=c(30.5,33.5), xlab="number of iterations", ylab="average out-of-sample loss (in 10^(-2))", main="decrease of out-of-sample loss")
abline(h=c(31.46842), col="red", lty=2)
abline(h=c(32.17123), col="green", lty=2)
legend(x="topright", col=c("magenta", "red", "green"), lty=c(1,2,2), lwd=c(1,1,1), pch=c(-1,-1,-1), legend=c(paste("Model GLMBoost", sep=""), "Model PBM3", "Model GLM1"))





##########################################
#########  Shrinkage
##########################################

### Model BM3

J0 <- 3       #depth of tree
M0 <- 50      #iterations
nu <- .75
minbucket0 <- 10000

learn$fit0 <- learn$Exposure
test$fit0  <- test$Exposure

{ m <- 1            
             PBM.1 <- rpart(cbind(fit0,ClaimNb) ~ Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Density + Region, 
             data=learn, method="poisson",
             control=rpart.control(maxdepth=J0, maxsurrogate=0, xval=1, minbucket=minbucket0, cp=0.00001))     
             learn$fit0 <- learn$fit0 * predict(PBM.1)
             learn[,paste("PBM_",m, sep="")] <-  learn$fit0
             test$fit0 <- test$fit0 * predict(PBM.1, newdata=test)
             test[,paste("PBM_",m, sep="")] <-  test$fit0
 for (m in 2:M0){
    PBM.1 <- rpart(cbind(fit0,ClaimNb) ~ Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Density + Region, 
             data=learn, method="poisson",
             control=rpart.control(maxdepth=J0, maxsurrogate=0, xval=1, minbucket=minbucket0, cp=0.00001))     
             learn$fit0 <- learn$fit0 * (predict(PBM.1)^nu)
             learn[,paste("PBM_",m, sep="")] <-  learn$fit0
             test$fit0 <- test$fit0 * (predict(PBM.1, newdata=test)^nu)
             test[,paste("PBM_",m, sep="")] <-  test$fit0
              }
             } 

losses <- array(NA, c(2,M0))

for (m in 1:M0){
     losses[1,m] <- 200*(sum(learn[,paste("PBM_",m, sep="")])-sum(learn$ClaimNb)+sum(log((learn$ClaimNb/learn[,paste("PBM_",m, sep="")])^(learn$ClaimNb))))/n_l
     losses[2,m] <- 200*(sum(test[,paste("PBM_",m, sep="")])-sum(test$ClaimNb)+sum(log((test$ClaimNb/test[,paste("PBM_",m, sep="")])^(test$ClaimNb))))/n_t
       }
       
losses[,M0]       
