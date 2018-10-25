###############################################
#########  Data Analysis French MTPL2
#########  GLM, Regression Trees and Boosting
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
library(rpart)
library(rpart.plot)
library(Hmisc)

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
dat2$AreaRT <- as.integer(dat2$Area)
dat2$VehGasRT <- as.integer(dat2$VehGas)
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

###############################################
#########  GLM analysis
###############################################

### Model GLM1
d.glm1 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM
                        + VehBrand + VehGas + DensityGLM + Region + AreaGLM, 
                        data=learn, offset=log(Exposure), family=poisson())
                   
summary(d.glm1)  
#anova(d.glm1)   # this calculation takes a bit of time                                          

learn$fitGLM <- fitted(d.glm1)
test$fitGLM <- predict(d.glm1, newdata=test, type="response")

# in-sample and out-of-sample losses (in 10^(-2))
(insampleGLM <- 100*Poisson.Deviance(learn$fitGLM, learn$ClaimNb))
100*Poisson.Deviance(test$fitGLM, test$ClaimNb)


###############################################
#########  Regressionn tree analysis
###############################################

### Model RT2
tree1 <- rpart(cbind(Exposure,ClaimNb) ~ AreaRT + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGasRT + Density + Region, 
            learn, method="poisson",
            control=rpart.control(xval=1, minbucket=10000, cp=0.00001))     

rpart.plot(tree1)        # plot tree
tree1                    # show tree with all binary splits
printcp(tree1)           # cost-complexit statistics

learn$fitRT <- predict(tree1)*learn$Exposure
test$fitRT <- predict(tree1, newdata=test)*test$Exposure
100*Poisson.Deviance(learn$fitRT, learn$ClaimNb)
100*Poisson.Deviance(test$fitRT, test$ClaimNb)


average_loss <- cbind(tree1$cptable[,2], tree1$cptable[,3], tree1$cptable[,3]* tree1$frame$dev[1] / n_l)
plot(x=average_loss[,1], y=average_loss[,3]*100, type='l', col="blue", ylim=c(30.5,33.5), xlab="number of splits", ylab="average in-sample loss (in 10^(-2))", main="decrease of in-sample loss")
points(x=average_loss[,1], y=average_loss[,3]*100, pch=19, col="blue")
abline(h=c(insampleGLM), col="green", lty=2)
legend(x="topright", col=c("blue", "green"), lty=c(1,2), lwd=c(1,1), pch=c(19,-1), legend=c("Model RT2", "Model GLM1"))


# cross-validation and cost-complexity pruning
K <- 10                  # K-fold cross-validation value
set.seed(100)
xgroup <- rep(1:K, length = nrow(learn))
xfit <- xpred.rpart(tree1, xgroup)
(n_subtrees <- dim(tree1$cptable)[1])
std1 <- numeric(n_subtrees)
err1 <- numeric(n_subtrees)
err_group <- numeric(K)
for (i in 1:n_subtrees){
 for (k in 1:K){
  ind_group <- which(xgroup ==k)  
  err_group[k] <- Poisson.Deviance(learn[ind_group,"Exposure"]*xfit[ind_group,i],learn[ind_group,"ClaimNb"])
               }
  err1[i] <- mean(err_group)             
  std1[i] <- sd(err_group)
   }

x1 <- log10(tree1$cptable[,1])
xmain <- "cross-validation error plot"
xlabel <- "cost-complexity parameter (log-scale)"
ylabel <- "CV error (in 10^(-2))"
errbar(x=x1, y=err1*100, yplus=(err1+std1)*100, yminus=(err1-std1)*100, xlim=rev(range(x1)), col="blue", main=xmain, ylab=ylabel, xlab=xlabel)
lines(x=x1, y=err1*100, col="blue")
abline(h=c(min(err1+std1)*100), lty=1, col="orange")
abline(h=c(min(err1)*100), lty=1, col="magenta")
abline(h=c(insampleGLM), col="green", lty=2)
legend(x="topright", col=c("blue", "orange", "magenta", "green"), lty=c(1,1,1,2), lwd=c(1,1,1,1), pch=c(19,-1,-1,-1), legend=c("tree1", "1-SD rule", "min.CV rule", "Model GLM1"))

# prune to appropriate cp constant
printcp(tree1)
tree2 <- prune(tree1, cp=0.00003)
printcp(tree2)

learn$fitRT2 <- predict(tree2)*learn$Exposure
test$fitRT2 <- predict(tree2, newdata=test)*test$Exposure
100*Poisson.Deviance(learn$fitRT2, learn$ClaimNb)
100*Poisson.Deviance(test$fitRT2, test$ClaimNb)



###############################################
#########  Poisson regression tree boosting
###############################################

### Model PBM3
J0 <- 3       # depth of tree
M0 <- 50      # iterations
nu <- 1       # shrinkage constant 

learn$fitPBM <- learn$Exposure
test$fitPBM  <- test$Exposure

for (m in 1:M0){
    PBM.1 <- rpart(cbind(fitPBM,ClaimNb) ~ AreaRT + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGasRT + Density + Region, 
             data=learn, method="poisson",
             control=rpart.control(maxdepth=J0, maxsurrogate=0, xval=1, minbucket=10000, cp=0.00001))     
             if(m>1){
                   learn$fitPBM <- learn$fitPBM * predict(PBM.1)^nu
                   test$fitPBM <- test$fitPBM * predict(PBM.1, newdata=test)^nu
                   } else {
                   learn$fitPBM <- learn$fitPBM * predict(PBM.1)
                   test$fitPBM <- test$fitPBM * predict(PBM.1, newdata=test)
                   }
              }

100*Poisson.Deviance(learn$fitPBM, learn$ClaimNb)
100*Poisson.Deviance(test$fitPBM, test$ClaimNb)
