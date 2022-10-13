#####################################################################
#### Authors: Christian Lorentzen, Michael Mayer & Mario Wuthrich
#### Date 04/08/2022
#### Gini index for binary classification problem
#####################################################################

Bernoulli.deviance <- function(true, pred){-2*mean(true*log(pred)+(1-true)*log(1-pred))}

#############################################################
### binary classification: data generation
#############################################################

N0 <- 10000     # size data set

dat <- data.frame(array(NA, dim=c(N0, 3)))
set.seed(100)
# covariates (2-dimensional; X3 is used for information leakage only)
dat$X1 <- runif(n=nrow(dat))-1/2            # continuous covariate
dat$X2 <- as.factor(sample(x=c(1:3), size=nrow(dat), replace=TRUE)) # categorical covariate
dat$X3 <- 1
# learning and testing responses YL and YT
dat$YL <- 1
dat$YT <- 1
# simulate learning and test data responses
XX <- model.matrix(YL ~ X1 + X2, data=dat)
beta0 <- c(-2, 3, 1, -1)                           # true regression parameter
dat$pp <- 1/(1+exp(-as.vector( XX %*% beta0)))     # true Bernoulli probability
dat$YL <- rbinom(n=nrow(dat), size=1, prob=dat$pp) # learning data
dat$YT <- rbinom(n=nrow(dat), size=1, prob=dat$pp) # test data

# check simulated data for calibration property
round(c(mean(dat$pp), mean(dat$YL), mean(dat$YT)),4)


#############################################################
### fitting models 0 to 3
#############################################################

# null model p0
dat$p0 <- mean(dat$YL)

# underfitting model p1 (using not all covariates)
dglm <- glm(YL ~ X1, data=dat, family="binomial")
dat$p1 <- predict(dglm, newdata=dat, type="response")

# best possible GLM p2
dglm <- glm(YL ~ X1 + X2, data=dat, family="binomial")
dat$p2 <- predict(dglm, newdata=dat, type="response")

# construct overfitting model p3 by introducing a leakage of information through X3
dat[which(dat$YL==1),"X3"] <- rbinom(n=sum(dat$YL), size=1, prob=.65)
dat[which(dat$YL==0),"X3"] <- rbinom(n=nrow(dat)-sum(dat$YL), size=1, prob=.35)
dglm <- glm(YL ~ X1 + X2 + X3, data=dat, family="binomial")
dat$p3 <- predict(dglm, newdata=dat, type="response")

str(dat)


#############################################################
### numerical results of the fitted models:
### deviance losses of Section 2.1
#############################################################

## deviance losses of Table 1
# in-sample deviance loss
cbind(
round(c(Bernoulli.deviance(dat$YL, dat$pp), Bernoulli.deviance(dat$YL, dat$p0), Bernoulli.deviance(dat$YL, dat$p1), Bernoulli.deviance(dat$YL, dat$p2), Bernoulli.deviance(dat$YL, dat$p3)),4),
# out-of-sample deviance loss
round(c(Bernoulli.deviance(dat$YT, dat$pp), Bernoulli.deviance(dat$YT, dat$p0), Bernoulli.deviance(dat$YT, dat$p1), Bernoulli.deviance(dat$YT, dat$p2), Bernoulli.deviance(dat$YT, dat$p3)),4))

# Score decomposition of Bernoulli deviance
library(reliabilitydiag)
library(dplyr)
# We need the Bernoulli deviance as function with care for boundaries.
bernoulli_deviance <- function(obs, pred){
  ifelse(((obs==0 & pred==0) | (obs==1 & pred==1)),
         0,
         -2*obs * log(pred) - 2*(1 - obs) * log(1 - pred))
}

reldiag <- reliabilitydiag(
  pp = dat$pp,
  p0 = dat$p0,
  p1 = dat$p1,
  p2 = dat$p2,
  p3 = dat$p3,
  y = dat$YT,
  xtype = "continuous",
  region.level = NA #0.9,
  #region.method = "continuous_asymptotics",
  #region.position = "diagonal"
)

options(pillar.sigfig = 4)
summary(reldiag, score = "bernoulli_deviance")
# restore default
options(pillar.sigfig = 3)

## balance property of Table 2
# in-sample
cbind(
(-round(c(mean(dat$YL-dat$pp),mean(dat$YL-dat$p0),mean(dat$YL-dat$p1),mean(dat$YL-dat$p2),mean(dat$YL-dat$p3)),4)),
# out-of-sample
(-round(c(mean(dat$YT-dat$pp),mean(dat$YT-dat$p0),mean(dat$YT-dat$p1),mean(dat$YT-dat$p2),mean(dat$YT-dat$p3)),4)),
# estimated standard deviation
#simulation uncertainty
round(c(sqrt(mean(dat$pp*(1-dat$pp))/nrow(dat)), sqrt(mean(dat$p0*(1-dat$p0))/nrow(dat)), sqrt(mean(dat$p1*(1-dat$p1))/nrow(dat)), sqrt(mean(dat$p2*(1-dat$p2))/nrow(dat)), sqrt(mean(dat$p3*(1-dat$p3))/nrow(dat))),4))


#############################################################
### numerical results of the fitted models:
### Gini indices and AUC of Sections 2.3 and 2.4
#############################################################

library(pROC)

Gini <- c(2*roc(response=dat$YT, predictor=dat$pp)$auc-1,
          2*roc(response=dat$YT, predictor=dat$p0)$auc-1,
          2*roc(response=dat$YT, predictor=dat$p1)$auc-1,
          2*roc(response=dat$YT, predictor=dat$p2)$auc-1,
          2*roc(response=dat$YT, predictor=dat$p3)$auc-1)

## Gini indices and AUC of Tables 3 & 5
cbind(round(Gini,4), round((Gini + 1)/2,4))

## Backtest with Somers' D with Kendall's tau
tau1 <- c(cov(dat$YT, dat$pp, method="kendall"), cov(dat$YT, dat$p0, method="kendall"), cov(dat$YT, dat$p1, method="kendall"), cov(dat$YT, dat$p2, method="kendall"), cov(dat$YT, dat$p3, method="kendall"))
tau2 <- cov(dat$YT, dat$YT, method="kendall")
round(tau1/tau2,4)

## Confusion matrix for the ROC curve of Table 4
dat1 <- dat
tau <- 0.4
dat1$YY <- as.integer(dat1$p2 >= tau) 
(CM <- rbind(c(sum(as.integer((dat1$YY==1)&(dat1$YT==1))), sum(as.integer((dat1$YY==0)&(dat1$YT==1)))),
             c(sum(as.integer((dat1$YY==1)&(dat1$YT==0))), sum(as.integer((dat1$YY==0)&(dat1$YT==0))))))
# TPR and FPR tau=40%              
round(CM/rowSums(CM)[row(CM)],2)[,1]               


#############################################################
### auto-calibration of Sections 2.6
#############################################################

library(locfit)

## auto-calibration plot (for convenience we only consider model p3)
set.seed(100)
nn <- sample(1:nrow(dat1))  # we only plot a random sub-sample for calculation reasons
xx <- dat1[nn,"p3"] 
yy <- predict(locfit(dat1$YT ~ dat1$p3 , alpha=0.1, deg=2), newdata=xx)
# Figure 8: auto-calibration plot of model p3
# note that we use here the out-of-sample test data for this, more correctly we should have two out-of-sample data sets at this stage!
xx0 <- c(0,.8)
plot(x=xx, y=yy, xlim=xx0, ylim=xx0, main=list("auto-calibration: model p3", cex=1.5), xlab="prediction", ylab="auto-calibration", cex.lab=1.5) 
abline(a=0, b=1, col="orange", lwd=3)
abline(h=.6, col="red", lty=2)
abline(v=.6, col="red", lty=2)

## auto-calibrate model p3 using locfit (motivated by Remark 2.13)
# note that we use here the out-of-sample test data for this, more correctly we should have two out-of-sample data sets at this stage!
dat1$p3_auto <- predict(locfit(dat1$YT ~ dat1[,"p3"] , alpha=0.1, deg=2), newdata=dat1[,"p3"])

## deviance losses of Table 6
# in-sample deviance loss
cbind(
round(c(Bernoulli.deviance(dat1$YL, dat1$pp), Bernoulli.deviance(dat1$YL, dat1$p0), Bernoulli.deviance(dat1$YL, dat1$p1), Bernoulli.deviance(dat1$YL, dat1$p2), Bernoulli.deviance(dat1$YL, dat1$p3), Bernoulli.deviance(dat1$YL, dat1$p3_auto)),4),
# out-of-sample deviance loss
round(c(Bernoulli.deviance(dat1$YT, dat1$pp), Bernoulli.deviance(dat1$YT, dat1$p0), Bernoulli.deviance(dat1$YT, dat1$p1), Bernoulli.deviance(dat1$YT, dat1$p2), Bernoulli.deviance(dat1$YT, dat1$p3), Bernoulli.deviance(dat1$YT, dat1$p3_auto)),4))

## Gini indices of Table 6
round(c(2*roc(response=dat1$YT, predictor=dat1$pp)$auc-1,
        2*roc(response=dat1$YT, predictor=dat1$p0)$auc-1,
        2*roc(response=dat1$YT, predictor=dat1$p1)$auc-1,
        2*roc(response=dat1$YT, predictor=dat1$p2)$auc-1,
        2*roc(response=dat1$YT, predictor=dat1$p3)$auc-1,
        2*roc(response=dat1$YT, predictor=dat1$p3_auto)$auc-1),4)


#############################################################
#############################################################
### plots
#############################################################
#############################################################

## Lorenz curves
MirrorLorenz <- function(dat1, model){
  cc <- paste("c", model, sep="")
  pp <- paste("p", model, sep="")
  dat1[,cc] <- rev(sort(dat1[,pp]))
  for (jj in 2:nrow(dat1)){dat1[jj,cc] <- dat1[jj,cc] + dat1[jj-1,cc]}
  dat1[,cc] <- dat1[,cc] / sum(dat1[,pp])
  dat1
      }

dat1 <- dat[rev(order(dat$pp)),]
dat1 <- MirrorLorenz(dat1, "p")
for (kk in 0:3){dat1 <- MirrorLorenz(dat1, kk)}

# Figure 4
col1 <- rev(rainbow(n=4, start=0, end=1/2))
col1[3] <- "orange"
plot(x=1-c(1:nrow(dat1))/nrow(dat1), y=1-dat1$c0, type='l', lwd=2, col=col1[1], main=list("Lorenz curves of fitted regression functions", cex=1.5), ylab="Lorenz curve", xlab="alpha", cex.lab=1.5)
lines(x=1-c(1:nrow(dat1))/nrow(dat1), y=1-dat1$c1, col=col1[2], lwd=2)
lines(x=1-c(1:nrow(dat1))/nrow(dat1), y=1-dat1$c2, col=col1[3], lwd=3)
lines(x=1-c(1:nrow(dat1))/nrow(dat1), y=1-dat1$c3, col=col1[4], lwd=2)
legend(x="topleft", cex=1.5, lty=c(1,1,1,1), col=col1, lwd=c(2,2,3,2), legend=paste("model p_",c(0:3), sep=""))

# Mirrored Lorenz cuves
plot(x=c(1:nrow(dat1))/nrow(dat1), y=dat1$c0, type='l', lwd=2, col=col1[1], main=list("mirrored Lorenz curves", cex=1.5), ylab="mirrored Lorenz curve", xlab="proportion", cex.lab=1.5)
lines(x=c(1:nrow(dat1))/nrow(dat1), y=dat1$c1, col=col1[2], lwd=2)
lines(x=c(1:nrow(dat1))/nrow(dat1), y=dat1$c2, col=col1[3], lwd=3)
lines(x=c(1:nrow(dat1))/nrow(dat1), y=dat1$c3, col=col1[4], lwd=2)
legend(x="bottomright", cex=1.5, lty=c(1,1,1,1), col=col1, lwd=c(2,2,3,2), legend=paste("model p_",c(0:3), sep=""))

##############################

## cumulative accuracy profiles CAP
CAP <- function(dat1, model){
  cc <- paste("cap", model, sep="")
  pp <- paste("p", model, sep="")
  dat2 <- dat1[rev(order(dat1[,pp])),]
  dat2[,cc] <- dat2$YT
  for (jj in 2:nrow(dat2)){dat2[jj,cc] <- dat2[jj,cc] + dat2[jj-1,cc]}
  dat2[,cc] <- dat2[,cc] / sum(dat2[,"YT"])
  dat1[,cc] <- dat2[,cc]
  dat1
      }

for (kk in 1:3){dat1 <- CAP(dat1, kk)}

# Figure 6
plot(x=c(1:nrow(dat1))/nrow(dat1), y=dat1$cap1, type='l', lwd=2, col=col1[2], lty=1, main=list("model selection with Gini indices (out-of-sample)", cex=1.5), ylab="cumulative accuracy curve", xlab="alpha", cex.lab=1.5)
lines(x=c(1:nrow(dat1))/nrow(dat1), y=dat1$cap2, col=col1[3], lwd=3)
lines(x=c(1:nrow(dat1))/nrow(dat1), y=dat1$cap3, col=col1[4], lwd=2)
lines(x=c(0:1), y=c(0:1), col=col1[1], lwd=2)
lines(x=c(0,mean(dat1$YT),1), y=c(0,1,1), col="blue", lwd=2, lty=2)
legend(x="bottomright", cex=1.5, lty=c(1,1,1,1), col=col1, lwd=c(2,2,3,2), legend=paste("model p_",c(0:3), sep=""))

   
# Figure 9: ABC for model p3
xx <- c(1:nrow(dat1))/nrow(dat1)
plot(x=xx, y=dat1$c3, type='l', lwd=2, col="blue", lty=1, main=list(paste("ABC of model 3",sep=""), cex=1.5), ylab="Lorenz curve / CAP", xlab="alpha", cex.lab=1.5)
lines(x=xx, y=dat1$cap3, col="cyan", lwd=2, lty=1)
polygon(x=c(xx, rev(xx)), y=c(dat1$c3,rev(dat1$cap3)), border=FALSE, col=adjustcolor("orange",alpha.f=0.2)) 
legend(x="bottomright", cex=1.5, border=FALSE, fill=c("blue","cyan","orange"), legend=c("Lorenz curve", "CAP", "ABC"))


        