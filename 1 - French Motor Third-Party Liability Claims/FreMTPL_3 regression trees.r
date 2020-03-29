##########################################
#########  Data Analysis French MTPL
#########  Regression Trees (CART)
#########  Author: Mario Wuthrich
#########  Version March 02, 2020
##########################################

source("./Tools/FreMTPL_1b load data.R")

str(learn.GLM)

##########################################
#########  Regression tree analysis
##########################################

### Model RT1
{t1 <- proc.time()
tree1 <- rpart(cbind(Exposure,ClaimNb) ~ Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Density + Region, 
            learn.GLM, method="poisson",
            control=rpart.control(xval=10, minbucket=10000, cp=0.0005))     # testing also 0.0001
(proc.time()-t1)[3]}

tree1
printcp(tree1)
rpart.plot(tree1)

average_loss <- cbind(tree1$cptable[,2], tree1$cptable[,3], tree1$cptable[,3]* tree1$frame$dev[1] / n_l)
plot(x=average_loss[,1], y=average_loss[,3]*100, type='l', col="blue", ylim=c(30.5,33.5), xlab="number of splits", ylab="average in-sample loss (in 10^(-2))", main="decrease of in-sample loss")
points(x=average_loss[,1], y=average_loss[,3]*100, pch=19, col="blue")
abline(h=c(31.26738), col="green", lty=2)
legend(x="topright", col=c("blue", "green"), lty=c(1,2), lwd=c(1,1), pch=c(19,-1), legend=c("Model RT1", "Model GLM1"))


learn.GLM$fit <- predict(tree1)*learn.GLM$Exposure
test.GLM$fit <- predict(tree1, newdata=test.GLM)*test.GLM$Exposure
c(Poisson.Deviance(learn.GLM$fit, learn.GLM$ClaimNb),Poisson.Deviance(test.GLM$fit, test.GLM$ClaimNb))


##########################################
#########  Cost-complexity pruning (cross-validation)
##########################################

K <- 10
{t1 <- proc.time()
tree2 <- rpart(cbind(Exposure,ClaimNb) ~ Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Density + Region, 
            learn.GLM, method="poisson",
            control=rpart.control(xval=K, minbucket=10000, cp=0.00001))
(proc.time()-t1)[3]}

printcp(tree2)

set.seed(100)
{t1 <- proc.time()
xgroup <- rep(1:K, length = nrow(learn.GLM))
xfit <- xpred.rpart(tree2, xgroup)
(n_subtrees <- dim(tree2$cptable)[1])
std3 <- numeric(n_subtrees)
err3 <- numeric(n_subtrees)
err_group <- numeric(K)
for (i in 1:n_subtrees){
 for (k in 1:K){
  ind_group <- which(xgroup ==k)  
  err_group[k] <- 2*sum(learn.GLM[ind_group,"Exposure"]*xfit[ind_group,i]-learn.GLM[ind_group,"ClaimNb"]+log((learn.GLM[ind_group,"ClaimNb"]/(learn.GLM[ind_group,"Exposure"]*xfit[ind_group,i]))^learn.GLM[ind_group,"ClaimNb"]))
               }
  err3[i] <- mean(err_group)             
  std3[i] <- sd(err_group)
   }
(proc.time()-t1)[3]}


average_loss <- cbind(tree2$cptable[,2], tree2$cptable[,3], tree2$cptable[,3]* tree2$frame$dev[1] / n_l)
y1 <- err3*K/n_l
s1 <- K*std3/n_l
x1 <- log10(tree2$cptable[,1])


xmain <- "cross-validation error plot"
xlabel <- "cost-complexity parameter (log-scale)"
ylabel <- "CV error (in 10^(-2))"


errbar(x=x1, y=y1*100, yplus=(y1+s1)*100, yminus=(y1-s1)*100, xlim=rev(range(x1)), col="blue", main=xmain, ylab=ylabel, xlab=xlabel)
lines(x=x1, y=y1*100, col="blue")
abline(h=c(min(y1+s1)*100), lty=1, col="orange")
abline(h=c(min(y1)*100), lty=1, col="magenta")
abline(h=c(31.26738), col="green", lty=2)
legend(x="topright", col=c("blue", "orange", "magenta", "green"), lty=c(1,1,1,2), lwd=c(1,1,1,1), pch=c(19,-1,-1,-1), legend=c("tree2", "1-SD rule", "min.CV rule", "Model GLM1"))

printcp(tree2)

# 1-SD rule:  cp 0.003 (4-5 splits)
# min.CV rule: minimal cp (33 splits)


learn.GLM$fit <- predict(tree2)*learn.GLM$Exposure
test.GLM$fit <- predict(tree2, newdata=test.GLM)*test.GLM$Exposure
c(Poisson.Deviance(learn.GLM$fit, learn.GLM$ClaimNb),Poisson.Deviance(test.GLM$fit, test.GLM$ClaimNb))


### Model RT3
tree3 <- prune(tree2, cp=0.003)
printcp(tree3)

rpart.plot(tree3)

learn.GLM$fit <- predict(tree3)*learn.GLM$Exposure
test.GLM$fit <- predict(tree3, newdata=test.GLM)*test.GLM$Exposure
c(Poisson.Deviance(learn.GLM$fit, learn.GLM$ClaimNb),Poisson.Deviance(test.GLM$fit, test.GLM$ClaimNb))






##########################################
#########  Smaller buckets
##########################################

K <- 10
{t1 <- proc.time()
tree2 <- rpart(cbind(Exposure,ClaimNb) ~ Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Density + Region, 
            learn.GLM, method="poisson",
            control=rpart.control(xval=K, minbucket=1000, cp=0.00001))
(proc.time()-t1)[3]}

printcp(tree2)

set.seed(100)
{t1 <- proc.time()
xgroup <- rep(1:K, length = nrow(learn.GLM))
xfit <- xpred.rpart(tree2, xgroup)
(n_subtrees <- dim(tree2$cptable)[1])
std3 <- numeric(n_subtrees)
err3 <- numeric(n_subtrees)
err_group <- numeric(K)
for (i in 1:n_subtrees){
 for (k in 1:K){
  ind_group <- which(xgroup ==k)  
  err_group[k] <- 2*sum(learn.GLM[ind_group,"Exposure"]*xfit[ind_group,i]-learn.GLM[ind_group,"ClaimNb"]+log((learn.GLM[ind_group,"ClaimNb"]/(learn.GLM[ind_group,"Exposure"]*xfit[ind_group,i]))^learn.GLM[ind_group,"ClaimNb"]))
               }
  err3[i] <- mean(err_group)             
  std3[i] <- sd(err_group)
   }
(proc.time()-t1)[3]}


average_loss <- cbind(tree2$cptable[,2], tree2$cptable[,3], tree2$cptable[,3]* tree2$frame$dev[1] / n_l)
y1 <- err3*K/n_l
s1 <- K*std3/n_l
x1 <- log10(tree2$cptable[,1])


xmain <- "cross-validation error plot"
xlabel <- "cost-complexity parameter (log-scale)"
ylabel <- "CV error (in 10^(-2))"

errbar(x=x1, y=y1*100, yplus=(y1+s1)*100, yminus=(y1-s1)*100, xlim=rev(range(x1)), col="blue", main=xmain, ylab=ylabel, xlab=xlabel)
lines(x=x1, y=y1*100, col="blue")
abline(h=c(min(y1+s1)*100), lty=1, col="orange")
abline(h=c(min(y1)*100), lty=1, col="magenta")
abline(h=c(31.26738), col="green", lty=2)
legend(x="topright", col=c("blue", "orange", "magenta", "green"), lty=c(1,1,1,2), lwd=c(1,1,1,1), pch=c(19,-1,-1,-1), legend=c("tree 1000", "1-SD rule", "min.CV rule", "Model GLM1"))

printcp(tree2)

# min.CV rule: minimal cp 0.000098707 (62 splits)


### Model RT3
tree3 <- prune(tree2, cp=0.000098707)
printcp(tree3)

rpart.plot(tree3)

learn.GLM$fit <- predict(tree3)*learn.GLM$Exposure
test.GLM$fit <- predict(tree3, newdata=test.GLM)*test.GLM$Exposure
c(Poisson.Deviance(learn.GLM$fit, learn.GLM$ClaimNb),Poisson.Deviance(test.GLM$fit, test.GLM$ClaimNb))







