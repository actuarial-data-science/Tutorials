#### Purpose: Prepare Data for RNns
#### Authors: Ronald Richman and Mario Wuthrich
#### Date: August 12, 2019


##################################################################
### prepare data for Recurrent Neural Networks 
##################################################################

data.preprocessing.RNNs <- function(data.raw, gender, T0, tau0, ObsYear=1999){    
    mort_rates <- data.raw[which(data.raw$Gender==gender), c("Year", "Age", "logmx")] 
    mort_rates <- dcast(mort_rates, Year ~ Age, value.var="logmx")
    # selecting data
    train.rates <- as.matrix(mort_rates[which(mort_rates$Year <= ObsYear),])
    # adding padding at the border
    (delta0 <- (tau0-1)/2)
    if (delta0>0){for (i in 1:delta0){
         train.rates <- as.matrix(cbind(train.rates[,1], train.rates[,2], train.rates[,-1], train.rates[,ncol(train.rates)]))
       }}
   train.rates <- train.rates[,-1]
   (t1 <- nrow(train.rates)-(T0-1)-1)
   (a1 <- ncol(train.rates)-(tau0-1)) 
   (n.train <- t1 * a1) # number of training samples
   xt.train <- array(NA, c(n.train, T0, tau0))
   YT.train <- array(NA, c(n.train))
   for (t0 in (1:t1)){
     for (a0 in (1:a1)){
           xt.train[(t0-1)*a1+a0,,] <- train.rates[t0:(t0+T0-1), a0:(a0+tau0-1)]
           YT.train[(t0-1)*a1+a0] <-   train.rates[t0+T0, a0+delta0]
      }}
   list(xt.train, YT.train)
      }
      
      
##################################################################
### recursive prediction
##################################################################
      
recursive.prediction <- function(ObsYear, all_mort2, gender, T0, tau0, x.min, x.max, model.p){       
   single.years <- array(NA, c(2016-ObsYear))

   for (ObsYear1 in ((ObsYear+1):2016)){
     data2 <- data.preprocessing.RNNs(all_mort2[which(all_mort2$Year >= (ObsYear1-10)),], gender, T0, tau0, ObsYear1)
     # MinMaxScaler (with minimum and maximum from above)
     x.vali <- array(2*(data2[[1]]-x.min)/(x.min-x.max)-1, dim(data2[[1]]))
     y.vali <- -data2[[2]]
     Yhat.vali2 <- exp(-as.vector(model.p %>% predict(x.vali)))
     single.years[ObsYear1-ObsYear] <- round(10^4*mean((Yhat.vali2-exp(-y.vali))^2),4)
     predicted <- all_mort2[which(all_mort2$Year==ObsYear1),]
     keep <- all_mort2[which(all_mort2$Year!=ObsYear1),]
     predicted$logmx <- -as.vector(model %>% predict(x.vali))
     predicted$mx <- exp(predicted$logmx)
     all_mort2 <- rbind(keep,predicted)
     all_mort2 <- all_mort2[order(Gender, Year, Age),]
                       }
     list(all_mort2, single.years)
     }                  


recursive.prediction.Gender <- function(ObsYear, all_mort2, gender, T0, tau0, x.min, x.max, model.p){       
   single.years <- array(NA, c(2016-ObsYear))

   for (ObsYear1 in ((ObsYear+1):2016)){
     data2 <- data.preprocessing.RNNs(all_mort2[which(all_mort2$Year >= (ObsYear1-10)),], gender, T0, tau0, ObsYear1)
     # MinMaxScaler (with minimum and maximum from above)
     x.vali <- array(2*(data2[[1]]-x.min)/(x.min-x.max)-1, dim(data2[[1]]))
     if (gender=="Female"){yy <- 0}else{yy <- 1}
     x.vali <- list(x.vali, rep(yy, dim(x.vali)[1]))
     y.vali <- -data2[[2]]
     Yhat.vali2 <- exp(-as.vector(model.p %>% predict(x.vali)))
     single.years[ObsYear1-ObsYear] <- round(10^4*mean((Yhat.vali2-exp(-y.vali))^2),4)
     predicted <- all_mort2[which(all_mort2$Year==ObsYear1),]
     keep <- all_mort2[which(all_mort2$Year!=ObsYear1),]
     predicted$logmx <- -as.vector(model %>% predict(x.vali))
     predicted$mx <- exp(predicted$logmx)
     all_mort2 <- rbind(keep,predicted)
     all_mort2 <- all_mort2[order(Gender, Year, Age),]
                       }
     list(all_mort2, single.years)
     }                  

##################################################################
### plotting functions
##################################################################

plot.losses <- function(name.model, gender, val_loss, loss){
     plot(val_loss,col="cyan3", pch=20, ylim=c(0,0.1), main=list(paste("early stopping: ",name.model,", ", gender, sep=""), cex=1.5),xlab="epochs", ylab="MSE loss", cex=1.5, cex.lab=1.5)
     lines(loss,col="blue")
     abline(h=0.05, lty=1, col="black")
     legend(x="bottomleft", col=c("blue","cyan3"), lty=c(1,-1), lwd=c(1,-1), pch=c(-1,20), legend=c("in-sample loss", "out-of-sample loss"))
   }   