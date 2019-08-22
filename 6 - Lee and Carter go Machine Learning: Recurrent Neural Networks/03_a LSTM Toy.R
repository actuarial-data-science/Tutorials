#### Purpose: Toy Examples Recurrent Neural Nets
#### Authors: Ronald Richman and Mario Wuthrich
#### Date: August 12, 2019


# select parameters
path.data <- "CHE_mort.csv"           # path and name of data file
region <- "CHE"                    # country to be loaded (code is for one selected country)

# load corresponding data
source(file="00_a package - load data.R")
str(all_mort)


##################################################################
### LSTMs and GRUs
##################################################################

source(file="00_b package - network definitions.R")

T0 <- 10
tau0 <- 3
tau1 <- 5
tau2 <- 4

summary(LSTM1(T0, tau0, tau1, 0, "nadam"))
summary(LSTM2(T0, tau0, tau1, tau2, 0, "nadam"))
summary(LSTM_TD(T0, tau0, tau1, 0, "nadam"))
summary(GRU1(T0, tau0, tau1, 0, "nadam"))
summary(GRU2(T0, tau0, tau1, tau2, 0, "nadam"))
summary(FNN(T0, tau0, tau1, tau2, 0, "nadam"))


##################################################################
### Bringing the data in the right structure for a toy example
##################################################################

gender <- "Female"
ObsYear <- 2000

mort_rates <- all_mort[which(all_mort$Gender==gender), c("Year", "Age", "logmx")] 
mort_rates <- dcast(mort_rates, Year ~ Age, value.var="logmx")


T0 <- 10     # lookback period
tau0 <- 3    # dimension of x_t (should be odd for our application)
delta0 <- (tau0-1)/2

toy_rates <- as.matrix(mort_rates[which(mort_rates$Year %in% c((ObsYear-T0):(ObsYear+1))),])

xt <- array(NA, c(2,ncol(toy_rates)-tau0, T0, tau0))
YT <- array(NA, c(2,ncol(toy_rates)-tau0))

for (i in 1:2){for (a0 in 1:(ncol(toy_rates)-tau0)){ 
    xt[i,a0,,] <- toy_rates[c(i:(T0+i-1)),c((a0+1):(a0+tau0))]
    YT[i,a0] <- toy_rates[T0+i,a0+1+delta0]
      }}

plot(x=toy_rates[1:T0,1], y=toy_rates[1:T0,2], col="white", xlab="calendar years", ylab="raw log-mortality rates", cex.lab=1.5, cex=1.5, main=list("data toy example", cex=1.5), xlim=range(toy_rates[,1]), ylim=range(toy_rates[,-1]), type='l')
for (a0 in 2:ncol(toy_rates)){
  if (a0 %in% (c(1:100)*3)){
    lines(x=toy_rates[1:T0,1], y=toy_rates[1:T0,a0])    
    points(x=toy_rates[(T0+1):(T0+2),1], y=toy_rates[(T0+1):(T0+2),a0], col=c("blue", "red"), pch=20)
    lines(x=toy_rates[(T0):(T0+1),1], y=toy_rates[(T0):(T0+1),a0], col="blue", lty=2)
    lines(x=toy_rates[(T0+1):(T0+2),1], y=toy_rates[(T0+1):(T0+2),a0], col="red", lty=2)
    }}
    
    
##################################################################
### LSTMs and GRUs
##################################################################

x.train <- array(2*(xt[1,,,]-min(xt))/(max(xt)-min(xt))-1, c(ncol(toy_rates)-tau0, T0, tau0))
x.vali  <- array(2*(xt[2,,,]-min(xt))/(max(xt)-min(xt))-1, c(ncol(toy_rates)-tau0, T0, tau0))
y.train <- - YT[1,]
(y0 <- mean(y.train))
y.vali  <- - YT[2,]

### examples

tau1 <- 5    # dimension of the outputs z_t^(1) first RNN layer
tau2 <- 4    # dimension of the outputs z_t^(2) second RNN layer

CBs <- callback_model_checkpoint("./CallBack/best_model", monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)
model <- LSTM2(T0, tau0, tau1, tau2, y0, "nadam")     
summary(model)

# takes 40 seconds on my laptop
{t1 <- proc.time()
  fit <- model %>% fit(x=x.train, y=y.train, validation_data=list(x.vali, y.vali),
                                  batch_size=10, epochs=500, verbose=0, callbacks=CBs)
 proc.time()-t1}

plot(fit[[2]]$val_loss,col="red", ylim=c(0,0.2), main=list("early stopping rule", cex=1.5),xlab="epochs", ylab="MSE loss", cex=1.5, cex.lab=1.5)
lines(fit[[2]]$loss,col="blue")
abline(h=0.1, lty=1, col="black")
legend(x="bottomleft", col=c("blue","red"), lty=c(1,-1), lwd=c(1,-1), pch=c(-1,1), legend=c("in-sample loss", "out-of-sample loss"))



load_model_weights_hdf5(model, "./CallBack/best_model")
Yhat.train1 <- as.vector(model %>% predict(x.train))
Yhat.vali1 <- as.vector(model %>% predict(x.vali))
c(round(mean((Yhat.train1-y.train)^2),4), round(mean((Yhat.vali1-y.vali)^2),4))




