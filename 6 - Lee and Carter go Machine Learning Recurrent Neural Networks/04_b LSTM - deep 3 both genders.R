#### Purpose: Fit recurrent neural network to HMD data
#### Authors: Ronald Richman and Mario Wuthrich
#### Date: August 12, 2019


# select parameters
path.data <- "CHE_mort.csv"           # path and name of data file
region <- "CHE"                    # country to be loaded (code is for one selected country)

# load corresponding data
source(file="00_a package - load data.R")
str(all_mort)

# load pre-defined networks
source(file="00_b package - network definitions.R")

##################################################################
### Setting the parameters and load data
##################################################################

# package containing relevant functions
source(file="00_c package - data preparation RNNs.R")

# choice of parameters
T0 <- 10
tau0 <- 5
ObsYear <- 1999

# training data pre-processing 
data1 <- data.preprocessing.RNNs(all_mort, "Female", T0, tau0, ObsYear)
data2 <- data.preprocessing.RNNs(all_mort, "Male", T0, tau0, ObsYear)

xx <- dim(data1[[1]])[1]
x.train <- array(NA, dim=c(2*xx, dim(data1[[1]])[c(2,3)]))
y.train <- array(NA, dim=c(2*xx))
gender.indicator <- rep(c(0,1), xx)
for (l in 1:xx){
   x.train[(l-1)*2+1,,] <- data1[[1]][l,,]
   x.train[(l-1)*2+2,,] <- data2[[1]][l,,]
   y.train[(l-1)*2+1] <- -data1[[2]][l]
   y.train[(l-1)*2+2] <- -data2[[2]][l]
          }
# MinMaxScaler data pre-processing
x.min <- min(x.train)
x.max <- max(x.train)
x.train <- list(array(2*(x.train-x.min)/(x.min-x.max)-1, dim(x.train)), gender.indicator)
y0 <- mean(y.train)



# validation data pre-processing
all_mort2.Female <- all_mort[which((all_mort$Year > (ObsYear-10))&(Gender=="Female")),]
all_mortV.Female <- all_mort2.Female
vali.Y.Female <- all_mortV.Female[which(all_mortV.Female$Year > ObsYear),]
all_mort2.Male <- all_mort[which((all_mort$Year > (ObsYear-10))&(Gender=="Male")),]
all_mortV.Male <- all_mort2.Male
vali.Y.Male <- all_mortV.Male[which(all_mortV.Male$Year > ObsYear),]

 
   

##################################################################
### LSTM architectures
##################################################################

# network architecture deep 3 network
tau1 <- 20
tau2 <- 15
tau3 <- 10
optimizer <- 'adam'

# choose either LSTM or GRU network
RNN.type <- "LSTM"
#RNN.type <- "GRU"

{if (RNN.type=="LSTM"){model <- LSTM3.Gender(T0, tau0, tau1, tau2, tau3, y0, optimizer)}else{model <- GRU3.Gender(T0, tau0, tau1, tau2, tau3, y0, optimizer)}
 name.model <- paste(RNN.type,"3_", tau0, "_", tau1, "_", tau2, "_", tau3, sep="")
 #file.name <- paste("./Model_Full_Param/best_model_", name.model, sep="")
 file.name <- paste("./CallBack/best_model_", name.model, sep="")
 summary(model)}

# define callback
CBs <- callback_model_checkpoint(file.name, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)

# gradient descent fitting: takes roughly 400 seconds on my laptop
{t1 <- proc.time()
  fit <- model %>% fit(x=x.train, y=y.train, validation_split=0.2,
                                        batch_size=100, epochs=500, verbose=0, callbacks=CBs)                                        
proc.time()-t1}

# plot loss figures
plot.losses(name.model, "Both", fit[[2]]$val_loss, fit[[2]]$loss)

# calculating in-sample loss: LC is c(Female=3.7573, Male=8.8110)
load_model_weights_hdf5(model, file.name)
round(10^4*mean((exp(-as.vector(model %>% predict(x.train)))-exp(-y.train))^2),4)

# calculating out-of-sample loss: LC is c(Female=0.6045, Male=1.8152)
# Female
pred.result <- recursive.prediction.Gender(ObsYear, all_mort2.Female, "Female", T0, tau0, x.min, x.max, model)
vali <- pred.result[[1]][which(all_mort2.Female$Year > ObsYear),]
round(10^4*mean((vali$mx-vali.Y.Female$mx)^2),4)
# Male
pred.result <- recursive.prediction.Gender(ObsYear, all_mort2.Male, "Male", T0, tau0, x.min, x.max, model)
vali <- pred.result[[1]][which(all_mort2.Male$Year > ObsYear),]
round(10^4*mean((vali$mx-vali.Y.Male$mx)^2),4)

