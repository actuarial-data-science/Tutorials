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
gender <- "Female"
ObsYear <- 1999

# training data pre-processing 
data1 <- data.preprocessing.RNNs(all_mort, gender, T0, tau0, ObsYear)

# validation data pre-processing
all_mort2 <- all_mort[which((all_mort$Year > (ObsYear-10))&(Gender==gender)),]
all_mortV <- all_mort2
vali.Y <- all_mortV[which(all_mortV$Year > ObsYear),]
 
# MinMaxScaler data pre-processing
x.min <- min(data1[[1]])
x.max <- max(data1[[1]])
x.train <- array(2*(data1[[1]]-x.min)/(x.min-x.max)-1, dim(data1[[1]]))
y.train <- - data1[[2]]
y0 <- mean(y.train)

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

{if (RNN.type=="LSTM"){model <- LSTM3(T0, tau0, tau1, tau2, tau3, y0, optimizer)}else{model <- GRU3(T0, tau0, tau1, tau2, tau3, y0, optimizer)}
 name.model <- paste(RNN.type,"3_", tau0, "_", tau1, "_", tau2, "_", tau3, sep="")
 file.name <- paste("./CallBack/best_model_", name.model,"_", gender, sep="")
 summary(model)}

# define callback
CBs <- callback_model_checkpoint(file.name, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)

# gradient descent fitting: takes roughly 200 seconds on my laptop
{t1 <- proc.time()
  fit <- model %>% fit(x=x.train, y=y.train, validation_split=0.2,
                                        batch_size=100, epochs=500, verbose=0, callbacks=CBs)                                        
proc.time()-t1}

# plot loss figures
plot.losses(name.model, gender, fit[[2]]$val_loss, fit[[2]]$loss)

# calculating in-sample loss: LC is c(Female=3.7573, Male=8.8110)
load_model_weights_hdf5(model, file.name)
round(10^4*mean((exp(-as.vector(model %>% predict(x.train)))-exp(-y.train))^2),4)

# calculating out-of-sample loss: LC is c(Female=0.6045, Male=1.8152)
pred.result <- recursive.prediction(ObsYear, all_mort2, gender, T0, tau0, x.min, x.max, model)
vali <- pred.result[[1]][which(all_mort2$Year > ObsYear),]
round(10^4*mean((vali$mx-vali.Y$mx)^2),4)

