##########################################
#########  Data Analysis French MTPL
#########  Neural Networks with Keras
#########  Author: Mario Wuthrich
#########  Version October 8, 2018      
##########################################

##########################################
#########  load packages and data
##########################################
 
source(file="V1 preprocess data.R")
 
##########################################
#########  designing the shallow network
##########################################

optimizers = c('sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam', 'adamax', 'nadam')

features <- c(14:ncol(dat2), 3)         # select features
(q0 <- length(features))                # dimension of features
Xlearn <- as.matrix(learn[, features])  # design matrix learning sample
Xtest <- as.matrix(test[, features])    # design matrix test sample

# define network and load pre specified weights
q1 <- 20                                # number of hidden neurons in hidden layer
V0 <- 1  
beta0 <- as.matrix(read.table(file=paste("./Parameters/beta_neurons_",q1,"_V",V0,".csv", sep=""), header=FALSE, sep=";"))
W0 <- as.matrix(read.table(file=paste("./Parameters/W1_neurons_",q1,"_V",V0,".csv", sep=""), header=FALSE, sep=";"))
W.init <- list(array(t(W0[,-1]), dim=c(q0, q1)), array(t(W0[,1])), array(t(beta0[-1,1]),dim=c(q1, 1)), array(beta0[1,1]))

model <- keras_model_sequential() 
model %>% 
     layer_dense(units = q1, activation = 'tanh', input_shape = c(q0)) %>% 
     layer_dense(units = 1, activation = k_exp)
model %>% compile(loss = 'poisson', optimizer = optimizers[4])
summary(model)

if (q1==20){set_weights(model, W.init)}


# run stochastic gradient descent in Keras
epochs <- 10
batch.size <- 10000

{t1 <- proc.time()
fit <- model %>% fit(Xlearn, learn$ClaimNb, epochs=epochs, batch_size=batch.size)
(proc.time()-t1)}

plot(fit)

learn$fit.keras <- as.vector(model %>% predict(Xlearn))
test$fit.keras <- as.vector(model %>% predict(Xtest))
100*Poisson.Deviance(learn$fit.keras, learn$ClaimNb)
100*Poisson.Deviance(test$fit.keras, test$ClaimNb)


#####################################################
#########  designing the deep network (with dropout)
#####################################################

# define network
q1 <- 20                               # number of neurons in first hidden layer
q2 <- 10                               # number of neurons in second hidden layer
p0 <- 0.1                              # drop out probability

model.deep <- keras_model_sequential() 
model.deep %>% 
     layer_dense(units = q1, activation = 'tanh', input_shape = c(q0)) %>% 
     layer_dense(units = q2, activation = 'tanh') %>% 
     layer_dropout(rate=p0) %>%
     layer_dense(units = 1, activation = k_exp)
model.deep %>% compile(loss = 'poisson', optimizer = optimizers[4])
summary(model.deep)

if ( (q1==20) & (q2==10)){load_model_weights_hdf5(model.deep, paste("./Parameters/DeepNN",q1, "_", q2, sep=""))}

# run stochastic gradient descent in Keras
epochs <- 10
batch.size <- 10000

{t1 <- proc.time()
fit <- model.deep %>% fit(Xlearn, learn$ClaimNb, epochs=epochs, batch_size=batch.size, verbose=0)
(proc.time()-t1)}

plot(fit)

learn$fit.keras <- as.vector(model.deep %>% predict(Xlearn))
test$fit.keras <- as.vector(model.deep %>% predict(Xtest))
100*Poisson.Deviance(learn$fit.keras, learn$ClaimNb)
100*Poisson.Deviance(test$fit.keras, test$ClaimNb)

#save_model_weights_hdf5(model.deep, paste("./Parameters/DeepNN",q1, "_", q2, sep=""))
              