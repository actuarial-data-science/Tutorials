##########################################
#########  Data Analysis French MTPL
#########  Insights from Inside Neural Networks
#########  Author: Mario Wuthrich
#########  Version April 22, 2020
##########################################

# Reference
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3226852

##########################################
#########  load packages and data
##########################################
 
source(file="./Tools/V0_1 pre_process data.R")
# pro-processing uses dummy coding
source(file="./Tools/V0_2 network architectures.R")
str(learn)    

   
#####################################################################
######## data preparation and homogeneous model
#####################################################################

# feature selection
features <- c(14:ncol(dat2), 3)
# definition of design matrix (learning and test data)
learn.XX <- as.matrix(learn[,features])
test.XX <- as.matrix(test[,features])

# homogeneous model: in-sample and out-of-sample losses
(lambda.0 <- sum(learn$ClaimNb)/sum(learn$Exposure))
c(Poisson.loss(lambda.0*learn$Exposure, learn$ClaimNb),Poisson.loss(lambda.0*test$Exposure, test$ClaimNb))

#####################################################################
######## plain vanilla shallow network
#####################################################################

q1 <- 20                               # number of neurons
(qqq <- c(length(features), c(q1), 1)) # dimension of all layers including input and output
seed <- 100                            # set seed

# define plain vanilla shallow network
model <- shallow.plain.vanilla(seed, qqq, log(lambda.0))
model

# install callback
path0 <- paste("./Parameters/shallow_plain_vanilla", sep="")
CBs <- callback_model_checkpoint(path0, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)

# compile model
model %>% compile(loss = 'poisson', optimizer = 'nadam')

# fit network
epoch0 <- 100  # replication of Table 2 uses 1000 epochs which takes 440 seconds
{t1 <- proc.time()
  fit <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                       batch_size=10000, epochs=epoch0, verbose=0, callbacks=CBs)
(proc.time()-t1)[3]}

# illustrate gradient descent performance
plot.loss("topright", fit[[2]], ylim0=c(min(unlist(fit[[2]])), max(unlist(fit[[2]]))), plot.yes=0, 
           paste("./Plots/plain_vanilla_shallow.pdf", sep=""), col0=c("blue","darkgreen"))
  
# results after all epochs
learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))
# losses
c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))

# results on best validation model (callback)
load_model_weights_hdf5(model, path0)
w1 <- get_weights(model)
learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))
# losses
c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))


#####################################################################
######## plain vanilla deep network with 3 hidden layers
#####################################################################

q1 <- c(20,15,10)                      # number of neurons
(qqq <- c(length(features), c(q1), 1)) # dimension of all layers including input and output
seed <- 200                            # set seed

# define plain vanilla deep3 network (this uses tanh activation)
model <- deep3.plain.vanilla(seed, qqq, log(lambda.0))
model

# ReLU activation
#model <- deep3.ReLU(seed, qqq, log(lambda.0))
#model

# install callback
path0 <- paste("./Parameters/deep3_plain_vanilla", sep="")
CBs <- callback_model_checkpoint(path0, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)

# compile model
model %>% compile(loss = 'poisson', optimizer = 'nadam')

# fit network
epoch0 <- 300  # replication of Table 9 uses 1000 epochs which takes 600 seconds
{t1 <- proc.time()
  fit <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                       batch_size=5000, epochs=epoch0, verbose=0, callbacks=CBs)
(proc.time()-t1)[3]}

# illustrate gradient descent performance
plot.loss("topright", fit[[2]], ylim0=c(min(unlist(fit[[2]])), max(unlist(fit[[2]]))), plot.yes=0, 
           paste("./Plots/plain_vanilla_deep3.pdf", sep=""), col0=c("blue","darkgreen"))
  
# results after all epochs
learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))
# losses
c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))

# results on best validation model (callback)
load_model_weights_hdf5(model, path0)
w1 <- get_weights(model)
learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))
# losses
c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))


#####################################################################
######## deep network with 3 hidden layers and ridge regularization
#####################################################################

q1 <- c(20,15,10)                      # number of neurons
(qqq <- c(length(features), c(q1), 1)) # dimension of all layers including input and output
seed <- 200                            # set seed

# deep3 network with ridge regularizer
w0 <- rep(0.00001,3)                   #  regularization parameter
model <- deep3.ridge(seed, qqq, w0, log(lambda.0))
model

# install callback
path0 <- paste("./Parameters/deep3_ridge", sep="")
CBs <- callback_model_checkpoint(path0, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)

# compile model
model %>% compile(loss = 'poisson', optimizer = 'nadam')

# fit network
epoch0 <- 300  # replication of Table 10 uses 2000 epochs
{t1 <- proc.time()
  fit <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                       batch_size=5000, epochs=epoch0, verbose=0, callbacks=CBs)
(proc.time()-t1)[3]}

# illustrate gradient descent performance
plot.loss("topright", fit[[2]], ylim0=c(min(unlist(fit[[2]])), max(unlist(fit[[2]]))), plot.yes=0, 
           paste("./Plots/ridge_deep3.pdf", sep=""), col0=c("blue","darkgreen"))
  
# results after all epochs
learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))
# losses
c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))

# results on best validation model (callback)
load_model_weights_hdf5(model, path0)
w1 <- get_weights(model)
learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))
# losses
c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))



#####################################################################
######## deep network with 3 hidden layers with dropout and normalization
#####################################################################

q1 <- c(20,15,10)                      # number of neurons
(qqq <- c(length(features), c(q1), 1)) # dimension of all layers including input and output
seed <- 200                            # set seed

# define deep3 network with normalization layers and dropouts
w0 <- 0.05                             #  dropout rate
model <- deep3.norm.dropout(seed, qqq, w0, log(lambda.0))
model

# install callback
path0 <- paste("./Parameters/deep3_norm_dropout", sep="")
CBs <- callback_model_checkpoint(path0, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)

# compile model
model %>% compile(loss = 'poisson', optimizer = 'nadam')

# fit network
epoch0 <- 100  # replication of Table 13 uses 500 epochs and a run time of 900 seconds
{t1 <- proc.time()
  fit <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                       batch_size=5000, epochs=epoch0, verbose=0, callbacks=CBs)
(proc.time()-t1)[3]}

# illustrate gradient descent performance
plot.loss("topright", fit[[2]], ylim0=c(min(unlist(fit[[2]])), max(unlist(fit[[2]]))), plot.yes=0, 
           paste("./Plots/norm_dropout_deep3.pdf", sep=""), col0=c("blue","darkgreen"))
  
# results after all epochs
learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))
# losses
c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))

# results on best validation model (callback)
load_model_weights_hdf5(model, path0)
w1 <- get_weights(model)
learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))
# losses
c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))

#####################################################################
######## bias regularizer
#####################################################################

q1 <- c(20,15,10)                      # number of neurons
(qqq <- c(length(features), c(q1), 1)) # dimension of all layers including input and output

# define plain vanilla deep3 network (this uses tanh activation)
model <- deep3.plain.vanilla(seed=200, qqq, log(lambda.0))
model

# compile model
model %>% compile(loss = 'poisson', optimizer = 'nadam')

# restore parameters from fitting above
path0 <- paste("./Parameters/deep3_plain_vanilla", sep="")
load_model_weights_hdf5(model, path0)
w1 <- get_weights(model)
learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))
# losses
c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))

# extract neuron activations in the last hidden layer
zz <- keras_model(inputs=model$input, outputs=get_layer(model, 'layer3')$output)
zz.learn <- data.frame(zz %>% predict(list(learn.XX)))
zz.learn$yy <- learn$ClaimNb
zz.test <- data.frame(zz %>% predict(list(test.XX)))
# perform GLM step on the last hidden layer
glm1 <- glm(as.formula(glm.formula(q1[3])), data=zz.learn, family=poisson())
learn.y2 <- fitted(glm1)
test.y2 <- predict(glm1, newdata=zz.test, type="response")
# losses
c(Poisson.loss(learn.y2, learn$ClaimNb),Poisson.loss(test.y2, test$ClaimNb))
# balance property
c(mean(learn$ClaimNb), mean(learn.y), mean(learn.y2))

# update network weights
w1[[7]]  <- array(glm1$coefficients[2:(q1[3]+1)], dim=c(q1[3],1))
w1[[8]]  <- array(glm1$coefficients[1], dim=c(1))
# store updated parameters
set_weights(model, w1)
path1 <- paste("./Parameters/deep3_plain_vanilla_bias_regularized", sep="")
save_model_weights_hdf5(model, path1)

load_model_weights_hdf5(model, path1)
w1 <- get_weights(model)
learn.y3 <- as.vector(model %>% predict(list(learn.XX)))
test.y3 <- as.vector(model %>% predict(list(test.XX)))
# losses
c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))
c(Poisson.loss(learn.y3, learn$ClaimNb),Poisson.loss(test.y3, test$ClaimNb))
# balance property
c(mean(learn$ClaimNb), mean(learn.y), mean(learn.y3))
