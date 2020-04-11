##########################################################
### Authors: Simon Rentzmann and Mario Wüthrich
### Date: August 9, 2019
### Tutorial: Unsupervised learning: What is a Sports Car?
##########################################################

require(MASS)
library(plyr)
library(stringr)
library(plotrix)
library(matrixStats)
library(keras)

#####################################
##### loss functions
#####################################

Frobenius.loss <- function(X1,X2){sqrt(sum(as.matrix((X1-X2)^2))/nrow(X1))}
mean.squared.loss <- function(X1,X2){mean(as.matrix((X1-X2)^2))}


#####################################
##### bottleneck one layer
#####################################


bottleneck.1 <- function(q00, q22){
   Input <- layer_input(shape = c(q00), dtype = 'float32', name = 'Input')
   
   Output = Input %>% 
          layer_dense(units=q22, activation='tanh', use_bias=FALSE, name='Bottleneck') %>% 
          layer_dense(units=q00, activation='linear', use_bias=FALSE, name='Output')

   model <- keras_model(inputs = Input, outputs = Output)
   
   model %>% compile(optimizer = optimizer_nadam(), loss = 'mean_squared_error')
   model
   }

#####################################
##### bottleneck three layers
#####################################


bottleneck.3 <- function(q00, q11, q22){   
   Input <- layer_input(shape = c(q00), dtype = 'float32', name = 'Input')
   
   Encoder = Input %>% 
          layer_dense(units=q11, activation='tanh', use_bias=FALSE, name='Layer1') %>%
          layer_dense(units=q22, activation='tanh', use_bias=FALSE, name='Bottleneck') 

   Decoder = Encoder %>% 
          layer_dense(units=q11, activation='tanh', use_bias=FALSE, name='Layer3') %>% 
          layer_dense(units=q00, activation='linear', use_bias=FALSE, name='Output')

   model <- keras_model(inputs = Input, outputs = Decoder)
   model %>% compile(optimizer = optimizer_nadam(), loss = 'mean_squared_error')
   model
   }


