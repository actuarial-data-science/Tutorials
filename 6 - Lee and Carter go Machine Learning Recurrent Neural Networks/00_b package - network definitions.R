#### Purpose: Library defining the different networks considered
#### Authors: Ronald Richman and Mario Wuthrich
#### Date: August 12, 2019


##################################################################
### Designing an LSTMs
##################################################################

library(keras)

# single hidden LSTM layer
LSTM1 <- function(T0, tau0, tau1, y0=0, optimizer){
    Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
    Output = Input %>%  
       layer_lstm(units=tau1, activation='tanh', recurrent_activation='tanh', name='LSTM1') %>%
       layer_dense(units=1, activation=k_exp, name="Output",
                   weights=list(array(0,dim=c(tau1,1)), array(log(y0),dim=c(1)))) 
    model <- keras_model(inputs = list(Input), outputs = c(Output))
    model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
     }
     

# double hidden LSTM layers
LSTM2 <- function(T0, tau0, tau1, tau2, y0=0, optimizer){
    Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
    Output = Input %>%  
       layer_lstm(units=tau1, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='LSTM1') %>%
       layer_lstm(units=tau2, activation='tanh', recurrent_activation='tanh', name='LSTM2') %>%           
       layer_dense(units=1, activation=k_exp, name="Output",
                   weights=list(array(0,dim=c(tau2,1)), array(log(y0),dim=c(1))))  
    model <- keras_model(inputs = list(Input), outputs = c(Output))
    model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
     }


# triple hidden LSTM layers
LSTM3 <- function(T0, tau0, tau1, tau2, tau3, y0=0, optimizer){
    Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
    Output = Input %>%  
       layer_lstm(units=tau1, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='LSTM1') %>%
       layer_lstm(units=tau2, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='LSTM2') %>%           
       layer_lstm(units=tau3, activation='tanh', recurrent_activation='tanh', name='LSTM3') %>%           
       layer_dense(units=1, activation=k_exp, name="Output",
                   weights=list(array(0,dim=c(tau3,1)), array(log(y0),dim=c(1))))  
    model <- keras_model(inputs = list(Input), outputs = c(Output))
    model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
     }
     

# time-distributed layer on LSTMs
LSTM_TD <- function(T0, tau0, tau1, y0=0, optimizer){
    Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
    Output = Input %>%  
       layer_lstm(units=tau1, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='LSTM1') %>%
       time_distributed(layer_dense(units=1, activation=k_exp, name="Output"), name='TD') 
    model <- keras_model(inputs = list(Input), outputs = c(Output))
    model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
     }
     
# single hidden GRU layer
GRU1 <- function(T0, tau0, tau1, y0=0, optimizer){
    Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
    Output = Input %>%  
       layer_gru(units=tau1, activation='tanh', recurrent_activation='tanh', name='GRU1') %>%
       layer_dense(units=1, activation=k_exp, name="Output",
                   weights=list(array(0,dim=c(tau1,1)), array(log(y0),dim=c(1)))) 
    model <- keras_model(inputs = list(Input), outputs = c(Output))
    model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
     }
     

# double hidden GRU layers
GRU2 <- function(T0, tau0, tau1, tau2, y0=0, optimizer){
    Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
    Output = Input %>%  
       layer_gru(units=tau1, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='GRU1') %>%
       layer_gru(units=tau2, activation='tanh', recurrent_activation='tanh', name='GRU2') %>%           
       layer_dense(units=1, activation=k_exp, name="Output",
          weights=list(array(0,dim=c(tau2,1)), array(log(y0),dim=c(1)))) 
    model <- keras_model(inputs = list(Input), outputs = c(Output))
    model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
     }
     
# triple hidden GRU layers
GRU3 <- function(T0, tau0, tau1, tau2, tau3, y0=0, optimizer){
    Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
    Output = Input %>%  
       layer_gru(units=tau1, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='GRU1') %>%
       layer_gru(units=tau2, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='GRU2') %>%           
       layer_gru(units=tau3, activation='tanh', recurrent_activation='tanh', name='GRU3') %>%           
       layer_dense(units=1, activation=k_exp, name="Output",
          weights=list(array(0,dim=c(tau3,1)), array(log(y0),dim=c(1)))) 
    model <- keras_model(inputs = list(Input), outputs = c(Output))
    model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
     }

# triple hidden LSTM layers both genders
LSTM3.Gender <- function(T0, tau0, tau1, tau2, tau3, y0=0, optimizer){
    Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
    Gender <- layer_input(shape=c(1), dtype='float32', name='Gender') 
    RNN = Input %>%  
       layer_lstm(units=tau1, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='LSTM1') %>%
       layer_lstm(units=tau2, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='LSTM2') %>%           
       layer_lstm(units=tau3, activation='tanh', recurrent_activation='tanh', name='LSTM3')  
       Output = list(RNN, Gender) %>% layer_concatenate(name="Concat")%>%                    
       layer_dense(units=1, activation=k_exp, name="Output",
                   weights=list(array(0,dim=c(tau3+1,1)), array(log(y0),dim=c(1))))  
    model <- keras_model(inputs = list(Input, Gender), outputs = c(Output))
    model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
     }


# triple hidden GRU layers both gender
GRU3.Gender <- function(T0, tau0, tau1, tau2, tau3, y0=0, optimizer){
    Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
    Gender <- layer_input(shape=c(1), dtype='float32', name='Gender') 
    RNN = Input %>%  
       layer_gru(units=tau1, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='GRU1') %>%
       layer_gru(units=tau2, activation='tanh', recurrent_activation='tanh', 
                  return_sequences=TRUE, name='GRU2') %>%           
       layer_gru(units=tau3, activation='tanh', recurrent_activation='tanh', name='GRU3') 
    Output = list(RNN, Gender) %>% layer_concatenate(name="Concat")%>%           
       layer_dense(units=1, activation=k_exp, name="Output",
          weights=list(array(0,dim=c(tau3+1,1)), array(log(y0),dim=c(1)))) 
    model <- keras_model(inputs = list(Input, Gender), outputs = c(Output))
    model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
     }


# double hidden FNN layers
FNN <- function(T0, tau0, tau1, tau2, y0=0, optimizer){
    Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input')
    Output = Input %>% layer_reshape(target_shape=c(T0*tau0), name='Reshape') %>%
       layer_dense(units=tau1, activation='tanh', name='Layer1') %>%
       layer_dense(units=tau2, activation='tanh', name='Layer2') %>%           
       layer_dense(units=1, activation=k_exp, name="Output",
                   weights=list(array(0,dim=c(tau2,1)), array(log(y0),dim=c(1)))) 
    model <- keras_model(inputs = list(Input), outputs = c(Output))
    model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
     }
     

