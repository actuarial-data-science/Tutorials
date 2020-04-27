##########################################
#########  Data Analysis French MTPL
#########  Insights from Inside Neural Networks
#########  Author: Mario Wuthrich
#########  Version April 22, 2020
##########################################

# Reference
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3226852

##########################################
#########  tools
##########################################

# loss function
Poisson.loss <- function(pred, obs){200*(mean(pred)-mean(obs)+mean(log((obs/pred)^obs)))}

# plot of gradient descent performance
plot.loss <- function(pos0, loss, ylim0, plot.yes=0, filey1, col0){
  if (plot.yes==1){pdf(filey1)}      
   plot(loss$val_loss,col=col0[2], ylim=ylim0, main=list("gradient descent algorithm losses", cex=1.5),xlab="epoch", ylab="loss", cex=1.5, cex.lab=1.5)
   lines(loss$loss,col=col0[1])
   legend(x=pos0, col=col0, lty=c(1,-1), lwd=c(1,-1), pch=c(-1,1), legend=c("training loss", "validation loss"))
if (plot.yes==1){dev.off()}          
   }

# regression function for bias regularization
glm.formula <- function(nb){
   string <- "yy ~ X1"
   if (nb>1){for (ll in 2:nb){ string <- paste(string, "+X",ll, sep="")}}
   string
    }  


##########################################
#########  networks 
##########################################

shallow.plain.vanilla <- function(seed, q0, y0){
    set.seed(seed)
    use_session_with_seed(seed)  
    design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 
    #
    output = design %>%
        layer_dense(units=q0[2], activation='tanh', name='layer1') %>%
        layer_dense(units=q0[3], activation='exponential', name='output', 
                    weights=list(array(0, dim=c(q0[2],q0[3])), array(y0, dim=c(q0[3]))))
    #
    model <- keras_model(inputs = c(design), outputs = c(output))
    model
    }

###################################################

deep2.plain.vanilla <- function(seed, q0, y0){
    set.seed(seed)
    use_session_with_seed(seed)  
    design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 
    #
    output = design %>%
        layer_dense(units=q0[2], activation='tanh', name='layer1') %>%
        layer_dense(units=q0[3], activation='tanh', name='layer2') %>%
        layer_dense(units=q0[4], activation='exponential', name='output', 
                    weights=list(array(0, dim=c(q0[3],q0[4])), array(y0, dim=c(q0[4]))))
    #
    model <- keras_model(inputs = c(design), outputs = c(output))
    model
    }


###################################################

deep3.plain.vanilla <- function(seed, q0, y0){
    set.seed(seed)
    use_session_with_seed(seed)  
    design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 
    #
    output = design %>%
        layer_dense(units=q0[2], activation='tanh', name='layer1') %>%
        layer_dense(units=q0[3], activation='tanh', name='layer2') %>%
        layer_dense(units=q0[4], activation='tanh', name='layer3') %>%
        layer_dense(units=q0[5], activation='exponential', name='output', 
                    weights=list(array(0, dim=c(q0[4],q0[5])), array(y0, dim=c(q0[5]))))
    #
    model <- keras_model(inputs = c(design), outputs = c(output))
    model
    }

###################################################

deep3.ridge <- function(seed, q0, w0, y0){
    set.seed(seed)
    use_session_with_seed(seed)  
    design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 
    #
    output = design %>%
        layer_dense(units=q0[2], kernel_regularizer=regularizer_l2(w0[1]), activation='tanh', name='layer1') %>%
        layer_dense(units=q0[3], kernel_regularizer=regularizer_l2(w0[2]), activation='tanh', name='layer2') %>%
        layer_dense(units=q0[4], kernel_regularizer=regularizer_l2(w0[3]), activation='tanh', name='layer3') %>%
        layer_dense(units=q0[5], activation='exponential', name='output', 
                    weights=list(array(0, dim=c(q0[4],q0[5])), array(y0, dim=c(q0[5]))))
    #
    model <- keras_model(inputs = c(design), outputs = c(output))
    model
    }
    
################################################### 

deep3.dropout <- function(seed, q0, w0, y0){
    set.seed(seed)
    use_session_with_seed(seed)  
    design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 
    #
    output = design %>%
        layer_dense(units=q0[2], activation='tanh', name='layer1') %>%
        layer_dropout (rate = w0[1]) %>%
        layer_dense(units=q0[3], activation='tanh', name='layer2') %>%
        layer_dropout (rate = w0[2]) %>%
        layer_dense(units=q0[4], activation='tanh', name='layer3') %>%
        layer_dropout (rate = w0[3]) %>%
        layer_dense(units=q0[5], activation='exponential', name='output', 
                    weights=list(array(0, dim=c(q0[4],q0[5])), array(y0, dim=c(q0[5]))))
    #
    model <- keras_model(inputs = c(design), outputs = c(output))
    model
    }


###################################################

deep3.ReLU <- function(seed, q0, y0){
    set.seed(seed)
    use_session_with_seed(seed)  
    design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 
    #
    output = design %>%
        layer_dense(units=q0[2], activation='relu', name='layer1') %>%
        layer_dense(units=q0[3], activation='relu', name='layer2') %>%
        layer_dense(units=q0[4], activation='relu', name='layer3') %>%
        layer_dense(units=q0[5], activation='exponential', name='output', 
                    weights=list(array(0, dim=c(q0[4],q0[5])), array(y0, dim=c(q0[5]))))
    #
    model <- keras_model(inputs = c(design), outputs = c(output))
    model
    }

################################################### 

deep3.norm.dropout <- function(seed, q0, w0, y0){
    set.seed(seed)
    use_session_with_seed(seed)  
    design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 
    #
    output = design %>%
        layer_dense(units=q0[2], activation='tanh', name='layer1') %>%
        layer_batch_normalization() %>%
        layer_dropout (rate = w0) %>%
        layer_dense(units=q0[3], activation='tanh', name='layer2') %>%
        layer_batch_normalization() %>%
        layer_dropout (rate = w0) %>%
        layer_dense(units=q0[4], activation='tanh', name='layer3') %>%
        layer_dropout (rate = w0) %>%
        layer_dense(units=q0[5], activation='exponential', name='output', 
                    weights=list(array(0, dim=c(q0[4],q0[5])), array(y0, dim=c(q0[5]))))
    #
    model <- keras_model(inputs = c(design), outputs = c(output))
    model
    }

