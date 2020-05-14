#===============================================
# Peeking into the Black Box
# Modeling
# Author: Michael Mayer
# Version from: May 8, 2020
#===============================================

library(dplyr)      # 0.8.5
library(splines)    # 3.6.3
library(splitTools) # 0.2.0
library(xgboost)    # 1.0.0.2
library(keras)      # 2.2.5.0
# reticulate::py_discover_config("tensorflow") # python 3.6.10
# tensorflow::tf_version() # >= 2.0

#===============================================
# Stratified split
#===============================================

ind <- partition(dat[["group_id"]], p = c(train = 0.8, test = 0.2), 
                 seed = 22, type = "grouped")
train <- dat[ind$train, ]
test <- dat[ind$test, ]

#===============================================
# GLM
# Using "quasipoisson" avoids warning about
# non-integer response. Has no impact on
# coefficients/predictions
#===============================================

fit_glm <- glm(Freq ~ VehPower + ns(VehAge, 5) + VehBrand +
                 VehGas + ns(DrivAge, 5) + logDensity + PolicyRegion,
               data = train,
               family = quasipoisson(),
               weights = train[[w]])

#===============================================
#  XGBoost GBM
#===============================================

# Input maker
prep_xgb <- function(dat, x) {
  data.matrix(dat[, x, drop = FALSE])
}

# Data interface to XGBoost
dtrain <- xgb.DMatrix(prep_xgb(train, x), 
                      label = train[[y]], 
                      weight = train[[w]])

# Parameters chosen by 5-fold grouped CV
params_freq <- list(learning_rate = 0.2,
                    max_depth = 5,
                    alpha = 3,
                    lambda = 0.5,
                    max_delta_step = 2,
                    min_split_loss = 0,
                  #  monotone_constraints = c(0,-1,0,0,0,0,0), 
                  #  interaction_constraints = list(4, c(0, 1, 2, 3, 5, 6)),
                    colsample_bytree = 1,
                    subsample = 0.9)

# Fit
set.seed(1)
fit_xgb <- xgb.train(params_freq, 
                     data = dtrain,
                     nrounds = 580,
                     objective = "count:poisson",
                     watchlist = list(train = dtrain),
                     print_every_n = 10)

# Save and load model
# xgb.save(fit_xgb, "xgb.model") 
# fit_xgb <- xgb.load("xgb.model")

#===============================================
# NEURAL NET (not fully reproducible)
#===============================================

# Input list maker
prep_nn <- function(dat, x, cat_cols = c("PolicyRegion", "VehBrand")) {
  dense_cols <- setdiff(x, cat_cols)
  c(list(dense1 = data.matrix(dat[, dense_cols])), 
    lapply(dat[, cat_cols], function(z) as.integer(z) - 1))
}

# Initialize neural net
new_neural_net <- function() {
  k_clear_session()
  set.seed(1)
  if ("set_seed" %in% names(tensorflow::tf$random)) {
    tensorflow::tf$random$set_seed(0)
  } else if ("set_random_seed" %in% names(tensorflow::tf$random)) {
    tensorflow::tf$random$set_random_seed(0)
  } else {
    print("Check tf version")
  }
  
  # Model architecture
  dense_input <- layer_input(5, name = "dense1", dtype = "float32")
  PolicyRegion_input <- layer_input(1, name = "PolicyRegion", dtype = "int8")
  VehBrand_input <- layer_input(1, name = "VehBrand", dtype = "int8")

  PolicyRegion_emb <- PolicyRegion_input %>% 
    layer_embedding(22, 1) %>% 
    layer_flatten()
  
  VehBrand_emb <- VehBrand_input %>% 
    layer_embedding(11, 1) %>% 
    layer_flatten()

  outputs <- list(dense_input, PolicyRegion_emb, VehBrand_emb) %>% 
    layer_concatenate() %>% 
    layer_dense(20, activation = "tanh") %>%
    layer_dense(15, activation = "tanh") %>%
    layer_dense(10, activation = "tanh") %>% 
    layer_dense(1, activation = "exponential")
  
  inputs <- list(dense1 = dense_input, 
                 PolicyRegion = PolicyRegion_input, 
                 VehBrand = VehBrand_input)
  
  model <- keras_model(inputs, outputs)
  
  model %>% 
    compile(loss = loss_poisson,
            optimizer = optimizer_nadam(),
            weighted_metrics = "poisson")
  
  return(model)
}

neural_net <- new_neural_net()

neural_net %>% 
  summary()

history <- neural_net %>% 
  fit(x = prep_nn(train, x), 
      y = train[, y], 
      sample_weight = train[, w],
      batch_size = 1e4, 
      epochs = 300,
      verbose = 2)  
    
plot(history)

# Calibrate by using last hidden layer activations as GLM input encoder
encoder <- keras_model(inputs = neural_net$input, 
                       outputs = get_layer(neural_net, "dense_2")$output)

# Creates input for calibration GLM (extends prep_nn)
prep_nn_calib <- function(dat, x, cat_cols = c("PolicyRegion", "VehBrand"), 
                          enc = encoder) {
  prep_nn(dat, x, cat_cols) %>% 
    predict(enc, ., batch_size = 1e4) %>% 
    data.frame()
}

# Calibration GLM
fit_nn <- glm(Freq ~ .,
              data = cbind(train["Freq"], prep_nn_calib(train, x)), 
              family = quasipoisson(), 
              weights = train[[w]])

# Save and load model
# save_model_weights_tf("neural_net.ckpt")
# neural_net <- new_neural_net() %>% 
#    load_model_weights_tf("neural_net.ckpt")


