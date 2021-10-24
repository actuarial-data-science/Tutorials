## -----------------------------------------------------------------------------
#library(mgcv)
library(keras)
library(magrittr)
library(dplyr)
library(tibble)
library(purrr)
library(ggplot2)
library(gridExtra)
library(splitTools)
library(tidyr)


## -----------------------------------------------------------------------------
# plotting parameters in R Markdown notebook
knitr::opts_chunk$set(fig.width = 9, fig.height = 9)
# plotting parameters in Jupyter notebook
library(repr)  # only needed for Jupyter notebook
options(repr.plot.width = 9, repr.plot.height = 9)


## -----------------------------------------------------------------------------
options(encoding = 'UTF-8')


## -----------------------------------------------------------------------------
# set seed to obtain best reproducibility. note that the underlying architecture may affect results nonetheless, so full reproducibility cannot be guaranteed across different platforms.
seed <- 100
Sys.setenv(PYTHONHASHSEED = seed)
set.seed(seed)
reticulate::py_set_seed(seed)
tensorflow::tf$random$set_seed(seed)


## -----------------------------------------------------------------------------
summarize <- function(...) suppressMessages(dplyr::summarize(...))


## -----------------------------------------------------------------------------
load_data <- function(file) {
  load(file.path("../0_data/", file), envir = parent.frame(1))
}


## -----------------------------------------------------------------------------
# Poisson deviance
PoissonDeviance <- function(pred, obs) {
    200 * (sum(pred) - sum(obs) + sum(log((obs/pred)^(obs)))) / length(pred)
}


## -----------------------------------------------------------------------------
plot_freq <- function(test, xvar, title, model, mdlvariant) {
  out <- test %>% group_by(!!sym(xvar)) %>%
            summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(!!sym(mdlvariant)) / sum(Exposure))
  
  ggplot(out, aes(x = !!sym(xvar), group = 1)) +
    geom_point(aes(y = pred, colour = model)) +
    geom_point(aes(y = obs, colour = "observed")) + 
    geom_line(aes(y = pred, colour = model), linetype = "dashed") +
    geom_line(aes(y = obs, colour = "observed"), linetype = "dashed") +
    ylim(0, 0.35) + labs(x = xvar, y = "frequency", title = title) +
    theme(legend.position = "bottom")
}


## -----------------------------------------------------------------------------
plot_loss <- function(x) {
  if (length(x) > 1) {
    df_val <- data.frame(epoch = 1:length(x$loss), train_loss = x$loss, val_loss = x$val_loss)
    df_val <- gather(df_val, variable, loss, -epoch)
    p <- ggplot(df_val, aes(x = epoch, y = loss)) + geom_line() +
      facet_wrap(~variable, scales = "free") + geom_smooth()
    suppressMessages(print(p))
  }
}


## -----------------------------------------------------------------------------
load_data("freMTPL2freq.RData")


## -----------------------------------------------------------------------------
# Grouping id
distinct <- freMTPL2freq %>% 
  distinct_at(vars(-c(IDpol, Exposure, ClaimNb))) %>% 
  mutate(group_id = row_number())


## -----------------------------------------------------------------------------
dat <- freMTPL2freq %>% 
  left_join(distinct) %>% 
  mutate(ClaimNb = pmin(as.integer(ClaimNb), 4),
         VehAge = pmin(VehAge, 20),
         DrivAge = pmin(DrivAge, 90),
         BonusMalus = pmin(BonusMalus, 150),
         Density = round(log(Density), 2),
         VehGas = factor(VehGas),
         Exposure = pmin(Exposure, 1))


## -----------------------------------------------------------------------------
# Group sizes of suspected clusters
table(table(dat[, "group_id"]))


## -----------------------------------------------------------------------------
dat2 <- dat %>% mutate(
  AreaGLM = as.integer(Area),
  VehPowerGLM = as.factor(pmin(VehPower, 9)),
  VehAgeGLM = cut(VehAge, breaks = c(-Inf, 0, 10, Inf), labels = c("1","2","3")),
  DrivAgeGLM = cut(DrivAge, breaks = c(-Inf, 20, 25, 30, 40, 50, 70, Inf), labels = c("1","2","3","4","5","6","7")),
  BonusMalusGLM = as.integer(pmin(BonusMalus, 150)),
  DensityGLM = as.numeric(Density),
  VehAgeGLM = relevel(VehAgeGLM, ref = "2"),   
  DrivAgeGLM = relevel(DrivAgeGLM, ref = "5"),
  Region = relevel(Region, ref = "R24")
)


## -----------------------------------------------------------------------------
knitr::kable(head(dat2))


## -----------------------------------------------------------------------------
str(dat2)


## -----------------------------------------------------------------------------
summary(dat2)


## -----------------------------------------------------------------------------
ind <- partition(dat2[["group_id"]], p = c(train = 0.8, test = 0.2), 
                 seed = seed, type = "grouped")
train <- dat2[ind$train, ]
test <- dat2[ind$test, ]


## -----------------------------------------------------------------------------
# size of train/test
sprintf("Number of observations (train): %s", nrow(train))
sprintf("Number of observations (test): %s", nrow(test))

# Claims frequency of train/test
sprintf("Empirical frequency (train): %s", round(sum(train$ClaimNb) / sum(train$Exposure), 4))
sprintf("Empirical frequency (test): %s", round(sum(test$ClaimNb) / sum(test$Exposure), 4))


## -----------------------------------------------------------------------------
# initialize table to store all model results for comparison
df_cmp <- tibble(
 model = character(),
 epochs = numeric(),
 run_time = numeric(),
 parameters = numeric(),
 in_sample_loss = numeric(),
 out_sample_loss = numeric(),
 avg_freq = numeric(),
)


## -----------------------------------------------------------------------------
exec_time <- system.time(
  glm2 <- glm(ClaimNb ~ AreaGLM + VehPowerGLM + VehAgeGLM + BonusMalusGLM + VehBrand +
                        VehGas + DensityGLM + Region + DrivAge + log(DrivAge) +
                        I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4),
              data = train, offset = log(Exposure), family = poisson())
)
exec_time[1:5]
summary(glm2)


## -----------------------------------------------------------------------------
drop1(glm2, test = "LRT")


## -----------------------------------------------------------------------------
# Predictions
train$fitGLM2 <- fitted(glm2)
test$fitGLM2 <- predict(glm2, newdata = test, type = "response")
dat$fitGLM2 <- predict(glm2, newdata = dat2, type = "response")


## -----------------------------------------------------------------------------
# in-sample and out-of-sample losses (in 10^(-2))
sprintf("100 x Poisson deviance GLM (train): %s", PoissonDeviance(train$fitGLM2, train$ClaimNb))
sprintf("100 x Poisson deviance GLM (test): %s", PoissonDeviance(test$fitGLM2, test$ClaimNb))

# Overall estimated frequency
sprintf("average frequency (test): %s", round(sum(test$fitGLM2) / sum(test$Exposure), 4))


## -----------------------------------------------------------------------------
df_cmp %<>% bind_rows(
  data.frame(model = "M1: GLM", epochs = NA, run_time = round(exec_time[[3]], 0), parameters = length(coef(glm2)),
             in_sample_loss = round(PoissonDeviance(train$fitGLM2, as.vector(unlist(train$ClaimNb))), 4),
             out_sample_loss = round(PoissonDeviance(test$fitGLM2, as.vector(unlist(test$ClaimNb))), 4),
             avg_freq = round(sum(test$fitGLM2) / sum(test$Exposure), 4))
)
knitr::kable(df_cmp)


## -----------------------------------------------------------------------------
# Area
p1 <- plot_freq(test, "Area", "frequency by area", "GLM", "fitGLM2")
# VehPower
p2 <- plot_freq(test, "VehPower", "frequency by vehicle power", "GLM", "fitGLM2")
# VehBrand
p3 <- plot_freq(test, "VehBrand", "frequency by vehicle brand", "GLM", "fitGLM2")
# VehAge
p4 <- plot_freq(test, "VehAge", "frequency by vehicle age", "GLM", "fitGLM2")

grid.arrange(p1, p2, p3, p4)


## -----------------------------------------------------------------------------
# MinMax scaler
preprocess_minmax <- function(varData) {
  X <- as.numeric(varData)
  2 * (X - min(X)) / (max(X) - min(X)) - 1
}

# Dummy coding 
preprocess_catdummy <- function(data, varName, prefix) {
  varData <- data[[varName]]
  X <- as.integer(varData)
  n0 <- length(unique(X))
  n1 <- 2:n0
  addCols <- purrr::map(n1, function(x, y) {as.integer(y == x)}, y = X) %>%
    rlang::set_names(paste0(prefix, n1))
  cbind(data, addCols)
}

# Feature pre-processing using MinMax Scaler and Dummy Coding
preprocess_features <- function(data) {
  data %>%
    mutate_at(
      c(AreaX = "Area", VehPowerX = "VehPower", VehAgeX = "VehAge",
        DrivAgeX = "DrivAge", BonusMalusX = "BonusMalus", DensityX = "Density"),
      preprocess_minmax
    ) %>%
    mutate(
      VehGasX = as.integer(VehGas) - 1.5
    ) %>%
    preprocess_catdummy("VehBrand", "Br") %>%
    preprocess_catdummy("Region", "R")
}


## -----------------------------------------------------------------------------
dat2 <- preprocess_features(dat)


## -----------------------------------------------------------------------------
knitr::kable(head(dat2))


## -----------------------------------------------------------------------------
str(dat2)


## -----------------------------------------------------------------------------
summary(dat2)


## -----------------------------------------------------------------------------
ind <- partition(dat2[["group_id"]], p = c(train = 0.8, test = 0.2), 
                 seed = seed, type = "grouped")
train <- dat2[ind$train, ]
test <- dat2[ind$test, ]


## -----------------------------------------------------------------------------
# size of train/test
sprintf("Number of observations (train): %s", nrow(train))
sprintf("Number of observations (test): %s", nrow(test))

# Claims frequency of train/test
sprintf("Empirical frequency (train): %s", round(sum(train$ClaimNb) / sum(train$Exposure), 4))
sprintf("Empirical frequency (test): %s", round(sum(test$ClaimNb) / sum(test$Exposure), 4))


## -----------------------------------------------------------------------------
# select the feature space
col_start <- ncol(dat) + 1
col_end <- ncol(dat2)
features <- c(col_start:col_end)  # select features, be careful if pre-processing changes
print(colnames(train[, features]))


## -----------------------------------------------------------------------------
# feature matrix
Xtrain <- as.matrix(train[, features])  # design matrix training sample
Xtest <- as.matrix(test[, features])    # design matrix test sample


## -----------------------------------------------------------------------------
# available optimizers for keras
# https://keras.io/optimizers/
optimizers <- c('sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam', 'adamax', 'nadam')


## -----------------------------------------------------------------------------
# homogeneous model (train)
lambda_hom <- sum(train$ClaimNb) / sum(train$Exposure)


## -----------------------------------------------------------------------------
# define network and load pre-specified weights
q0 <- length(features)                  # dimension of features
q1 <- 20                                # number of hidden neurons in hidden layer

sprintf("Neural network with K=1 hidden layer")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons: q1 = %s", q1)
sprintf("Output dimension: %s", 1)


## -----------------------------------------------------------------------------
Design  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 
LogVol  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

Network <- Design %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

Response <- list(Network, LogVol) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

model_sh <- keras_model(inputs = c(Design, LogVol), outputs = c(Response))


## -----------------------------------------------------------------------------
model_sh %>% compile(
  loss = 'poisson',
  optimizer = optimizers[7]
)

summary(model_sh)


## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 300
batch_size <- 10000
validation_split <- 0.2  # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# expected run-time on Renku 8GB environment around 70 seconds
exec_time <- system.time(
  fit <- model_sh %>% fit(
    list(Xtrain, as.matrix(log(train$Exposure))), as.matrix(train$ClaimNb),
    epochs = epochs,
    batch_size = batch_size,
    validation_split = validation_split,
    verbose = verbose
  )
)
exec_time[1:5]


## ---- fig.height=6------------------------------------------------------------
plot(fit)


## ---- fig.height=6------------------------------------------------------------
plot_loss(x = fit[[2]])


## -----------------------------------------------------------------------------
# calculating the predictions
train$fitshNN <- as.vector(model_sh %>% predict(list(Xtrain, as.matrix(log(train$Exposure)))))
test$fitshNN <- as.vector(model_sh %>% predict(list(Xtest, as.matrix(log(test$Exposure)))))

# average in-sample and out-of-sample losses (in 10^(-2))
sprintf("100 x Poisson deviance shallow network (train): %s", PoissonDeviance(train$fitshNN, train$ClaimNb))
sprintf("100 x Poisson deviance shallow network (test): %s", PoissonDeviance(test$fitshNN, test$ClaimNb))

# average frequency
sprintf("Average frequency (test): %s", round(sum(test$fitshNN) / sum(test$Exposure), 4))


## -----------------------------------------------------------------------------
trainable_params <- sum(unlist(lapply(model_sh$trainable_weights, k_count_params)))
df_cmp %<>% bind_rows(
  data.frame(model = "M2: Shallow Plain Network", epochs = epochs,
             run_time = round(exec_time[[3]], 0), parameters = trainable_params,
             in_sample_loss = round(PoissonDeviance(train$fitshNN, train$ClaimNb), 4),
             out_sample_loss = round(PoissonDeviance(test$fitshNN, test$ClaimNb), 4),
             avg_freq = round(sum(test$fitshNN) / sum(test$Exposure), 4)
  ))

knitr::kable(df_cmp)


## -----------------------------------------------------------------------------
# Area
p1 <- plot_freq(test, "Area", "Frequency by area", "shNN", "fitshNN")
# VehPower
p2 <- plot_freq(test, "VehPower", "Frequency by vehicle power", "shNN", "fitshNN")
# VehBrand
p3 <- plot_freq(test, "VehBrand", "Frequency by vehicle brand", "shNN", "fitshNN")
# VehAge
p4 <- plot_freq(test, "VehAge", "Frequency by vehicle age", "shNN", "fitshNN")

grid.arrange(p1, p2, p3, p4)


## -----------------------------------------------------------------------------
# define network
q0 <- length(features)   # dimension of features
q1 <- 20                 # number of neurons in first hidden layer
q2 <- 15                 # number of neurons in second hidden layer
q3 <- 10                 # number of neurons in second hidden layer

sprintf("Neural network with K=3 hidden layer")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", q1)
sprintf("Number of hidden neurons second layer: q2 = %s", q2)
sprintf("Number of hidden neurons third layer: q3 = %s", q3)
sprintf("Output dimension: %s", 1)


## -----------------------------------------------------------------------------
Design <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 
LogVol <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

Network <- Design %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network', 
              weights = list(array(0, dim = c(q3, 1)), array(log(lambda_hom), dim = c(1))))

Response <- list(Network, LogVol) %>%
  layer_add(name = 'Add') %>% 
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

model_dp <- keras_model(inputs = c(Design, LogVol), outputs = c(Response))


## -----------------------------------------------------------------------------
model_dp %>% compile(
  loss = 'poisson',
  optimizer = optimizers[7]
)

summary(model_dp)


## -----------------------------------------------------------------------------
# select number of epochs and batch_size
epochs <- 300
batch_size <- 10000
validation_split <- 0.2  # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# expected run-time on Renku 8GB environment around 200 seconds
exec_time <- system.time(
  fit <- model_dp %>% fit(
    list(Xtrain, as.matrix(log(train$Exposure))), as.matrix(train$ClaimNb),
    epochs = epochs,
    batch_size = batch_size,
    validation_split = validation_split,
    verbose = verbose
  )
)
exec_time[1:5]


## ---- fig.height=6------------------------------------------------------------
plot(fit)


## ---- fig.height=6------------------------------------------------------------
plot_loss(x = fit[[2]])


## -----------------------------------------------------------------------------
# Validation: Poisson deviance
train$fitdpNN <- as.vector(model_dp %>% predict(list(Xtrain, as.matrix(log(train$Exposure)))))
test$fitdpNN <- as.vector(model_dp %>% predict(list(Xtest, as.matrix(log(test$Exposure)))))

sprintf("100 x Poisson deviance shallow network (train): %s", PoissonDeviance(train$fitdpNN, train$ClaimNb))
sprintf("100 x Poisson deviance shallow network (test): %s", PoissonDeviance(test$fitdpNN, test$ClaimNb))

# average frequency
sprintf("Average frequency (test): %s", round(sum(test$fitdpNN) / sum(test$Exposure), 4))


## -----------------------------------------------------------------------------
trainable_params <- sum(unlist(lapply(model_dp$trainable_weights, k_count_params)))
df_cmp %<>% bind_rows(
  data.frame(model = "M3: Deep Plain Network", epochs = epochs,
             run_time = round(exec_time[[3]], 0), parameters = trainable_params,
             in_sample_loss = round(PoissonDeviance(train$fitdpNN, train$ClaimNb), 4),
             out_sample_loss = round(PoissonDeviance(test$fitdpNN, test$ClaimNb), 4),
             avg_freq = round(sum(test$fitdpNN) / sum(test$Exposure), 4)
  ))
knitr::kable(df_cmp)


## -----------------------------------------------------------------------------
# Area
p1 <- plot_freq(test, "Area", "frequency by area", "dpNN", "fitdpNN")
# VehPower
p2 <- plot_freq(test, "VehPower", "frequency by vehicle power", "dpNN", "fitdpNN")
# VehBrand
p3 <- plot_freq(test, "VehBrand", "frequency by vehicle brand", "dpNN", "fitdpNN")
# VehAge
p4 <- plot_freq(test, "VehAge", "frequency by vehicle age", "dpNN", "fitdpNN")

grid.arrange(p1, p2, p3, p4)


## -----------------------------------------------------------------------------
# define network
q0 <- length(features)  # dimension of input features
q1 <- 20                # number of neurons in first hidden layer
q2 <- 15                # number of neurons in second hidden layer
q3 <- 10                # number of neurons in second hidden layer
p0 <- 0.05              # dropout rate     

sprintf("Neural network with K=3 hidden layer and 3 dropout layers")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", q1)
sprintf("Number of hidden neurons second layer: q2 = %s", q2)
sprintf("Number of hidden neurons third layer: q3 = %s", q3)
sprintf("Output dimension: %s", 1)


## -----------------------------------------------------------------------------
Design <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design')
LogVol <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

Network <- Design %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dropout(rate = p0) %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dropout(rate = p0) %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dropout(rate = p0) %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network', 
              weights = list(array(0, dim = c(q3, 1)), array(log(lambda_hom), dim = c(1))))

Response <- list(Network, LogVol) %>%
  layer_add(name = 'Add') %>% 
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE, 
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

model_dr <- keras_model(inputs = c(Design, LogVol), outputs = c(Response))


## -----------------------------------------------------------------------------
model_dr %>% compile(
  loss = 'poisson',
  optimizer = optimizers[7]
)

summary(model_dr)


## -----------------------------------------------------------------------------
# select number of epochs and batch_size
epochs <- 500
batch_size <- 10000
validation_split <- 0.2  # set >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# expected run-time on Renku 8GB environment around 110 seconds (for 100 epochs)
exec_time <- system.time(
  fit <- model_dr %>% fit(
    list(Xtrain, as.matrix(log(train$Exposure))), as.matrix(train$ClaimNb),
    epochs = epochs,
    batch_size = batch_size,
    validation_split = validation_split,
    verbose = verbose
  )
)
exec_time[1:5]


## ---- fig.height=6------------------------------------------------------------
plot(fit)


## ---- fig.height=6------------------------------------------------------------
plot_loss(x = fit[[2]])


## -----------------------------------------------------------------------------
# Validation: Poisson deviance
train$fitdrNN <- as.vector(model_dr %>% predict(list(Xtrain, as.matrix(log(train$Exposure)))))
test$fitdrNN <- as.vector(model_dr %>% predict(list(Xtest, as.matrix(log(test$Exposure)))))

sprintf("100 x Poisson deviance shallow network (train): %s", PoissonDeviance(train$fitdrNN, train$ClaimNb))
sprintf("100 x Poisson deviance shallow network (test): %s", PoissonDeviance(test$fitdrNN, test$ClaimNb))

# average frequency
sprintf("Average frequency (test): %s", round(sum(test$fitdrNN) / sum(test$Exposure), 4))


## -----------------------------------------------------------------------------
trainable_params <- sum(unlist(lapply(model_dr$trainable_weights, k_count_params)))
df_cmp %<>% bind_rows(
  data.frame(model = "M4: Deep Dropout Network", epochs = epochs,
             run_time = round(exec_time[[3]], 0), parameters = trainable_params,
             in_sample_loss = round(PoissonDeviance(train$fitdrNN, train$ClaimNb), 4),
             out_sample_loss = round(PoissonDeviance(test$fitdrNN, test$ClaimNb), 4),
             avg_freq = round(sum(test$fitdrNN) / sum(test$Exposure), 4)
  ))
knitr::kable(df_cmp)


## -----------------------------------------------------------------------------
# Area
p1 <- plot_freq(test, "Area", "frequency by area", "drNN", "fitdrNN")
# VehPower
p2 <- plot_freq(test, "VehPower", "frequency by vehicle power", "drNN", "fitdrNN")
# VehBrand
p3 <- plot_freq(test, "VehBrand", "frequency by vehicle brand", "drNN", "fitdrNN")
# VehAge
p4 <- plot_freq(test, "VehAge", "frequency by vehicle age", "drNN", "fitdrNN")

grid.arrange(p1, p2, p3, p4)


## -----------------------------------------------------------------------------
knitr::kable(df_cmp)


## ---- fig.width=12, fig.height=18---------------------------------------------
plot_cmp <- function(xvar, title, maxlim = 0.35) {
  out <- test %>% group_by(!!sym(xvar)) %>% summarize(
    vol = sum(Exposure),
    obs = sum(ClaimNb) / sum(Exposure),
    glm = sum(fitGLM2) / sum(Exposure),
    shNN = sum(fitshNN) / sum(Exposure),
    dpNN = sum(fitdpNN) / sum(Exposure),
    drNN = sum(fitdrNN) / sum(Exposure)
  )
  
  max_pri <- max(out$obs, out$glm, out$shNN, out$dpNN, out$drNN)
  max_sec <- 1.1 * max(out$vol)
  max_ratio <- max_pri / max_sec
  
  if (is.null(maxlim)) maxlim <- max_pri
  
  ggplot(out, aes(x = !!sym(xvar), group = 1)) +
    geom_point(aes(y = obs, colour = "observed")) + geom_line(aes(y = obs, colour = "observed"), linetype = "dashed") +
    geom_point(aes(y = glm, colour = "GLM")) + geom_line(aes(y = glm, colour = "GLM"), linetype = "dashed") +
    geom_point(aes(y = shNN, colour = "shNN")) + geom_line(aes(y = shNN, colour = "shNN"), linetype = "dashed") +
    geom_point(aes(y = dpNN, colour = "dpNN")) + geom_line(aes(y = dpNN, colour = "dpNN"), linetype = "dashed") +
    geom_point(aes(y = drNN, colour = "drNN")) + geom_line(aes(y = drNN, colour = "drNN"), linetype = "dashed") +
    geom_bar(aes(y = vol * (max_ratio)), colour = "grey", stat = "identity", alpha = 0.3) +
    scale_y_continuous(name = "Frequency", sec.axis = sec_axis( ~ . / (max_ratio), name = "Exposure"), limits = c(0, maxlim)) +
    labs(x = xvar, title = title) + theme(legend.position = "bottom", legend.text = element_text(size = 6))
}

# Area
p1 <- plot_cmp("Area", "Frequency and Exposure by Area")
# VehPower
p2 <- plot_cmp("VehPower", "Frequency and Exposure by VehPower")
# VehBrand
p3 <- plot_cmp("VehBrand", "Frequency and Exposure by VehBrand")
# VehAge
p4 <- plot_cmp("VehAge", "Frequency and Exposure by VehAge")
# DrivAge plot with exposure distribution
p5 <- plot_cmp("DrivAge", "Frequency and Exposure by DrivAge")

grid.arrange(p1, p2, p3, p4, p5)


## -----------------------------------------------------------------------------
plot_claims_freq <- function(xvar, yvar, xlab, ylab) {
  axis_min <- log(max(test[[xvar]], test[[yvar]]))
  axis_max <- log(min(test[[xvar]], test[[yvar]]))
  
  ggplot(test, aes(x = log(!!sym(xvar)), y = log(!!sym(yvar)), colour = Exposure)) + geom_point() +
    geom_abline(colour = "#000000", slope = 1, intercept = 0) +
    xlim(axis_max, axis_min) + ylim(axis_max, axis_min) +
    labs(x = xlab, y = ylab, title = "Claims frequency prediction (log-scale)") +
    scale_colour_gradient(low = "green", high = "red")
}


## -----------------------------------------------------------------------------
plot_claims_freq("fitshNN", "fitdpNN", "shNN", "dpNN")


## -----------------------------------------------------------------------------
plot_claims_freq("fitshNN", "fitGLM2", "shNN", "GLM")


## -----------------------------------------------------------------------------
plot_claims_freq("fitdpNN", "fitGLM2", "dpNN", "GLM")


## -----------------------------------------------------------------------------
plot_claims_freq("fitdrNN", "fitdpNN", "drNN", "dpNN")


## -----------------------------------------------------------------------------
sessionInfo()


## -----------------------------------------------------------------------------
reticulate::py_config()


## -----------------------------------------------------------------------------
tensorflow::tf_version()

