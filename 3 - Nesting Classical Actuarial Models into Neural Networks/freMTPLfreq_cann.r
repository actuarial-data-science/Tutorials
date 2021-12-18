library(mgcv)
library(keras)
library(magrittr)
library(dplyr)
library(tibble)
library(purrr)
library(ggplot2)
library(gridExtra)
library(splitTools)
library(tidyr)

options(encoding = 'UTF-8')

# set seed to obtain best reproducibility. note that the underlying architecture may affect results nonetheless, so full reproducibility cannot be guaranteed across different platforms.
seed <- 100
Sys.setenv(PYTHONHASHSEED = seed)
set.seed(seed)
reticulate::py_set_seed(seed)
tensorflow::tf$random$set_seed(seed)

summarize <- function(...) suppressMessages(dplyr::summarize(...))

# Poisson deviance
PoissonDeviance <- function(pred, obs) {
    200 * (sum(pred) - sum(obs) + sum(log((obs/pred)^(obs)))) / length(pred)
}

plot_freq <- function(out, xvar, title, model) {
  ggplot(out, aes(x = !!sym(xvar), group = 1)) + geom_point(aes(y = pred, colour = model)) + geom_point(aes(y = obs, colour = "observed")) + 
         geom_line(aes(y = pred, colour = model), linetype = "dashed") + geom_line(aes(y = obs, colour = "observed"), linetype = "dashed") +
         ylim(0, 0.35) + labs(x = xvar, y = "frequency", title = title) + theme(legend.position = "bottom")
}

plot_loss <- function(x) {
    if (length(x)>1) {
    df_val <- data.frame(epoch=1:length(x$loss),train_loss=x$loss,val_loss=x$val_loss)
    df_val <- gather(df_val, variable, loss, -epoch)
    p <- ggplot(df_val,aes(x=epoch,y=loss)) + geom_line() + facet_wrap(~variable,scales="free") + geom_smooth()
    suppressMessages(print(p))
    } else exit
}

load("freMTPL2freq.RData")

# Grouping id
distinct <- freMTPL2freq %>% 
  distinct_at(vars(-c(IDpol, Exposure, ClaimNb))) %>% 
  mutate(group_id = row_number())

dat <- freMTPL2freq %>% 
  left_join(distinct) %>% 
  mutate(ClaimNb = pmin(as.integer(ClaimNb), 4),
         VehAge = pmin(VehAge,20),
         DrivAge = pmin(DrivAge,90),
         BonusMalus = pmin(BonusMalus,150),
         Density = round(log(Density),2),
         VehGas = factor(VehGas),
         Exposure = pmin(Exposure, 1))

# Group sizes of suspected clusters
table(table(dat[, "group_id"]))

dat2 <- dat %>% mutate(
  AreaGLM = as.integer(Area),
  VehPowerGLM = as.factor(pmin(VehPower,9)),
  VehAgeGLM = cut(VehAge, breaks = c(-Inf, 0, 10, Inf), labels = c("1","2","3")),
  DrivAgeGLM = cut(DrivAge, breaks = c(-Inf, 20, 25, 30, 40, 50, 70, Inf), labels = c("1","2","3","4","5","6","7")),
  BonusMalusGLM = as.integer(pmin(BonusMalus, 150)),
  DensityGLM = as.numeric(Density),
  VehAgeGLM = relevel(VehAgeGLM, ref="2"),   
  DrivAgeGLM = relevel(DrivAgeGLM, ref="5"),
  Region = relevel(Region, ref="R24")    
)

head(dat2)

str(dat2)

summary(dat2)

ind <- partition(dat2[["group_id"]], p = c(train = 0.8, test = 0.2), 
                 seed = seed, type = "grouped")
train <- dat2[ind$train, ]
test <- dat2[ind$test, ]

# size of train/test
sprintf("Number of observations (train): %s", nrow(train))
sprintf("Number of observations (test): %s", nrow(test))

# Claims frequency of train/test
sprintf("Empirical frequency (train): %s", round(sum(train$ClaimNb) / sum(train$Exposure), 4))
sprintf("Empirical frequency (test): %s", round(sum(test$ClaimNb) / sum(test$Exposure), 4))

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

exec_time <- system.time(
  glm2 <- glm(ClaimNb ~ AreaGLM + VehPowerGLM + VehAgeGLM + BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region +
                        DrivAge + log(DrivAge) +  I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4),
              data = train, offset = log(Exposure), family = poisson())
)
exec_time[1:5]
summary(glm2)

drop1(glm2, test = "LRT")

# Predictions
train$fitGLM2 <- fitted(glm2)
test$fitGLM2 <- predict(glm2, newdata = test, type = "response")
dat$fitGLM2 <- predict(glm2, newdata = dat2, type = "response")

# in-sample and out-of-sample losses (in 10^(-2))
sprintf("100 x Poisson deviance GLM (train): %s", PoissonDeviance(train$fitGLM2, train$ClaimNb))
sprintf("100 x Poisson deviance GLM (test): %s", PoissonDeviance(test$fitGLM2, test$ClaimNb))

# Overall estimated frequency
sprintf("average frequency (test): %s", round(sum(test$fitGLM2) / sum(test$Exposure), 4))

df_cmp %<>% bind_rows(
  data.frame(model = "M1: GLM", epochs = NA, run_time = round(exec_time[[3]], 0), parameters = length(coef(glm2)),
             in_sample_loss = round(PoissonDeviance(train$fitGLM2, as.vector(unlist(train$ClaimNb))), 4),
             out_sample_loss = round(PoissonDeviance(test$fitGLM2, as.vector(unlist(test$ClaimNb))), 4),
             avg_freq = round(sum(test$fitGLM2) / sum(test$Exposure), 4))
)
df_cmp

# Area
out <- test %>% group_by(Area) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitGLM2) / sum(Exposure))
p1 <- plot_freq(out, "Area", "frequency by area", "GLM")
# VehPower
out <- test %>% group_by(VehPower) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitGLM2) / sum(Exposure))
p2 <- plot_freq(out, "VehPower", "frequency by vehicle power", "GLM")
# VehBrand
out <- test %>% group_by(VehBrand) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitGLM2) / sum(Exposure))
p3 <- plot_freq(out, "VehBrand", "frequency by vehicle brand", "GLM")
# VehAge
out <- test %>% group_by(VehAge) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitGLM2) / sum(Exposure))
p4 <- plot_freq(out, "VehAge", "frequency by vehicle age", "GLM")

grid.arrange(p1, p2, p3, p4)

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
      VehGasX = as.integer(VehGas) - 1.5,
      VehBrandX = as.integer(VehBrand) - 1,
      RegionX = as.integer(Region) - 1
    ) %>%
    preprocess_catdummy("VehBrand", "Br") %>%
    preprocess_catdummy("Region", "R")
}

dat2 <- preprocess_features(dat)

head(dat2)

str(dat2)

summary(dat2)

ind <- partition(dat2[["group_id"]], p = c(train = 0.8, test = 0.2), 
                 seed = seed, type = "grouped")
train <- dat2[ind$train, ]
test <- dat2[ind$test, ]

# size of train/test
sprintf("Number of observations (train): %s", nrow(train))
sprintf("Number of observations (test): %s", nrow(test))

# Claims frequency of train/test
sprintf("Empirical frequency (train): %s", round(sum(train$ClaimNb) / sum(train$Exposure), 4))
sprintf("Empirical frequency (test): %s", round(sum(test$ClaimNb) / sum(test$Exposure), 4))

# select the feature space

features <- c("AreaX","VehPowerX","VehAgeX","DrivAgeX","BonusMalusX","DensityX","VehGasX",
              "Br2","Br3","Br4","Br5","Br6","Br7","Br8","Br9","Br10","Br11",
              "R2","R3","R4","R5","R6","R7","R8","R9","R10","R11","R12","R13","R14","R15","R16","R17","R18","R19","R20","R21","R22")
print(colnames(train[, features]))

# feature matrix
Xtrain <- as.matrix(train[, features])  # design matrix training sample
Xtest <- as.matrix(test[, features])    # design matrix test sample

# available optimizers for keras
# https://keras.io/optimizers/
optimizers <- c('sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam', 'adamax', 'nadam')

# homogeneous model (train)
lambda_hom <- sum(train$ClaimNb) / sum(train$Exposure)

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

model_dp %>% compile(
  loss = 'poisson',
  optimizer = optimizers[7]
)

summary(model_dp)

# select number of epochs and batch.size
epochs <- 300
batch.size <- 10000
validation.split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1

# expected run-time on Renku 8GB environment around 200 seconds
exec_time <- system.time(
  fit <- model_dp %>% fit(
    list(Xtrain, as.matrix(log(train$Exposure))), as.matrix(train$ClaimNb),
    epochs = epochs,
    batch_size = batch.size,
    validation_split = validation.split,
    verbose = verbose
  )
)
exec_time[1:5]

plot(fit)

plot_loss(x=fit[[2]])

# Validation: Poisson deviance
train$fitdpNN <- as.vector(model_dp %>% predict(list(Xtrain, as.matrix(log(train$Exposure)))))
test$fitdpNN <- as.vector(model_dp %>% predict(list(Xtest, as.matrix(log(test$Exposure)))))

sprintf("100 x Poisson deviance shallow network (train): %s", PoissonDeviance(train$fitdpNN, train$ClaimNb))
sprintf("100 x Poisson deviance shallow network (test): %s", PoissonDeviance(test$fitdpNN, test$ClaimNb))

# average frequency
sprintf("Average frequency (test): %s", round(sum(test$fitdpNN) / sum(test$Exposure), 4))

trainable_params <- sum(unlist(lapply(model_dp$trainable_weights, k_count_params)))
df_cmp %<>% bind_rows(
  data.frame(model = "M2: Deep Plain Network", epochs = epochs,
             run_time = round(exec_time[[3]], 0), parameters = trainable_params,
             in_sample_loss = round(PoissonDeviance(train$fitdpNN, train$ClaimNb), 4),
             out_sample_loss = round(PoissonDeviance(test$fitdpNN, test$ClaimNb), 4),
             avg_freq = round(sum(test$fitdpNN) / sum(test$Exposure), 4)
  ))
df_cmp

# Area
out <- test %>% group_by(Area) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitdpNN) / sum(Exposure))
p1 <- plot_freq(out, "Area", "frequency by area", "dpNN")
# VehPower
out <- test %>% group_by(VehPower) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitdpNN) / sum(Exposure))
p2 <- plot_freq(out, "VehPower", "frequency by vehicle power", "dpNN")
# VehBrand
out <- test %>% group_by(VehBrand) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitdpNN) / sum(Exposure))
p3 <- plot_freq(out, "VehBrand", "frequency by vehicle brand", "dpNN")
# VehAge
out <- test %>% group_by(VehAge) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitdpNN) / sum(Exposure))
p4 <- plot_freq(out, "VehAge", "frequency by vehicle age", "dpNN")

grid.arrange(p1, p2, p3, p4)

# definition of non-embedded features
features <- c("AreaX", "VehPowerX", "VehAgeX", "DrivAgeX", "BonusMalusX", "VehGasX", "DensityX")
q0 <- length(features)   # number of non-embedded input features
q1 <- 20                 # number of neurons in first hidden layer
q2 <- 15                 # number of neurons in second hidden layer
q3 <- 10                 # number of neurons in second hidden layer

sprintf("Neural network with K=3 hidden layer")
sprintf("non-embedded feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", q1)
sprintf("Number of hidden neurons second layer: q2 = %s", q2)
sprintf("Number of hidden neurons third layer: q3 = %s", q3)
sprintf("Output dimension: %s", 1)

# training data
Xtrain <- as.matrix(train[, features])  # design matrix training sample
VehBrandtrain <- as.matrix(train$VehBrandX)
Regiontrain <- as.matrix(train$RegionX)
Ytrain <- as.matrix(train$ClaimNb)

# testing data
Xtest <- as.matrix(test[, features])    # design matrix test sample
VehBrandtest <- as.matrix(test$VehBrandX)
Regiontest <- as.matrix(test$RegionX)
Ytest <- as.matrix(test$ClaimNb)

# choosing the right volumes for EmbNN
Vtrain <- as.matrix(log(train$Exposure))
Vtest <- as.matrix(log(test$Exposure))
lambda_hom <- sum(train$ClaimNb) / sum(train$Exposure)

sprintf("Empirical frequency (train): %s", round(lambda_hom, 4))

# set the number of levels for the embedding variables
VehBrandLabel <- length(unique(train$VehBrandX))
RegionLabel <- length(unique(train$RegionX))

sprintf("Embedded VehBrand feature dimension: %s", VehBrandLabel)
sprintf("Embedded Region feature dimension: %s", RegionLabel)

# dimension embedding layers for categorical features
d <- 2 

# define the network architecture
Design   <- layer_input(shape = c(q0),  dtype = 'float32', name = 'Design')
VehBrand <- layer_input(shape = c(1),   dtype = 'int32', name = 'VehBrand')
Region   <- layer_input(shape = c(1),   dtype = 'int32', name = 'Region')
LogVol   <- layer_input(shape = c(1),   dtype = 'float32', name = 'LogVol')

BrandEmb <- VehBrand %>%
  layer_embedding(input_dim = VehBrandLabel, output_dim = d, input_length = 1, name = 'BrandEmb') %>%
  layer_flatten(name='Brand_flat')

RegionEmb <- Region %>%
  layer_embedding(input_dim = RegionLabel, output_dim = d, input_length = 1, name = 'RegionEmb') %>%
  layer_flatten(name='Region_flat')

Network <- list(Design, BrandEmb, RegionEmb) %>% layer_concatenate(name = 'concate') %>%
  layer_dense(units = q1, activation = 'tanh', name = 'hidden1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'hidden2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'hidden3') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q3,1)), array(log(lambda_hom), dim = 1)))

Response <- list(Network, LogVol) %>% layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1,1)), array(0, dim = 1)))

model_nn <- keras_model(inputs = c(Design, VehBrand, Region, LogVol), outputs = c(Response))

model_nn %>% compile(
  loss = 'poisson',
  optimizer = optimizers[7]
)

summary(model_nn)

# select number of epochs and batch.size
epochs <- 500
batch_size <- 10000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1

# fitting the neural network
# expected run-time on Renku 8GB environment around 50 seconds
exec_time <- system.time(
  fit <- model_nn %>% fit(
      list(Xtrain, VehBrandtrain, Regiontrain, Vtrain), Ytrain,
      epochs = epochs,
      batch_size = batch_size,
      verbose = verbose,
      validation_split = validation_split
  )
)
exec_time[1:5]

plot(fit)

plot_loss(x=fit[[2]])

# calculating the predictions
train$fitNN <- as.vector(model_nn %>% predict(list(Xtrain, VehBrandtrain, Regiontrain, Vtrain)))
test$fitNN <- as.vector(model_nn %>% predict(list(Xtest, VehBrandtest, Regiontest, Vtest)))

# average in-sample and out-of-sample losses (in 10^(-2))
sprintf("100 x Poisson deviance shallow network (train): %s", PoissonDeviance(train$fitNN, as.vector(unlist(train$ClaimNb))))
sprintf("100 x Poisson deviance shallow network (test): %s", PoissonDeviance(test$fitNN, as.vector(unlist(test$ClaimNb))))

# average frequency
sprintf("Average frequency (test): %s", round(sum(test$fitNN) / sum(test$Exposure), 4))

# extract the number of trainable parameters from the keras model
tot_params <- count_params(model_nn)
trainable_params <- sum(unlist(lapply(model_nn$trainable_weights, k_count_params)))
nontrainable_params <- sum(unlist(lapply(model_nn$non_trainable_weights, k_count_params)))

df_cmp %<>% bind_rows(
  data.frame(model = "M3: EmbNN", epochs = epochs,
             run_time = round(exec_time[[3]], 0), parameters = trainable_params,
             in_sample_loss = round(PoissonDeviance(train$fitNN, as.vector(unlist(train$ClaimNb))), 4),
             out_sample_loss = round(PoissonDeviance(test$fitNN, as.vector(unlist(test$ClaimNb))), 4),
             avg_freq = round(sum(test$fitNN) / sum(test$Exposure), 4)
  ))
df_cmp

# VehBrand
p1 <- ggplot(dat, aes(VehBrand)) +
  geom_bar(aes(weight = Exposure)) +
  labs(x = "Vehicle Brand", y = "exposure", title = "exposure by vehicle brand") +
  theme(axis.text.x = element_text(size = 5))

out <- dat %>% group_by(VehBrand) %>% summarize(freq = mean(ClaimNb) / mean(Exposure))
p2 <- ggplot(out, aes(x = VehBrand, y = freq, group = 1)) +
  geom_point() + geom_line(linetype = "dashed") +
  ylim(0, 0.35) + labs(x = "VehBrand", y = "frequency", title = "frequency by vehicle brand") +
  theme(axis.text.x = element_text(size = 5))

# Region
p3 <- ggplot(dat, aes(Region)) +
  geom_bar(aes(weight = Exposure)) +
  labs(x = "Region", y = "exposure", title = "exposure by region") +
  theme(axis.text.x = element_text(angle = 90, size = 5))

out <- dat %>% group_by(Region) %>% summarize(freq = mean(ClaimNb) / mean(Exposure))
p4 <- ggplot(out, aes(x = Region, y = freq, group = 1)) +
  geom_point() + geom_line(linetype = "dashed") +
  ylim(0, 0.35) + labs(x = "Region", y = "frequency", title = "frequency by region") +
  theme(axis.text.x = element_text(angle = 90, size = 5))

grid.arrange(p1, p2, p3, p4)

## VehBrand
emb_Brand <- (model_nn$get_layer("BrandEmb") %>% get_weights())[[1]] %>%
  as.data.frame() %>%
  add_column(levels(factor(dat2$VehBrand))) %>%
  rlang::set_names(c("Dim_1", "Dim_2", "Label"))

ggplot(emb_Brand, aes(x = Dim_1, y = Dim_2, label = Label)) + 
  geom_point(size = 1) + 
  geom_text(aes(label = Label), hjust = 0, vjust = 0, size = 3) + 
  ggtitle("2-dim embedding: VehBrand")

## Region
emb_Region <- (model_nn$get_layer("RegionEmb") %>% get_weights())[[1]] %>%
  as.data.frame() %>%
  add_column(levels(factor(dat2$Region))) %>%
  rlang::set_names(c("Dim_1", "Dim_2", "Label"))

ggplot(emb_Region, aes(x = Dim_1, y = Dim_2, label = Label)) + 
  geom_point(size = 1) + 
  geom_text(aes(label = Label), hjust = 0, vjust = 0, size = 3) + 
  ggtitle("2-dim embedding: Region")


# Area
out <- test %>% group_by(Area) %>% summarize(obs = sum(ClaimNb) / sum(Exposure),
                                             pred = sum(fitNN) / sum(Exposure))
p1 <- plot_freq(out, "Area", "frequency by area", "EmbNN")
# VehPower
out <- test %>% group_by(VehPower) %>% summarize(obs = sum(ClaimNb) / sum(Exposure),
                                                 pred = sum(fitNN) / sum(Exposure))
p2 <- plot_freq(out, "VehPower", "frequency by vehicle power", "EmbNN")
# VehBrand
out <- test %>% group_by(VehBrand) %>% summarize(obs = sum(ClaimNb) / sum(Exposure),
                                                 pred = sum(fitNN) / sum(Exposure))
p3 <- plot_freq(out, "VehBrand", "frequency by vehicle brand", "EmbNN")
# VehAge
out <- test %>% group_by(VehAge) %>% summarize(obs = sum(ClaimNb) / sum(Exposure),
                                               pred = sum(fitNN) / sum(Exposure))
p4 <- plot_freq(out, "VehAge", "frequency by vehicle age", "EmbNN")

grid.arrange(p1, p2, p3, p4)

# definition of non-embedded features
features <- c("AreaX", "VehPowerX", "VehAgeX", "DrivAgeX", "BonusMalusX", "VehGasX", "DensityX")
q0 <- length(features)   # number of non-embedded input features
q1 <- 20                 # number of neurons in first hidden layer
q2 <- 15                 # number of neurons in second hidden layer
q3 <- 10                 # number of neurons in second hidden layer

sprintf("Neural network with K=3 hidden layer")
sprintf("non-embedded feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", q1)
sprintf("Number of hidden neurons second layer: q2 = %s", q2)
sprintf("Number of hidden neurons third layer: q3 = %s", q3)
sprintf("Output dimension: %s", 1)

# training data
Xtrain <- as.matrix(train[, features])  # design matrix training sample
VehBrandtrain <- as.matrix(train$VehBrandX)
Regiontrain <- as.matrix(train$RegionX)
Ytrain <- as.matrix(train$ClaimNb)

# testing data
Xtest <- as.matrix(test[, features])    # design matrix test sample
VehBrandtest <- as.matrix(test$VehBrandX)
Regiontest <- as.matrix(test$RegionX)
Ytest <- as.matrix(test$ClaimNb)

# choosing the right volumes for CANN (difference to NN above!)
Vtrain <- as.matrix(log(train$fitGLM2))
Vtest <- as.matrix(log(test$fitGLM2))
lambda_hom <- sum(train$ClaimNb) / sum(train$fitGLM2)

sprintf("Empirical frequency (train): %s", round(lambda_hom, 4))

# set the number of levels for the embedding variables
VehBrandLabel <- length(unique(train$VehBrandX))
RegionLabel <- length(unique(train$RegionX))

sprintf("Embedded VehBrand feature dimension: %s", VehBrandLabel)
sprintf("Embedded Region feature dimension: %s", RegionLabel)

# dimensions embedding layers for categorical features
d <- 2        

# define the network architecture
Design   <- layer_input(shape = c(q0),  dtype = 'float32', name = 'Design')
VehBrand <- layer_input(shape = c(1),   dtype = 'int32', name = 'VehBrand')
Region   <- layer_input(shape = c(1),   dtype = 'int32', name = 'Region')
LogVol   <- layer_input(shape = c(1),   dtype = 'float32', name = 'LogVol')

BrandEmb <- VehBrand %>%
  layer_embedding(input_dim = VehBrandLabel, output_dim = d, input_length = 1, name = 'BrandEmb') %>%
  layer_flatten(name = 'Brand_flat')

RegionEmb <- Region %>%
  layer_embedding(input_dim = RegionLabel, output_dim = d, input_length = 1, name = 'RegionEmb') %>%
  layer_flatten(name = 'Region_flat')

Network <- list(Design, BrandEmb, RegionEmb) %>% layer_concatenate(name = 'concate') %>%
  layer_dense(units = q1, activation = 'tanh', name = 'hidden1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'hidden2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'hidden3') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q3, 1)), array(log(lambda_hom), dim = 1)))

Response <- list(Network, LogVol) %>% layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = 1)))

model_cann <- keras_model(inputs = c(Design, VehBrand, Region, LogVol), outputs = c(Response))


model_cann %>% compile(
  loss = 'poisson',
  optimizer = optimizers[7]
)

summary(model_cann)

# select number of epochs and batch.size
epochs <- 500
batch_size <- 10000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1

# fitting the neural network
# expected run-time on Renku 8GB environment around 50 seconds
exec_time <- system.time(
  fit <- model_cann %>% fit(list(Xtrain, VehBrandtrain, Regiontrain, Vtrain), Ytrain,
                       epochs = epochs, 
                       batch_size = batch_size,
                       verbose = verbose,
                       validation_split = validation_split)
)
exec_time[1:5]

plot(fit)

plot_loss(x=fit[[2]])

# calculating the predictions
train$fitCANN <- as.vector(model_cann %>% predict(list(Xtrain, VehBrandtrain, Regiontrain, Vtrain)))
test$fitCANN <- as.vector(model_cann %>% predict(list(Xtest, VehBrandtest, Regiontest, Vtest)))

# average in-sample and out-of-sample losses (in 10^(-2))
sprintf("100 x Poisson deviance shallow network (train): %s", PoissonDeviance(train$fitCANN, as.vector(unlist(train$ClaimNb))))
sprintf("100 x Poisson deviance shallow network (test): %s", PoissonDeviance(test$fitCANN, as.vector(unlist(test$ClaimNb))))

# average frequency
sprintf("Average frequency (test): %s", round(sum(test$fitCANN) / sum(test$Exposure), 4))

trainable_params <- sum(unlist(lapply(model_cann$trainable_weights, k_count_params)))
df_cmp %<>% bind_rows(
  data.frame(model = "M4: EmbCANN", epochs = epochs,
             run_time = round(exec_time[[3]], 0), parameters = trainable_params,
             in_sample_loss = round(PoissonDeviance(train$fitCANN, as.vector(unlist(train$ClaimNb))), 4),
             out_sample_loss = round(PoissonDeviance(test$fitCANN, as.vector(unlist(test$ClaimNb))), 4),
             avg_freq = round(sum(test$fitCANN)/sum(test$Exposure), 4)
  ))
df_cmp

## VehBrand
emb_Brand <- (model_cann$get_layer("BrandEmb") %>% get_weights())[[1]] %>%
  as.data.frame() %>%
  add_column(levels(factor(dat2$VehBrand))) %>%
  rlang::set_names(c("Dim_1", "Dim_2", "Label"))

ggplot(emb_Brand, aes(x = Dim_1, y = Dim_2, label = Label)) +
  geom_point(size = 1) +
  geom_text(aes(label = Label), hjust = 0, vjust = 0, size = 3) +
  ggtitle("2-dim embedding: VehBrand")

## Region
emb_Region <- (model_cann$get_layer("RegionEmb") %>% get_weights())[[1]] %>%
  as.data.frame() %>%
  add_column(levels(factor(dat2$Region))) %>%
  rlang::set_names(c("Dim_1", "Dim_2", "Label"))

ggplot(emb_Region, aes(x = Dim_1, y = Dim_2, label = Label)) +
  geom_point(size = 1) +
  geom_text(aes(label = Label), hjust = 0, vjust = 0, size = 3) +
  ggtitle("2-dim embedding: Region")


# Area
out <- test %>% group_by(Area) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitCANN) / sum(Exposure))
p1 <- plot_freq(out, "Area", "Frequency by area", "CANN")
# VehPower
out <- test %>% group_by(VehPower) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitCANN) / sum(Exposure))
p2 <- plot_freq(out, "VehPower", "Frequency by vehicle power", "CANN")
# VehBrand
out <- test %>% group_by(VehBrand) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitCANN) / sum(Exposure))
p3 <- plot_freq(out, "VehBrand", "Frequency by vehicle brand", "CANN")
# VehAge
out <- test %>% group_by(VehAge) %>%
  summarize(obs = sum(ClaimNb) / sum(Exposure), pred = sum(fitCANN) / sum(Exposure))
p4 <- plot_freq(out, "VehAge", "Frequency by vehicle age", "CANN")

grid.arrange(p1, p2, p3, p4)

exec_time <- system.time({
  datGAM <- train %>% group_by(VehAge, BonusMalus) %>% summarize(fitGLM2 = sum(fitGLM2), ClaimNb = sum(ClaimNb))
  gam1 <- gam(ClaimNb ~ s(VehAge, bs="cr") + s(BonusMalus, bs="cr"), data = datGAM, method = "GCV.Cp", offset = log(fitGLM2), family = poisson)
})
exec_time[1:5]

summary(gam1)

# Predictions
train$fitGAM1 <- exp(predict(gam1, newdata = train)) * train$fitGLM2
test$fitGAM1 <- exp(predict(gam1, newdata = test)) * test$fitGLM2
dat$fitGAM1 <- exp(predict(gam1, newdata = dat)) * dat$fitGLM2

# in-sample and out-of-sample losses (in 10^(-2))
sprintf("100 x Poisson deviance GLM (train): %s", PoissonDeviance(train$fitGAM1, train$ClaimNb))
sprintf("100 x Poisson deviance GLM (test): %s", PoissonDeviance(test$fitGAM1, test$ClaimNb))

# Overall estimated frequency
sprintf("average frequency (test): %s", round(sum(test$fitGAM1) / sum(test$Exposure), 4))


df_cmp %<>% bind_rows(
  data.frame(model = "M5: GAM1", epochs = NA,
             run_time = round(exec_time[[3]], 0), parameters = length(coef(glm2)) + round(sum(pen.edf(gam1)), 1),
             in_sample_loss = round(PoissonDeviance(train$fitGAM1, as.vector(unlist(train$ClaimNb))), 4),
             out_sample_loss = round(PoissonDeviance(test$fitGAM1, as.vector(unlist(test$ClaimNb))), 4),
             avg_freq = round(sum(test$fitGAM1) / sum(test$Exposure), 4)
  ))
df_cmp

# neural network definitions for model (3.11)
trainX <- list(as.matrix(train[, c("VehPowerX", "VehAgeX", "VehGasX")]),
                as.matrix(train[, "VehBrandX"]),
                as.matrix(train[, c("DrivAgeX", "BonusMalus")]),
                as.matrix(log(train$fitGAM1)) )

testX <- list(as.matrix(test[, c("VehPowerX", "VehAgeX", "VehGasX")]),
                as.matrix(test[, "VehBrandX"]),
                as.matrix(test[, c("DrivAgeX", "BonusMalus")]),
                as.matrix(log(test$fitGAM1)) )

neurons <- c(20, 15, 10)
no_labels <- length(unique(train$VehBrandX))

#  definition of neural network (3.11)

Model2IA <- function(no_labels) {
  Cont1 <- layer_input(shape = c(3), dtype = 'float32', name = 'Cont1')
  Cat1 <- layer_input(shape = c(1), dtype = 'int32', name = 'Cat1')
  Cont2 <- layer_input(shape = c(2), dtype = 'float32', name = 'Cont2')
  LogExposure <- layer_input(shape = c(1), dtype = 'float32', name = 'LogExposure')
  x_input <- c(Cont1, Cat1, Cont2, LogExposure)
  
  Cat1_embed <- Cat1 %>%
    layer_embedding(input_dim = no_labels, output_dim = 2, trainable = TRUE, input_length = 1, name = 'Cat1_embed') %>%
    layer_flatten(name = 'Cat1_flat')
  
  NNetwork1 <- list(Cont1, Cat1_embed) %>% layer_concatenate(name = 'cont') %>%
    layer_dense(units = neurons[1], activation = 'tanh', name = 'hidden1') %>%
    layer_dense(units = neurons[2], activation = 'tanh', name = 'hidden2') %>%
    layer_dense(units = neurons[3], activation = 'tanh', name = 'hidden3') %>%
    layer_dense(units = 1, activation = 'linear', name = 'NNetwork1',
                weights = list(array(0, dim = c(neurons[3], 1)), array(0, dim = 1)))
  
  NNetwork2 <- Cont2 %>%
    layer_dense(units = neurons[1], activation = 'tanh', name = 'hidden4') %>%
    layer_dense(units = neurons[2], activation = 'tanh', name = 'hidden5') %>%
    layer_dense(units = neurons[3], activation = 'tanh', name = 'hidden6') %>%
    layer_dense(units = 1, activation = 'linear', name = 'NNetwork2',
                weights = list(array(0, dim = c(neurons[3], 1)), array(0, dim = 1)))
  
  NNoutput <- list(NNetwork1, NNetwork2, LogExposure) %>% layer_add(name = 'Add') %>%
    layer_dense(units = 1, activation = k_exp, name = 'NNoutput', trainable = FALSE,
                weights = list(array(c(1), dim = c(1, 1)), array(0, dim = 1)))
  
  model <- keras_model(inputs = x_input, outputs = c(NNoutput))
  
  model %>% compile(optimizer = optimizer_nadam(),
                    loss = 'poisson')
  
  model
}

model_gamplus <- Model2IA(no_labels)

summary(model_gamplus)


# select number of epochs and batch_size
epochs <- 100
batch_size <- 10000
verbose <- 1

# expected run-time on Renku 8GB environment around 65 seconds
# may take a couple of minutes if epochs is more than 100
exec_time <- system.time(
 fit <- model_gamplus %>% fit(trainX, as.matrix(train$ClaimNb),
                          epochs = epochs,
                          batch_size = batch_size,
                          verbose = verbose, 
                          validation_data = list(testX, as.matrix(test$ClaimNb)))
)
exec_time[1:5]


plot(fit)

plot_loss(x=fit[[2]])

# calculating the predictions
train$fitGAMPlus <- as.vector(model_gamplus %>% predict(trainX))
test$fitGAMPlus <- as.vector(model_gamplus %>% predict(testX))

# average in-sample and out-of-sample losses (in 10^(-2))
sprintf("100 x Poisson deviance shallow network (train): %s", PoissonDeviance(train$fitGAMPlus, as.vector(unlist(train$ClaimNb))))
sprintf("100 x Poisson deviance shallow network (test): %s", PoissonDeviance(test$fitGAMPlus, as.vector(unlist(test$ClaimNb))))

# average frequency
sprintf("Average frequency (test): %s", round(sum(test$fitGAMPlus) / sum(test$Exposure), 4))

trainable_params <- sum(unlist(lapply(model_gamplus$trainable_weights, k_count_params)))
df_cmp %<>% bind_rows(
  data.frame(model = "M6: GAM+", epochs = epochs,
             run_time = round(exec_time[[3]], 0), parameters = trainable_params,
             in_sample_loss = round(PoissonDeviance(train$fitGAMPlus, as.vector(unlist(train$ClaimNb))), 4),
             out_sample_loss = round(PoissonDeviance(test$fitGAMPlus, as.vector(unlist(test$ClaimNb))), 4),
             avg_freq = round(sum(test$fitGAMPlus) / sum(test$Exposure), 4)
  ))
df_cmp

plot_freq_multi <- function(xvar, title) {
  out <- test %>% group_by(!!sym(xvar)) %>% summarize(
    obs = sum(ClaimNb) / sum(Exposure),
    glm = sum(fitGLM2) / sum(Exposure),
    gam = sum(fitGAM1) / sum(Exposure),
    gamp = sum(fitGAMPlus) / sum(Exposure)
  )
  ggplot(out, aes(x = !!sym(xvar), group = 1)) +
    geom_point(aes(y = obs, colour = "observed")) + geom_line(aes(y = obs, colour = "observed"), linetype = "dashed") +
    geom_point(aes(y = glm, colour = "GLM")) + geom_line(aes(y = glm, colour = "GLM"), linetype = "dashed") +
    geom_point(aes(y = gam, colour = "GAM1")) + geom_line(aes(y = gam, colour = "GAM1"), linetype = "dashed") +
    geom_point(aes(y = gamp, colour = "GAM+")) + geom_line(aes(y = gamp, colour = "GAM+"), linetype = "dashed") +
    ylim(0, 0.35) + labs(x = xvar, y = "frequency", title = title) + theme(legend.position = "bottom")
}
# Area
p1 <- plot_freq_multi("Area", "frequency by area")
# VehPower
p2 <- plot_freq_multi("VehPower", "frequency by VehPower")
# VehBrand
p3 <- plot_freq_multi("VehBrand", "frequency by VehBrand")
# VehAge
p4 <- plot_freq_multi("VehAge", "frequency by VehAge")

grid.arrange(p1, p2, p3, p4)


df_cmp

plot_cmp <- function(xvar, title, maxlim = 0.35) {
  out <- test %>% group_by(!!sym(xvar)) %>% summarize(
    vol = sum(Exposure),
    obs = sum(ClaimNb) / sum(Exposure),
    glm = sum(fitGLM2) / sum(Exposure),
    nn = sum(fitNN) / sum(Exposure),
    cann = sum(fitCANN) / sum(Exposure),
    gam = sum(fitGAM1) / sum(Exposure),
    gamplus = sum(fitGAMPlus) / sum(Exposure)
  )
  
  max_pri <- max(out$obs, out$glm, out$nn, out$cann, out$gam, out$gamplus)
  max_sec <- max(out$vol)
  max_ratio <- max_pri / max_sec
  
  if (is.null(maxlim)) maxlim <- max_pri
  
  ggplot(out, aes(x = !!sym(xvar), group = 1)) +
    geom_point(aes(y = obs, colour = "observed")) + geom_line(aes(y = obs, colour = "observed"), linetype = "dashed") +
    geom_point(aes(y = glm, colour = "GLM")) + geom_line(aes(y = glm, colour = "GLM"), linetype = "dashed") +
    geom_point(aes(y = nn, colour = "EmbNN")) + geom_line(aes(y = nn, colour = "EmbNN"), linetype = "dashed") +
    geom_point(aes(y = cann, colour = "CANN")) + geom_line(aes(y = cann, colour = "CANN"), linetype = "dashed") +
    geom_point(aes(y = gam, colour = "GAM")) + geom_line(aes(y = gam, colour = "GAM"), linetype = "dashed") +
    geom_point(aes(y = gamplus, colour = "GAM+")) + geom_line(aes(y = gamplus, colour = "GAM+"), linetype = "dashed") +
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
p5 <- plot_cmp("DrivAge", "Frequency and Exposure by DrivAge", maxlim = NULL)

grid.arrange(p1, p2, p3, p4, p5)


axis_min <- log(max(test$fitCANN, test$fitNN))
axis_max <- log(min(test$fitCANN, test$fitNN))

ggplot(test, aes(x = log(fitNN), y = log(fitCANN), colour = Exposure)) + geom_point() +
  geom_abline(colour = "#000000", slope = 1, intercept = 0) +
  xlim(axis_max, axis_min) + ylim(axis_max, axis_min) +
  labs(x = " EmbNN", y = "EmbCANN", title = "Claims frequency prediction (log-scale)") +
  scale_colour_gradient(low = "green", high = "red")


axis_min <- log(max(test$fitCANN, test$fitGLM2))
axis_max <- log(min(test$fitCANN, test$fitGLM2))

ggplot(test, aes(x = log(fitGLM2), y = log(fitCANN), colour = Exposure)) + geom_point() +
  geom_abline(colour = "#000000", slope = 1, intercept = 0) +
  xlim(axis_max, axis_min) + ylim(axis_max, axis_min) +
  labs(x = "GLM", y = "EmbCANN", title = "Claims frequency prediction (log-scale)") +
  scale_colour_gradient(low = "green", high = "red")


sessionInfo()

reticulate::py_config()

tensorflow::tf_version()
