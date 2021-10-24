## -----------------------------------------------------------------------------
library(keras)
library(locfit)
library(magrittr)
library(dplyr)
library(tibble)
library(purrr)
library(ggplot2)
library(gridExtra)
library(tidyr)
library(corrplot)
RNGversion("3.5.0")


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
# https://stackoverflow.com/questions/65366442/cannot-convert-a-symbolic-keras-input-output-to-a-numpy-array-typeerror-when-usi
# https://tensorflow.rstudio.com/guide/tfhub/examples/feature_column/
tensorflow::tf$compat$v1$disable_eager_execution()


## -----------------------------------------------------------------------------
ax_limit <- c(0,50000)
line_size <- 1.1


## -----------------------------------------------------------------------------
# MinMax scaler
preprocess_minmax <- function(varData) {
  X <- as.numeric(varData)
  2 * (X - min(X)) / (max(X) - min(X)) - 1
}


## -----------------------------------------------------------------------------
# One Hot encoding for categorical features
preprocess_cat_onehot <- function(data, varName, prefix) {
  varData <- data[[varName]]
  X <- as.integer(varData)
  n0 <- length(unique(X))
  n1 <- 1:n0
  addCols <- purrr::map(n1, function(x, y) {as.integer(y == x)}, y = X) %>%
    rlang::set_names(paste0(prefix, n1))
  cbind(data, addCols)
}


## -----------------------------------------------------------------------------
#https://stat.ethz.ch/pipermail/r-help/2013-July/356936.html
scale_no_attr <- function (x, center = TRUE, scale = TRUE) 
{
    x <- as.matrix(x)
    nc <- ncol(x)
    if (is.logical(center)) {
        if (center) {
            center <- colMeans(x, na.rm = TRUE)
            x <- sweep(x, 2L, center, check.margin = FALSE)
        }
    }
    else if (is.numeric(center) && (length(center) == nc)) 
        x <- sweep(x, 2L, center, check.margin = FALSE)
    else stop("length of 'center' must equal the number of columns of 'x'")
    if (is.logical(scale)) {
        if (scale) {
            f <- function(v) {
                v <- v[!is.na(v)]
                sqrt(sum(v^2)/max(1, length(v) - 1L))
            }
            scale <- apply(x, 2L, f)
            x <- sweep(x, 2L, scale, "/", check.margin = FALSE)
        }
    }
    else if (is.numeric(scale) && length(scale) == nc) 
        x <- sweep(x, 2L, scale, "/", check.margin = FALSE)
    else stop("length of 'scale' must equal the number of columns of 'x'")
    #if (is.numeric(center)) 
    #    attr(x, "scaled:center") <- center
    #if (is.numeric(scale)) 
    #    attr(x, "scaled:scale") <- scale
    x
}


## -----------------------------------------------------------------------------
square_loss <- function(y_true, y_pred){mean((y_true-y_pred)^2)}
gamma_loss  <- function(y_true, y_pred){2*mean((y_true-y_pred)/y_pred + log(y_pred/y_true))}
ig_loss     <- function(y_true, y_pred){mean((y_true-y_pred)^2/(y_pred^2*y_true))}
p_loss      <- function(y_true, y_pred, p){2*mean(y_true^(2-p)/((1-p)*(2-p))-y_true*y_pred^(1-p)/(1-p)+y_pred^(2-p)/(2-p))}

k_gamma_loss  <- function(y_true, y_pred){2*k_mean(y_true/y_pred - 1 - log(y_true/y_pred))}
k_ig_loss     <- function(y_true, y_pred){k_mean((y_true-y_pred)^2/(y_pred^2*y_true))}
k_p_loss      <- function(y_true, y_pred){2*k_mean(y_true^(2-p)/((1-p)*(2-p))-y_true*y_pred^(1-p)/(1-p)+y_pred^(2-p)/(2-p))}


## -----------------------------------------------------------------------------
keras_plot_loss_min <- function(x, seed) {
    x <- x[[2]]
    ylim <- range(x)
    vmin <- which.min(x$val_loss)
    df_val <- data.frame(epoch = 1:length(x$loss), train_loss = x$loss, val_loss = x$val_loss)
    df_val <- gather(df_val, variable, loss, -epoch)
    plt <- ggplot(df_val, aes(x = epoch, y = loss, group = variable, color = variable)) +
      geom_line(size = line_size) + geom_vline(xintercept = vmin, color = "green", size = line_size) +
      labs(title = paste("Train and validation loss for seed", seed),
           subtitle = paste("Green line: Smallest validation loss for epoch", vmin))
    suppressMessages(print(plt))
}


## -----------------------------------------------------------------------------
plot_size <- function(test, xvar, title, model, mdlvariant) {
  out <- test %>% group_by(!!sym(xvar)) %>%
    summarize(obs = mean(Claim) , pred = mean(!!sym(mdlvariant)))
  
  ggplot(out, aes(x = !!sym(xvar), group = 1)) +
    geom_point(aes(y = pred, colour = model)) +
    geom_point(aes(y = obs, colour = "observed")) +
    geom_line(aes(y = pred, colour = model), linetype = "dashed") +
    geom_line(aes(y = obs, colour = "observed"), linetype = "dashed") +
    ylim(ax_limit) + labs(x = xvar, y = "claim size", title = title) +
    theme(legend.position = "bottom")
}


## -----------------------------------------------------------------------------
load(file.path("../0_data/WorkersComp.RData"))  # relative path to .Rmd file


## -----------------------------------------------------------------------------
dat <- WorkersComp %>% filter(AccYear > 1987, HoursWorkedPerWeek > 0)


## -----------------------------------------------------------------------------
# Order claims in decreasing order for split train/test (see below), and add an ID
dat <- dat %>% arrange(desc(Claim))
dat <- dat %>% mutate(Id=1:nrow(dat))


## -----------------------------------------------------------------------------
# scaling and cut-off
dat <- dat %>% mutate(
        Age = pmax(16, pmin(70, Age)),
        AgeNN = scale_no_attr(Age),
        GenderNN = as.integer(Gender),
        GenderNN = scale_no_attr(GenderNN),
        DependentChildren = pmin(1, DependentChildren),
        DependentChildrenNN = scale_no_attr(DependentChildren),
        DependentsOther = pmin(1, DependentsOther),
        DependentsOtherNN = scale_no_attr(DependentsOther),
        WeeklyPay = pmin(1200, WeeklyPay),
        WeeklyPayNN = scale_no_attr(WeeklyPay),
        PartTimeFullTimeNN = scale_no_attr(as.integer(PartTimeFullTime)),
        HoursWorkedPerWeek = pmin(60, HoursWorkedPerWeek),
        HoursWorkedPerWeekNN = scale_no_attr(HoursWorkedPerWeek),
        DaysWorkedPerWeekNN = scale_no_attr(DaysWorkedPerWeek),
        AccYearNN = scale_no_attr(AccYear),
        AccMonthNN = scale_no_attr(AccMonth),
        AccWeekdayNN = scale_no_attr(AccWeekday),
        AccTimeNN = scale_no_attr(AccTime),
        RepDelay = pmin(100, RepDelay),
        RepDelayNN = scale_no_attr(RepDelay)
)


## -----------------------------------------------------------------------------
# one-hot encoding (not dummy encoding!)
dat <- dat %>% preprocess_cat_onehot("MaritalStatus", "Marital")


## -----------------------------------------------------------------------------
# add two additional randomly generated features (later used)
set.seed(seed)

dat <- dat %>% mutate(
    RandNN = rnorm(nrow(dat)),
    RandNN = scale_no_attr(RandNN),
    RandUN = runif(nrow(dat), min = -sqrt(3), max = sqrt(3)),
    RandUN = scale_no_attr(RandUN)
)


## -----------------------------------------------------------------------------
head(dat)


## -----------------------------------------------------------------------------
str(dat)


## -----------------------------------------------------------------------------
summary(dat)


## -----------------------------------------------------------------------------
idx <- sample(x = c(1:5), size = ceiling(nrow(dat) / 5), replace = TRUE)
idx <- (1:ceiling(nrow(dat) / 5) - 1) * 5 + idx

test <- dat[intersect(idx, 1:nrow(dat)), ]
learn <- dat[setdiff(1:nrow(dat), idx), ]

learn <- learn[sample(1:nrow(learn)), ]
test <- test[sample(1:nrow(test)), ]


## -----------------------------------------------------------------------------
# size of train/test
sprintf("Number of observations (learn): %s", nrow(learn))
sprintf("Number of observations (test): %s", nrow(test))


## -----------------------------------------------------------------------------
# Claims average of learn/test
sprintf("Empirical claims average (learn): %s", round(sum(learn$Claim) / length(learn$Claim), 0))
sprintf("Empirical claims average (test): %s", round(sum(test$Claim) / length(test$Claim), 0))


## -----------------------------------------------------------------------------
# Quantiles of learn/test
probs <- c(.1, .25, .5, .75, .9)
bind_rows(quantile(learn$Claim, probs = probs), quantile(test$Claim, probs = probs))


## -----------------------------------------------------------------------------
# initialize table to store all model results for comparison
df_cmp <- tibble(
 model = character(),
 learn_p2 = numeric(),
 learn_pp = numeric(),
 learn_p3 = numeric(),
 test_p2 = numeric(),
 test_pp = numeric(),
 test_p3 = numeric(),
 avg_size = numeric(),
)


## -----------------------------------------------------------------------------
range(dat$AccYear)


## -----------------------------------------------------------------------------
range(dat$RepYear)


## -----------------------------------------------------------------------------
range(dat$RepDelay)


## -----------------------------------------------------------------------------
round(range(dat$RepDelay) / 365, 4)


## -----------------------------------------------------------------------------
round(range(dat$RepDelay) / 7, 4)


## -----------------------------------------------------------------------------
round(mean(dat$RepDelay) / 7, 4)


## -----------------------------------------------------------------------------
# define acc_week, rep_week and RepDelay_week
min_accDay <- min(dat$AccDay)
dat <- dat %>% mutate(
    acc_week = floor((AccDay - min_accDay) / 7),
    rep_week = floor((RepDay - min_accDay) / 7),
    RepDelay_week = rep_week - acc_week
)


## -----------------------------------------------------------------------------
quantile(dat$RepDelay_week, probs = c(.9, .98, .99))


## -----------------------------------------------------------------------------
acc1 <- dat %>% group_by(acc_week, rep_week) %>% summarize(nr = n())
acc2 <- dat %>% group_by(acc_week) %>% summarize(mm = mean(RepDelay_week))


## -----------------------------------------------------------------------------
head(acc1)


## -----------------------------------------------------------------------------
head(acc2)


## -----------------------------------------------------------------------------
# to plot the quantiles
qq0 <- c(.9, .975, .99)
qq1 <- quantile(dat$RepDelay_week, probs = qq0)
qq1


## -----------------------------------------------------------------------------
ggplot(acc1, aes(x = rep_week - acc_week, y = max(acc_week) - acc_week)) +
    geom_point() +
    geom_point(data = acc2, aes(x = mm, y = acc_week), color = "cyan") +
    geom_vline(xintercept = qq1, color = "orange", size = line_size) +
    scale_y_continuous(
      labels = rev(c(min(dat$AccYear):(max(dat$AccYear) + 1))),
      breaks = c(0:(max(dat$AccYear) - min(dat$AccYear) + 1)) * 365 / 7 - 25
    ) + labs(title = "Claims reporting", x = "reporting delay (in weeks)", y = "accident date")


## -----------------------------------------------------------------------------
range(dat$Claim)


## -----------------------------------------------------------------------------
summary(dat$Claim)


## -----------------------------------------------------------------------------
p1 <- ggplot(dat %>% filter(Claim <= 10000), aes(x = Claim)) + geom_density(colour = "blue") +
    labs(title = "Empirical density of claims amounts", x = "claims amounts (<=10000)", y = "empirical density")

p2 <- ggplot(dat, aes(x = log(Claim))) + geom_density(colour = "blue") +
    labs(title = "Empirical density of log(claims amounts)", x = "claims amounts", y = "empirical density")

p3 <- ggplot(dat, aes(x = Claim^(1/3))) + geom_density(colour = "blue") +
    labs(title = "Empirical density of claims amounts^(1/3)", x = "claims amounts", y = "empirical density")

grid.arrange(p1, p2, p3, ncol = 2)


## -----------------------------------------------------------------------------
pp <- ecdf(dat$Claim)
xx <- min(log(dat$Claim)) + 0:100/100 * (max(log(dat$Claim)) - min(log(dat$Claim)))
ES <- predict(locfit(log(1.00001 - pp(dat$Claim)) ~ log(dat$Claim), alpha = 0.1, deg = 2), newdata = xx)
dat_loglog <- data.frame(xx = xx, ES = ES)


## -----------------------------------------------------------------------------
ggplot(dat_loglog, aes(x = xx, y = ES)) + geom_line(colour = "blue", size = line_size) +
    geom_vline(xintercept = log(c(1:10) * 1000), colour = "green", linetype = "dashed") +
    geom_vline(xintercept = log(c(1:10) * 10000), colour = "yellow", linetype = "dashed") +
    geom_vline(xintercept = log(c(1:10) * 100000), colour = "orange", linetype = "dashed") +
    geom_vline(xintercept = log(c(1:10) * 1000000), colour = "red", linetype = "dashed") +
    labs(title = "log-log plot of claim amounts", x = "logged claim amount", y = "logged survival probability") +
    scale_x_continuous(breaks = seq(3,16,1))


## -----------------------------------------------------------------------------
col_names <- c("Age","Gender","MaritalStatus","DependentChildren","DependentsOther",
               "WeeklyPay","PartTimeFullTime","HoursWorkedPerWeek","DaysWorkedPerWeek",
               "AccYear","AccMonth","AccWeekday","AccTime","RepDelay")

global_avg <- mean(dat$Claim)
severity_limits <- c(0,40000)


## -----------------------------------------------------------------------------
dat_tmp <- dat
dat_tmp <- dat_tmp %>% mutate(
    WeeklyPay = pmin(1200, ceiling(WeeklyPay / 100) * 100),
    HoursWorkedPerWeek = pmin(60, HoursWorkedPerWeek),
    RepDelay = pmin(100, floor(RepDelay / 10) * 10)
)


## ----loop_plot, fig.height=6, fig.width=9-------------------------------------
for (k in 1:length(col_names)) {
    xvar <- col_names[k]
    
    out <- dat_tmp %>% group_by(!!sym(xvar)) %>% summarize(vol = n(), avg = mean(Claim))

    tmp <- dat_tmp %>% select(!!sym(xvar))
    global_n <- nrow(dat_tmp) / length(levels(factor(tmp[[1]])))

    plt1 <- ggplot(out, aes(x = !!sym(xvar), group = 1)) + geom_bar(aes(weight = vol), fill = "gray40") +
        geom_hline(yintercept = global_n, colour = "green", size = line_size) +
        labs(title = paste("Number of claims:", col_names[k]), x = col_names[k], y = "claim counts")

    plt2 <- ggplot(out, aes(x = !!sym(xvar), group = 1)) + geom_bar(aes(weight = avg), fill = "gray60") +
        geom_hline(yintercept = global_avg, colour = "blue", size = line_size) +
        coord_cartesian(ylim = severity_limits) +
        labs(title = paste("Average claim amount:", col_names[k]), x = col_names[k], y = "average claim amount")

    grid.arrange(plt1, plt2, ncol = 2, top = col_names[k])
}


## -----------------------------------------------------------------------------
sel_col <- c("Age","WeeklyPay","HoursWorkedPerWeek","DaysWorkedPerWeek",
             "AccYear","AccMonth","AccWeekday","AccTime","RepDelay")
dat_tmp <- dat[, sel_col]


## -----------------------------------------------------------------------------
corrMat <- round(cor(dat_tmp, method = "pearson"), 2)
corrMat
corrplot(corrMat, method = "color")


## -----------------------------------------------------------------------------
corrMat <- round(cor(dat_tmp, method = "spearman"), 2)
corrMat
corrplot(corrMat, method = "color")


## -----------------------------------------------------------------------------
# used/selected features
col_features <- c("AgeNN","GenderNN","DependentChildrenNN","DependentsOtherNN",
                  "WeeklyPayNN","PartTimeFullTimeNN","HoursWorkedPerWeekNN",
                  "DaysWorkedPerWeekNN","AccYearNN","AccMonthNN","AccWeekdayNN",
                  "AccTimeNN","RepDelayNN","Marital1","Marital2","Marital3")
col_names <- c("Age","Gender","DependentChildren","DependentsOther","WeeklyPay",
               "PartTimeFullTime","HoursWorkedPerWeek","DaysWorkedPerWeek",
               "AccYear","AccMonth","AccWeekday","AccTime","RepDelay",
               "Marital1","Marital2","Marital3")


## -----------------------------------------------------------------------------
# select p in [2,3]
p <- 2.5


## -----------------------------------------------------------------------------
# homogeneous model (learn)
(size_hom <- round(mean(learn$Claim)))
log_size_hom <- log(size_hom)


## -----------------------------------------------------------------------------
df_cmp %<>% bind_rows(
  data.frame(
    model = "Null model",
    learn_p2 = round(gamma_loss(learn$Claim, size_hom), 4),
    learn_pp = round(p_loss(learn$Claim, size_hom, p) * 10, 4),
    learn_p3 = round(ig_loss(learn$Claim, size_hom) * 1000, 4),
    test_p2 = round(gamma_loss(test$Claim, size_hom), 4),
    test_pp = round(p_loss(test$Claim, size_hom, p) * 10, 4),
    test_p3 = round(ig_loss(test$Claim, size_hom) * 1000, 4),
    avg_size = round(size_hom, 0)
  ))
df_cmp


## -----------------------------------------------------------------------------
# Size of input for neural networks
q0 <- length(col_features)
qqq <- c(q0, c(20,15,10), 1)

sprintf("Neural network with K=3 hidden layer")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", qqq[2])
sprintf("Number of hidden neurons second layer: q2 = %s", qqq[3])
sprintf("Number of hidden neurons third layer: q3 = %s", qqq[4])
sprintf("Output dimension: %s", qqq[5])


## -----------------------------------------------------------------------------
# matrices
YY <- as.matrix(as.numeric(learn$Claim))
XX <- as.matrix(learn[, col_features]) 
TT <- as.matrix(test[, col_features]) 


## -----------------------------------------------------------------------------
Design  <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design')

Output <- Design %>%    
    layer_dense(units = qqq[2], activation = 'tanh', name = 'layer1') %>%
    layer_dense(units = qqq[3], activation = 'tanh', name = 'layer2') %>%
    layer_dense(units = qqq[4], activation = 'tanh', name = 'layer3') %>%
    layer_dense(units = 1, activation = 'exponential', name = 'output', 
                weights = list(array(0, dim = c(qqq[4], 1)), array(log_size_hom, dim = c(1))))

model_p2 <- keras_model(inputs = list(Design), outputs = c(Output))


## -----------------------------------------------------------------------------
model_p2 %>% compile(
    loss = k_gamma_loss,
    optimizer = 'nadam'
)

summary(model_p2)


## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("./Networks/model_p2")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_p2 <- model_p2 %>%
  fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
  )


## -----------------------------------------------------------------------------
plot(fit_p2)


## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_p2, seed)


## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_p2, cp_path)


## -----------------------------------------------------------------------------
# calculating the predictions
learn$fitshp2 <- as.vector(model_p2 %>% predict(list(XX)))
test$fitshp2 <- as.vector(model_p2 %>% predict(list(TT)))

# average in-sample and out-of-sample losses (in 10^(0))
sprintf("Gamma deviance shallow network (train): %s", round(gamma_loss(learn$Claim, learn$fitshp2), 4))
sprintf("Gamma deviance shallow network (test): %s", round(gamma_loss(test$Claim, test$fitshp2), 4))

# average claims size
sprintf("Average size (test): %s", round(mean(test$fitshp2), 1))


## -----------------------------------------------------------------------------
df_cmp %<>% bind_rows(
  data.frame(model = "Plain-vanilla p2 (gamma)",
             learn_p2 = round(gamma_loss(learn$Claim, learn$fitshp2), 4),
             learn_pp = round(p_loss(learn$Claim, learn$fitshp2, p) * 10, 4),
             learn_p3 = round(ig_loss(learn$Claim, learn$fitshp2) * 1000, 4),
             test_p2 = round(gamma_loss(test$Claim, test$fitshp2), 4),
             test_pp = round(p_loss(test$Claim, test$fitshp2, p) * 10, 4),
             test_p3 = round(ig_loss(test$Claim, test$fitshp2) * 1000, 4),
             avg_size = round(mean(test$fitshp2), 0)
  ))
df_cmp


## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "shp2", "fitshp2")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "shp2", "fitshp2")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "shp2", "fitshp2")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "shp2", "fitshp2")

grid.arrange(plt1, plt2, plt3, plt4)


## -----------------------------------------------------------------------------
Design  <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design')

Output <- Design %>%    
    layer_dense(units=qqq[2], activation='tanh', name='layer1') %>%
    layer_dense(units=qqq[3], activation='tanh', name='layer2') %>%
    layer_dense(units=qqq[4], activation='tanh', name='layer3') %>%
    layer_dense(units=1, activation='exponential', name='output', 
                weights=list(array(0, dim=c(qqq[4],1)), array(log_size_hom, dim=c(1))))

model_pp <- keras_model(inputs = list(Design), outputs = c(Output))


## -----------------------------------------------------------------------------
model_pp %>% compile(
    loss = k_p_loss,
    optimizer = 'nadam'
)

summary(model_pp)


## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("./Networks/model_pp")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_pp <- model_pp %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),  
    verbose = verbose
)


## -----------------------------------------------------------------------------
plot(fit_pp)


## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_pp, seed)


## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_pp, cp_path)


## -----------------------------------------------------------------------------
# calculating the predictions
learn$fitshpp <- as.vector(model_pp %>% predict(list(XX)))
test$fitshpp <- as.vector(model_pp %>% predict(list(TT)))

# average in-sample and out-of-sample losses (in 10^(0))
sprintf("p-loss deviance shallow network (train): %s", round(p_loss(learn$Claim, learn$fitshpp, p), 4))
sprintf("p-loss deviance shallow network (test): %s", round(p_loss(test$Claim, test$fitshpp, p), 4))

# average claims size
sprintf("Average size (test): %s", round(mean(test$fitshpp), 1))


## -----------------------------------------------------------------------------
df_cmp %<>% bind_rows(
  data.frame(model = paste0("Plain-vanilla pp (p=", p,")"),
             learn_p2 = round(gamma_loss(learn$Claim, learn$fitshpp), 4),
             learn_pp = round(p_loss(learn$Claim, learn$fitshpp, p) * 10, 4),
             learn_p3 = round(ig_loss(learn$Claim, learn$fitshpp) * 1000, 4),
             test_p2 = round(gamma_loss(test$Claim, test$fitshpp), 4),
             test_pp = round(p_loss(test$Claim, test$fitshpp, p) * 10, 4),
             test_p3 = round(ig_loss(test$Claim, test$fitshpp) * 1000, 4),
             avg_size = round(mean(test$fitshpp), 0)
  ))
df_cmp


## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "shpp", "fitshpp")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "shpp", "fitshpp")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "shpp", "fitshpp")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "shpp", "fitshpp")

grid.arrange(plt1, plt2, plt3, plt4)


## -----------------------------------------------------------------------------
Design  <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design')

Output <- Design %>%    
    layer_dense(units=qqq[2], activation='tanh', name='layer1') %>%
    layer_dense(units=qqq[3], activation='tanh', name='layer2') %>%
    layer_dense(units=qqq[4], activation='tanh', name='layer3') %>%
    layer_dense(units=1, activation='exponential', name='output', 
                weights=list(array(0, dim=c(qqq[4],1)), array(log_size_hom, dim=c(1))))

model_p3 <- keras_model(inputs = list(Design), outputs = c(Output))


## -----------------------------------------------------------------------------
model_p3 %>% compile(
    loss = k_ig_loss,
    optimizer = 'nadam'
)

summary(model_p3)


## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("./Networks/model_p3")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_p3 <- model_p3 %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
)


## -----------------------------------------------------------------------------
plot(fit_p3)


## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_p3, seed)


## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_p3, cp_path)


## -----------------------------------------------------------------------------
# calculating the predictions
learn$fitshp3 <- as.vector(model_p3 %>% predict(list(XX)))
test$fitshp3 <- as.vector(model_p3 %>% predict(list(TT)))

# average in-sample and out-of-sample losses (in 10^(0))
sprintf("IG deviance shallow network (train): %s", round(ig_loss(learn$Claim, learn$fitshp3), 4))
sprintf("IG deviance shallow network (test): %s", round(ig_loss(test$Claim, test$fitshp3), 4))

# average claims size
sprintf("Average size (test): %s", round(mean(test$fitshp3), 1))


## -----------------------------------------------------------------------------
df_cmp %<>% bind_rows(
  data.frame(model = "Plain-vanilla p3 (inverse gaussian)",
             learn_p2 = round(gamma_loss(learn$Claim, learn$fitshp3), 4),
             learn_pp = round(p_loss(learn$Claim, learn$fitshp3, p) * 10, 4),
             learn_p3 = round(ig_loss(learn$Claim, learn$fitshp3) * 1000, 4),
             test_p2 = round(gamma_loss(test$Claim, test$fitshp3), 4),
             test_pp = round(p_loss(test$Claim, test$fitshp3, p) * 10, 4),
             test_p3 = round(ig_loss(test$Claim, test$fitshp3) * 1000, 4),
             avg_size = round(mean(test$fitshp3), 0)
  ))
df_cmp


## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "shp3", "fitshp3")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "shp3", "fitshp3")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "shp3", "fitshp3")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "shp3", "fitshp3")

grid.arrange(plt1, plt2, plt3, plt4)


## -----------------------------------------------------------------------------
df_cmp


## -----------------------------------------------------------------------------
# Size of input for neural networks
q0 <- length(col_features)
qqq <- c(q0, c(20, 15, 10), 1)

sprintf("Neural network with K=3 hidden layer")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", qqq[2])
sprintf("Number of hidden neurons second layer: q2 = %s", qqq[3])
sprintf("Number of hidden neurons third layer: q3 = %s", qqq[4])
sprintf("Number of hidden neurons third layer: q4 = %s", qqq[1])
sprintf("Output dimension: %s", qqq[5])


## -----------------------------------------------------------------------------
# matrices
YY <- as.matrix(as.numeric(learn$Claim))
XX <- as.matrix(learn[, col_features])
TT <- as.matrix(test[, col_features])


## -----------------------------------------------------------------------------
# neural network structure
Design  <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design') 

Attention <- Design %>%    
    layer_dense(units=qqq[2], activation='tanh', name='layer1') %>%
    layer_dense(units=qqq[3], activation='tanh', name='layer2') %>%
    layer_dense(units=qqq[4], activation='tanh', name='layer3') %>%
    layer_dense(units=qqq[1], activation='linear', name='attention')

Output <- list(Design, Attention) %>% layer_dot(name='LocalGLM', axes=1) %>% 
    layer_dense(
      units=1, activation='exponential', name='output',
      weights=list(array(0, dim=c(1,1)), array(log_size_hom, dim=c(1)))
    )


## -----------------------------------------------------------------------------
model_lgn_p2 <- keras_model(inputs = list(Design), outputs = c(Output))


## -----------------------------------------------------------------------------
model_lgn_p2 %>% compile(
    loss = k_gamma_loss,
    optimizer = 'nadam'
)
summary(model_lgn_p2)


## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("./Networks/model_lgn_p2")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_lgn_p2 <- model_lgn_p2 %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
)


## -----------------------------------------------------------------------------
plot(fit_lgn_p2)


## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_lgn_p2, seed)


## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_lgn_p2, cp_path)


## -----------------------------------------------------------------------------
# calculating the predictions
learn$fitlgnp2 <- as.vector(model_lgn_p2 %>% predict(list(XX)))
test$fitlgnp2 <- as.vector(model_lgn_p2 %>% predict(list(TT)))

# average in-sample and out-of-sample losses (in 10^(0))
sprintf("Gamma deviance shallow network (train): %s", round(gamma_loss(learn$Claim, learn$fitlgnp2), 4))
sprintf("Gamma deviance shallow network (test): %s", round(gamma_loss(test$Claim, test$fitlgnp2), 4))

# average claims size
sprintf("Average size (test): %s", round(mean(test$fitlgnp2), 1))


## -----------------------------------------------------------------------------
df_cmp %<>% bind_rows(
  data.frame(model = "LocalGLMnet p2 (gamma)",
             learn_p2 = round(gamma_loss(learn$Claim, learn$fitlgnp2), 4),
             learn_pp = round(p_loss(learn$Claim, learn$fitlgnp2, p) * 10, 4),
             learn_p3 = round(ig_loss(learn$Claim, learn$fitlgnp2) * 1000, 4),
             test_p2 = round(gamma_loss(test$Claim, test$fitlgnp2), 4),
             test_pp = round(p_loss(test$Claim, test$fitlgnp2, p) * 10, 4),
             test_p3 = round(ig_loss(test$Claim, test$fitlgnp2) * 1000, 4),
             avg_size = round(mean(test$fitlgnp2), 0)
  ))
df_cmp


## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "lgnp2", "fitlgnp2")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "lgnp2", "fitlgnp2")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "lgnp2", "fitlgnp2")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "lgnp2", "fitlgnp2")

grid.arrange(plt1, plt2, plt3, plt4)


## -----------------------------------------------------------------------------
model_lgn_pp <- keras_model(inputs = list(Design), outputs = c(Output))


## -----------------------------------------------------------------------------
model_lgn_pp %>% compile(
    loss = k_p_loss,
    optimizer = 'nadam'
)
summary(model_lgn_pp)


## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("./Networks/model_lgn_pp")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_lgn_pp <- model_lgn_pp %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
)


## -----------------------------------------------------------------------------
plot(fit_lgn_pp)


## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_lgn_pp, seed)


## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_lgn_pp, cp_path)


## -----------------------------------------------------------------------------
# calculating the predictions
learn$fitlgnpp <- as.vector(model_pp %>% predict(list(XX)))
test$fitlgnpp <- as.vector(model_pp %>% predict(list(TT)))

# average in-sample and out-of-sample losses (in 10^(0))
sprintf("p-loss deviance shallow network (train): %s", round(p_loss(learn$Claim, learn$fitlgnpp, p), 4))
sprintf("p-loss deviance shallow network (test): %s", round(p_loss(test$Claim, test$fitlgnpp, p), 4))

# average claims size
sprintf("Average size (test): %s", round(mean(test$fitlgnpp), 1))


## -----------------------------------------------------------------------------
df_cmp %<>% bind_rows(
  data.frame(model = "LocalGLMnet pp (p=2.5)",
             learn_p2 = round(gamma_loss(learn$Claim, learn$fitlgnpp), 4),
             learn_pp = round(p_loss(learn$Claim, learn$fitlgnpp, p) * 10, 4),
             learn_p3 = round(ig_loss(learn$Claim, learn$fitlgnpp) * 1000, 4),
             test_p2 = round(gamma_loss(test$Claim, test$fitlgnpp), 4),
             test_pp = round(p_loss(test$Claim, test$fitlgnpp, p) * 10, 4),
             test_p3 = round(ig_loss(test$Claim, test$fitlgnpp) * 1000, 4),
             avg_size = round(mean(test$fitlgnpp), 0)
  ))
df_cmp


## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "lgnpp", "fitlgnpp")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "lgnpp", "fitlgnpp")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "lgnpp", "fitlgnpp")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "lgnpp", "fitlgnpp")

grid.arrange(plt1, plt2, plt3, plt4)


## -----------------------------------------------------------------------------
model_lgn_p3 <- keras_model(inputs = list(Design), outputs = c(Output))


## -----------------------------------------------------------------------------
model_lgn_p3 %>% compile(
    loss = k_ig_loss,
    optimizer = 'nadam'
)
summary(model_lgn_p3)


## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("./Networks/model_lgn_p3")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_lgn_p3 <- model_lgn_p3 %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
)


## -----------------------------------------------------------------------------
plot(fit_lgn_p3)


## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_lgn_p3, seed)


## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_lgn_p3, cp_path)


## -----------------------------------------------------------------------------
# calculating the predictions
learn$fitlgnp3 <- as.vector(model_lgn_p3 %>% predict(list(XX)))
test$fitlgnp3 <- as.vector(model_lgn_p3 %>% predict(list(TT)))

# average in-sample and out-of-sample losses (in 10^(0))
sprintf("IG deviance shallow network (train): %s", round(ig_loss(learn$Claim, learn$fitlgnp3), 4))
sprintf("IG deviance shallow network (test): %s", round(ig_loss(test$Claim, test$fitlgnp3), 4))

# average claims size
sprintf("Average size (test): %s", round(mean(test$fitlgnp3), 1))


## -----------------------------------------------------------------------------
df_cmp %<>% bind_rows(
  data.frame(model = "LocalGLMnet p3 (inverse gaussian)",
             learn_p2 = round(gamma_loss(learn$Claim, learn$fitlgnp3), 4),
             learn_pp = round(p_loss(learn$Claim, learn$fitlgnp3, p) * 10, 4),
             learn_p3 = round(ig_loss(learn$Claim, learn$fitlgnp3) * 1000, 4),
             test_p2 = round(gamma_loss(test$Claim, test$fitlgnp3), 4),
             test_pp = round(p_loss(test$Claim, test$fitlgnp3, p) * 10, 4),
             test_p3 = round(ig_loss(test$Claim, test$fitlgnp3) * 1000, 4),
             avg_size = round(mean(test$fitlgnp3), 0)
  ))
df_cmp


## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "lgnp3", "fitlgnp3")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "lgnp3", "fitlgnp3")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "lgnp3", "fitlgnp3")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "lgnp3", "fitlgnp3")

grid.arrange(plt1, plt2, plt3, plt4, top="LocalGLMnet Inverse Gaussian")


## -----------------------------------------------------------------------------
df_cmp


## -----------------------------------------------------------------------------
# used/selected features
col_features <- c("AgeNN","GenderNN","DependentChildrenNN","DependentsOtherNN",
                  "WeeklyPayNN","PartTimeFullTimeNN","HoursWorkedPerWeekNN",
                  "DaysWorkedPerWeekNN","AccYearNN","AccMonthNN","AccWeekdayNN",
                  "AccTimeNN","RepDelayNN","RandUN","RandNN","Marital1","Marital2","Marital3") 
col_names <- c("Age","Gender","DependentChildren","DependentsOther","WeeklyPay",
               "PartTimeFullTime","HoursWorkedPerWeek","DaysWorkedPerWeek",
               "AccYear","AccMonth","AccWeekday","AccTime","RepDelay","RandUN",
               "RandNN","Marital1","Marital2","Marital3")


## -----------------------------------------------------------------------------
# Size of input for neural networks
q0 <- length(col_features)
qqq <- c(q0, c(20, 15, 10), 1)

sprintf("Neural network with K=3 hidden layer")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", qqq[2])
sprintf("Number of hidden neurons second layer: q2 = %s", qqq[3])
sprintf("Number of hidden neurons third layer: q3 = %s", qqq[4])
sprintf("Number of hidden neurons third layer: q4 = %s", qqq[1])
sprintf("Output dimension: %s", qqq[5])


## -----------------------------------------------------------------------------
# matrices
YY <- as.matrix(as.numeric(learn$Claim))
XX <- as.matrix(learn[, col_features]) 
TT <- as.matrix(test[, col_features])


## -----------------------------------------------------------------------------
# neural network structure
Design <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design') 

Attention <- Design %>%    
    layer_dense(units=qqq[2], activation='tanh', name='layer1') %>%
    layer_dense(units=qqq[3], activation='tanh', name='layer2') %>%
    layer_dense(units=qqq[4], activation='tanh', name='layer3') %>%
    layer_dense(units=qqq[1], activation='linear', name='attention')

Output <- list(Design, Attention) %>% layer_dot(name='LocalGLM', axes=1) %>% 
    layer_dense(
      units=1, activation='exponential', name='output', 
      weights=list(array(0, dim=c(1,1)), array(log_size_hom, dim=c(1))))


## -----------------------------------------------------------------------------
model_var <- keras_model(inputs = list(Design), outputs = c(Output))


## -----------------------------------------------------------------------------
model_var %>% compile(
    loss = k_gamma_loss,
    optimizer = 'nadam'
)
summary(model_var)


## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("./Networks/model_var")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_var <- model_var %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
)


## -----------------------------------------------------------------------------
plot(fit_var)


## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_var, seed)


## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_var, cp_path)


## -----------------------------------------------------------------------------
# calculating the predictions
learn$fitvar <- as.vector(model_var %>% predict(list(XX)))
test$fitvar <- as.vector(model_var %>% predict(list(TT)))

# average in-sample and out-of-sample losses (in 10^(0))
sprintf("Gamma deviance: %s", round(gamma_loss(learn$Claim, learn$fitvar), 4))
sprintf("Gamma deviance: %s", round(gamma_loss(test$Claim, test$fitvar), 4))

# average claims size
sprintf("Average claim size (test): %s", round(mean(test$fitvar), 1))


## -----------------------------------------------------------------------------
# select submodel such that the prediction provides the unweighted regression attentions
zz <- keras_model(inputs = model_var$input, outputs = get_layer(model_var, 'attention')$output)
summary(zz)


## -----------------------------------------------------------------------------
# Calculate the regression attentions for model zz on the test set
beta_x <- data.frame(zz %>% predict(list(TT)))
names(beta_x) <- paste0("Beta", col_names)


## -----------------------------------------------------------------------------
# multiply with the corresponding weight
ww <- get_weights(model_var)
beta_x <- beta_x * as.numeric(ww[[9]])
str(beta_x)


## -----------------------------------------------------------------------------
# attention beta(x) on the learning set
beta_xL <- data.frame(zz %>% predict(list(XX)))
names(beta_xL) <- paste0("Beta", col_names)
beta_xL <- beta_xL * as.numeric(ww[[9]])


## -----------------------------------------------------------------------------
# mean and std.dev. of additional (randomly generated) components q+1 and q+2
mean_rand <- c(mean(beta_xL$BetaRandUN), mean(beta_xL$BetaRandNN))
sd_rand <- c(sd(beta_xL$BetaRandUN), sd(beta_xL$BetaRandNN))
sprintf("The means of the random variables regression attentions are:  %s", paste(round(mean_rand, 4), collapse = "  "))
sprintf("The stdevs of the random variables regression attentions are:  %s", paste(round(sd_rand, 4), collapse = "  "))


## -----------------------------------------------------------------------------
# 0.1% rejection region
quant_rand <- mean(sd_rand) * abs(qnorm(0.0005))
cat("The 0.1% rejection region is 0 +/-", quant_rand)


## -----------------------------------------------------------------------------
# in-sample coverage ratio for all features
num_names <- 1:(length(col_names)-3)
II <- data.frame(array(NA, c(1, length(num_names))))
names(II) <- col_names[num_names] 
for (k1 in 1:length(num_names)) {
  II[1, k1] <- 1 - (sum(as.integer(-beta_xL[, num_names[k1]] > quant_rand)) +
                    sum(as.integer(beta_xL[, num_names[k1]] > quant_rand))) / nrow(beta_xL)
}
round(II, 4)


## -----------------------------------------------------------------------------
## merge covariates x with beta(x) on the test set
beta_x <- cbind(TT, beta_x)


## -----------------------------------------------------------------------------
## select at random 5000 claims to not overload plots
nsample <- 5000
set.seed(seed)
idx <- sample(x = 1:nrow(test), size = nsample)


## -----------------------------------------------------------------------------
# data for plotting
beta_x_smp <- beta_x[idx, ]
test_smp <- test[idx, ]
test_smp$Gender <- as.numeric(test_smp$Gender)
test_smp$PartTimeFullTime <- as.numeric(test_smp$PartTimeFullTime)


## -----------------------------------------------------------------------------
# Plotting for all continuous and binary features
for (ll in 1:length(num_names)) {
    kk <- num_names[ll]
    dat_plt <- data.frame(var = test_smp[, col_names[ll]],
                          bx = beta_x_smp[, kk + length(col_features)],
                          col = rep("green", nsample))
    plt <- ggplot(dat_plt, aes(x = var, y = bx)) + geom_point() + 
              geom_hline(yintercept = 0, colour = "red", size = line_size) + 
              geom_hline(yintercept = c(-quant_rand, quant_rand), colour = "green", size = line_size) +
              geom_hline(yintercept = c(-1,1)/4, colour = "orange", size = line_size, linetype = "dashed") +
              geom_rect(
                mapping = aes(xmin = min(var), xmax = max(var), ymin = -quant_rand, ymax = quant_rand),
                fill = dat_plt$col, alpha = 0.002
              ) + lims(y = c(-0.75,0.75)) +
              labs(title = paste0("Regression attention: ", col_names[ll]),
                   subtitle = paste0("Coverage Ratio: ", paste0(round(II[, col_names[ll]] * 100, 2)), "%"),
                   x = paste0(col_names[ll], " x"), y = "regression attention beta(x)")
    print(plt)
}


## -----------------------------------------------------------------------------
# calculate variable importance
var_imp <- abs(beta_xL[, num_names])

# store as data.frame for plotting
dat_var_imp <- data.frame(vi = colMeans(var_imp), names = col_names[1:length(num_names)])

# limits from random generated variables
dat_var_imp_limit <- dat_var_imp %>% filter(names %in% c("RandUN", "RandNN"))
limit_rand <- max(dat_var_imp_limit$vi)


## -----------------------------------------------------------------------------
ggplot(dat_var_imp, aes(x = vi)) + geom_col(aes(y = reorder(names, vi))) +
    geom_vline(xintercept = seq(0.1, 0.4, by = 0.1), col = "gray1", linetype = "dashed") +
    geom_vline(xintercept = limit_rand, col = "red", size = line_size) +
    theme(axis.text = element_text(size = 12)) +
    labs(title = "Variable importance", x = "variable importance", y = "variable")


## -----------------------------------------------------------------------------
reduce <- c("AccWeekday", "HoursWorkedPerWeek", "DaysWorkedPerWeek")

col_featuresR <- setdiff(col_features, c(paste0(reduce, "NN"), "RandUN", "RandNN"))
col_namesR <- setdiff(col_names, c(reduce, "RandUN", "RandNN"))


## -----------------------------------------------------------------------------
# Size of input for neural networks
q0 <- length(col_featuresR)
qqq <- c(q0, c(20, 15, 10), 1)

sprintf("Neural network with K=3 hidden layer")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", qqq[2])
sprintf("Number of hidden neurons second layer: q2 = %s", qqq[3])
sprintf("Number of hidden neurons third layer: q3 = %s", qqq[4])
sprintf("Output dimension: %s", qqq[5])


## -----------------------------------------------------------------------------
# matrices
# YY remains the same
XX <- as.matrix(learn[, col_featuresR])
TT <- as.matrix(test[, col_featuresR])


## -----------------------------------------------------------------------------
# neural network structure
Design <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design') 

Attention <- Design %>%    
    layer_dense(units = qqq[2], activation = 'tanh', name = 'layer1') %>%
    layer_dense(units = qqq[3], activation = 'tanh', name = 'layer2') %>%
    layer_dense(units = qqq[4], activation = 'tanh', name = 'layer3') %>%
    layer_dense(units = qqq[1], activation = 'linear', name = 'attention')

Output <- list(Design, Attention) %>% layer_dot(name = 'LocalGLM', axes = 1) %>% 
    layer_dense(units = 1, activation = 'exponential', name = 'output', 
                weights = list(array(0, dim = c(1,1)), array(log_size_hom, dim = c(1))))


## -----------------------------------------------------------------------------
model_red <- keras_model(inputs = list(Design), outputs = c(Output))


## -----------------------------------------------------------------------------
model_red %>% compile(
    loss = k_gamma_loss,
    optimizer = 'nadam'
)
summary(model_red)


## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("./Networks/model_red")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_red <- model_red %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
)


## -----------------------------------------------------------------------------
plot(fit_red)


## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_red, seed)


## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_red, cp_path)


## -----------------------------------------------------------------------------
# calculating the predictions
learn$fitred <- as.vector(model_red %>% predict(list(XX)))
test$fitred <- as.vector(model_red %>% predict(list(TT)))

# average in-sample and out-of-sample losses (in 10^(0))
sprintf("Gamma deviance: %s", round(gamma_loss(learn$Claim, learn$fitred), 4))
sprintf("Gamma deviance: %s", round(gamma_loss(test$Claim, test$fitred), 4))

# average claims size
sprintf("Average claim size (test): %s", round(mean(test$fitred), 0))


## -----------------------------------------------------------------------------
# define predicted layer and corresponding weights
zz <- keras_model(inputs = model_red$input, outputs = get_layer(model_red, 'attention')$output)
ww <- get_weights(model_red)


## -----------------------------------------------------------------------------
## attention weights of test samples
beta_x <- data.frame(zz %>% predict(list(TT)))
names(beta_x) <- paste0("Beta", col_namesR)
beta_x <- beta_x * as.numeric(ww[[9]])
str(beta_x)


## -----------------------------------------------------------------------------
## merge covariates x with beta(x)
beta_x <- cbind(TT, beta_x)


## -----------------------------------------------------------------------------
## select at random 5000 claims to not overload plots
nsample <- 5000
set.seed(seed)
idx <- sample(x = 1:nrow(test), size = nsample)


## -----------------------------------------------------------------------------
# continuous, binary and categorical variables in col_namesR
var_cont <- c(1, 5, 7, 8, 9, 10)
var_bin <- c(2, 3, 4, 6)
var_cat <- 11:13


## -----------------------------------------------------------------------------
# data for plotting
beta_x_smp <- beta_x[idx, ]
test_smp <- test[idx, ]


## -----------------------------------------------------------------------------
# Plotting for all continuous features
for (ll in 1:length(var_cont)) {
    kk <- var_cont[ll]
    dat_plt <- data.frame(var = test_smp[, col_namesR[kk]],
                          bx = beta_x_smp[, kk + length(col_featuresR)] * beta_x_smp[, kk],
                          col = rep("green", nsample))
    plt <- ggplot(dat_plt, aes(x = var, y = bx)) + geom_point() +
        geom_smooth(size = line_size) +
        geom_hline(yintercept = 0, colour = "red", size = line_size) +
        geom_hline(yintercept = c(-1, 1) / 4, colour = "orange", size = line_size, linetype = "dashed") +
        lims(y = c(-1.25, 1.25)) +
        labs(title = paste0("Covariate contribution: ", col_namesR[kk]),
             x = paste0(col_namesR[kk], " x"),
             y = "covariate contribution beta(x) * x")
    suppressMessages(print(plt))
}


## -----------------------------------------------------------------------------
# Plotting for all binary features
for (ll in 1:length(var_bin)) {
    kk <- var_bin[ll]
    dat_plt <- data.frame(var = factor(test_smp[, col_namesR[kk]]),
                          bx = beta_x_smp[, kk + length(col_featuresR)] * beta_x_smp[, kk],
                          col = rep("green", nsample))
    plt <- ggplot(dat_plt, aes(x = var, y = bx)) + geom_boxplot() +
        geom_hline(yintercept = 0, colour = "red", size = line_size) +
        geom_hline(yintercept = c(-1,1)/4, colour = "orange", size = line_size, linetype = "dashed") +
        lims(y = c(-1.25, 1.25)) +
        labs(title = paste0("Covariate contribution: ", col_namesR[kk]),
             x = paste0(col_namesR[kk], " x"),
             y = "covariate contribution beta(x) * x")
    suppressMessages(print(plt))
}


## -----------------------------------------------------------------------------
# data preparation for plotting
beta_xx <- cbind(test$MaritalStatus, beta_x[, 11:13] * beta_x[, 24:26])
beta_xx$BetaMarital <- rowSums(beta_xx[, -1]) 
beta_xx <- beta_xx[, c(1, ncol(beta_xx))]
names(beta_xx) <- c("MaritalStatus", "BetaMarital")
str(beta_xx)


## -----------------------------------------------------------------------------
namesCat <- "MaritalStatus"
dat_plt <- beta_xx[idx, ]


## -----------------------------------------------------------------------------
ggplot(dat_plt, aes(x = MaritalStatus, y = BetaMarital)) + geom_boxplot() +
     geom_hline(yintercept = 0, colour = "red", size = line_size) +
     geom_hline(yintercept = c(-1, 1) / 4, colour = "orange", size = line_size, linetype = "dashed") +
     lims(y = c(-1.25, 1.25)) +
     labs(title = paste0("Covariate contribution: ", namesCat),
          x = paste0(namesCat, " x"),
          y = "covariate contribution beta(x) * x")


## -----------------------------------------------------------------------------
# limits for plotting (y-axis)
ax_limit <- c(-.6, .6)

# number of support points
n_points <- 100


## -----------------------------------------------------------------------------
zz <- keras_model(inputs = model_red$input, outputs = get_layer(model_red, 'attention')$output)
ww <- get_weights(zz)


## -----------------------------------------------------------------------------
Input <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'Design2') 

Attention <- Input %>% 
          layer_dense(units = qqq[2], activation = 'tanh', name = 'FNLayer1') %>%
          layer_dense(units = qqq[3], activation = 'tanh', name = 'FNLayer2') %>%
          layer_dense(units = qqq[4], activation = 'tanh', name = 'FNLayer3') %>%
          layer_dense(units = qqq[1], activation = 'linear', name = 'attention')

model_int <- keras_model(inputs = c(Input), outputs = c(Attention))  


## -----------------------------------------------------------------------------
set_weights(model_int, ww)


## -----------------------------------------------------------------------------
col_names_cont <- col_namesR[var_cont]
n_col_names <- length(col_names_cont)


## -----------------------------------------------------------------------------
for (jj in 1:length(col_names_cont)) {
    beta_j <- Attention %>% layer_lambda(function(x) x[, var_cont[jj]])
    model_grad1 <- keras_model(inputs = c(Input), outputs = c(beta_j))
    grad <- beta_j %>% layer_lambda(function(x) k_gradients(model_grad1$outputs, model_grad1$inputs))

    model_grad2 <- keras_model(inputs = c(Input), outputs = c(grad))
    grad_beta <- data.frame(model_grad2 %>% predict(as.matrix(TT)))
    grad_beta <- grad_beta[, var_cont]
    names(grad_beta) <- paste0("Grad", col_names_cont)

    beta_x <- cbind(test[, col_names_cont[jj]], grad_beta)
    names(beta_x)[1] <- col_names_cont[jj]
    beta_x <- beta_x[order(beta_x[, 1]), ]

    rr <- range(beta_x[, 1])
    xx <- rr[1] + (rr[2] - rr[1]) * 0:n_points / n_points
    yy <- array(NA, c(n_points + 1, n_col_names))
    for (kk in 1:length(var_cont)) {
        yy[, kk] <- predict(locfit(beta_x[, kk + 1]~ beta_x[, 1], alpha = 0.7, deg = 2), newdata = xx)
    }

    dat_plt <- data.frame(xx, yy)
    colnames(dat_plt) <- c("x", col_names_cont)
    dat_plt <- dat_plt %>% gather(key = "variable", value = "value", -x)

    plt <- ggplot(dat_plt, aes(x = x, y = value)) + 
        geom_line(aes(color = variable), size = line_size) +
        ylim(ax_limit) +
        labs(title = paste0("Interactions of covariate ", col_names_cont[jj]),
             x = col_names_cont[jj],
             y = "interaction strengths")
    print(plt)
}


## -----------------------------------------------------------------------------
col_names_cont <- col_namesR[var_cont]
col_names_bin <- col_namesR[var_bin]
n_col_names <- length(col_names_bin)


## -----------------------------------------------------------------------------
for (jj in 1:length(col_names_cont)) {
    beta_j <- Attention %>% layer_lambda(function(x) x[, var_cont[jj]])
    model_grad1 <- keras_model(inputs = c(Input), outputs = c(beta_j))
    grad <- beta_j %>% layer_lambda(function(x) k_gradients(model_grad1$outputs, model_grad1$inputs))

    model_grad2 <- keras_model(inputs = c(Input), outputs = c(grad))
    grad_beta <- data.frame(model_grad2 %>% predict(as.matrix(TT)))
    grad_beta <- grad_beta[, var_cont]
    names(grad_beta) <- paste0("Grad", col_names_cont)

    beta_x <- cbind(test[, col_names_cont[jj]], grad_beta)
    names(beta_x)[1] <- col_names_cont[jj]
    beta_x <- beta_x[order(beta_x[, 1]), ]

    rr <- range(beta_x[, 1])
    xx <- rr[1] + (rr[2] - rr[1]) * 0:n_points/n_points
    yy <- array(NA, c(n_points + 1, n_col_names))
    for (kk in 1:length(var_bin)) {
        yy[, kk] <- predict(locfit(beta_x[, kk + 1]~ beta_x[, 1] , alpha = 0.7, deg = 2), newdata = xx)
    }

    dat_plt <- data.frame(xx, yy)
    colnames(dat_plt) <- c("x", col_names_bin)
    dat_plt <- dat_plt %>% gather(key = "variable", value = "value", -x)

    plt <- ggplot(dat_plt, aes(x = x, y = value)) + 
        geom_line(aes(color = variable), size = line_size) +
        ylim(ax_limit) +
        labs(title = paste0("Interactions of covariate ", col_names_cont[jj]),
             x = col_names_cont[jj],
             y = "interaction strengths")
    print(plt)
}


## -----------------------------------------------------------------------------
sessionInfo()


## -----------------------------------------------------------------------------
reticulate::py_config()


## -----------------------------------------------------------------------------
tensorflow::tf_version()

