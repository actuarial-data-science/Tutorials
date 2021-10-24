## -----------------------------------------------------------------------------
# library(mgcv)
library(dplyr)
library(tibble)
library(ggplot2)
library(splitTools)


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


## -----------------------------------------------------------------------------
summarize <- function(...) suppressMessages(dplyr::summarize(...))


## -----------------------------------------------------------------------------
load_data <- function(file) {
  load(file.path("../0_data/", file), envir = parent.frame(1))
}


## -----------------------------------------------------------------------------
# Poisson deviance
PoissonDeviance <- function(pred, obs) {
  200 * (sum(pred) - sum(obs) + sum(log((obs / pred)^(obs)))) / length(pred)
}


## -----------------------------------------------------------------------------
plot_freq <- function(test, xvar, title, model, mdlvariant) {
  out <- test %>% group_by(!!sym(xvar)) %>% summarize(obs = sum(ClaimNb) / sum(Exposure),
                                                      pred = sum(!!sym(mdlvariant)) / sum(Exposure))
  
  ggplot(out, aes(x = !!sym(xvar), group = 1)) + geom_point(aes(y = pred, colour = model)) +
    geom_point(aes(y = obs, colour = "observed")) +
    geom_line(aes(y = pred, colour = model), linetype = "dashed") +
    geom_line(aes(y = obs, colour = "observed"), linetype = "dashed") +
    ylim(0, 0.35) + labs(x = xvar, y = "frequency", title = title) +
    theme(legend.position = "bottom")
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
n_l <- nrow(train)
n_t <- nrow(test)
sprintf("Number of observations (train): %s", n_l)
sprintf("Number of observations (test): %s", n_t)

# Claims frequency of train/test
sprintf("Empirical frequency (train): %s", round(sum(train$ClaimNb) / sum(train$Exposure), 4))
sprintf("Empirical frequency (test): %s", round(sum(test$ClaimNb) / sum(test$Exposure), 4))


## -----------------------------------------------------------------------------
# exposure and number of claims of train/test
# see https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3164764, p. 11 (figures do not match)
train1 <- train %>% group_by(ClaimNb) %>% summarize(n = n(), exp = sum(Exposure))
print(train1)
print(round(100 * train1$n / sum(train1$n), 3))

test1 <- test %>% group_by(ClaimNb) %>% summarize(n = n(), exp = sum(Exposure))
print(test1)
print(round(100 * test1$n / sum(test1$n), 3))


## -----------------------------------------------------------------------------
# table to store all model results for comparison
df_cmp <- tibble(
  model = character(),
  run_time = numeric(),
  parameters = numeric(),
  aic = numeric(),
  in_sample_loss = numeric(),
  out_sample_loss = numeric(),
  avg_freq = numeric()
)


## -----------------------------------------------------------------------------
exec_time <- system.time(glm0 <- glm(ClaimNb ~ 1, data = train, offset = log(Exposure), family = poisson()))
exec_time[1:5]
summary(glm0)


## -----------------------------------------------------------------------------
# Predictions
train$fitGLM0 <- fitted(glm0)
test$fitGLM0 <- predict(glm0, newdata = test, type = "response")
dat$fitGLM0 <- predict(glm0, newdata = dat2, type = "response")


## -----------------------------------------------------------------------------
# in-sample and out-of-sample losses (in 10^(-2))
sprintf("100 x Poisson deviance GLM (train): %s", PoissonDeviance(train$fitGLM0, train$ClaimNb))
sprintf("100 x Poisson deviance GLM (test): %s", PoissonDeviance(test$fitGLM0, test$ClaimNb))

# Overall estimated frequency
sprintf("average frequency (test): %s", round(sum(test$fitGLM0) / sum(test$Exposure), 6))


## -----------------------------------------------------------------------------
df_cmp[1, ] <- list("GLM0", round(exec_time[[3]], 0), length(coef(glm0)), round(AIC(glm0), 0),
                   round(PoissonDeviance(train$fitGLM0, as.vector(unlist(train$ClaimNb))), 4),
                   round(PoissonDeviance(test$fitGLM0, as.vector(unlist(test$ClaimNb))), 4),
                   round(sum(test$fitGLM0) / sum(test$Exposure), 4))
knitr::kable(df_cmp)


## -----------------------------------------------------------------------------
exec_time <- system.time(
  glm1 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand +
                        VehGas + DensityGLM + Region + AreaGLM,
              data = train, offset = log(Exposure), family = poisson()))
exec_time[1:5]
summary(glm1)


## -----------------------------------------------------------------------------
# needs sufficient resources!
drop1(glm1, test = "LRT")


## -----------------------------------------------------------------------------
# needs sufficient resources!
anova(glm1)


## -----------------------------------------------------------------------------
# Predictions
train$fitGLM1 <- fitted(glm1)
test$fitGLM1 <- predict(glm1, newdata = test, type = "response")
dat$fitGLM1 <- predict(glm1, newdata = dat2, type = "response")


## -----------------------------------------------------------------------------
# in-sample and out-of-sample losses (in 10^(-2))
sprintf("100 x Poisson deviance GLM (train): %s", PoissonDeviance(train$fitGLM1, train$ClaimNb))
sprintf("100 x Poisson deviance GLM (test): %s", PoissonDeviance(test$fitGLM1, test$ClaimNb))

# Overall estimated frequency
sprintf("average frequency (test): %s", round(sum(test$fitGLM1) / sum(test$Exposure), 4))


## -----------------------------------------------------------------------------
df_cmp[2, ] <- list("GLM1", round(exec_time[[3]], 0), length(coef(glm1)), round(AIC(glm1), 0),
                   round(PoissonDeviance(train$fitGLM1, as.vector(unlist(train$ClaimNb))), 4),
                   round(PoissonDeviance(test$fitGLM1, as.vector(unlist(test$ClaimNb))), 4),
                   round(sum(test$fitGLM1) / sum(test$Exposure), 4))
knitr::kable(df_cmp)


## -----------------------------------------------------------------------------
# Area
p1 <- plot_freq(test, "AreaGLM", "frequency by area", "GLM", "fitGLM1")

# VehPower
p2 <- plot_freq(test, "VehPowerGLM", "frequency by vehicle power", "GLM", "fitGLM1")

# VehBrand
p3 <- plot_freq(test, "VehBrand", "frequency by vehicle brand", "GLM", "fitGLM1")

# VehAge
p4 <- plot_freq(test, "VehAgeGLM", "frequency by vehicle age", "GLM", "fitGLM1")

gridExtra::grid.arrange(p1, p2, p3, p4)


## -----------------------------------------------------------------------------
exec_time <- system.time(
  glm2 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM +
                        VehBrand + VehGas + DensityGLM + Region,
              data = train, offset = log(Exposure), family = poisson()))
exec_time[1:5]
summary(glm2)


## -----------------------------------------------------------------------------
# needs sufficient resources!
drop1(glm2, test = "LRT")


## -----------------------------------------------------------------------------
# needs sufficient resources!
anova(glm2)


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
df_cmp[3, ] <- list("GLM2", round(exec_time[[3]], 0), length(coef(glm2)), round(AIC(glm2), 0),
                   round(PoissonDeviance(train$fitGLM2, as.vector(unlist(train$ClaimNb))), 4),
                   round(PoissonDeviance(test$fitGLM2, as.vector(unlist(test$ClaimNb))), 4),
                   round(sum(test$fitGLM2) / sum(test$Exposure), 4))
knitr::kable(df_cmp)


## -----------------------------------------------------------------------------
# Area
p1 <- plot_freq(test, "Region", "frequency by area", "GLM", "fitGLM2")

# VehPower
p2 <- plot_freq(test, "VehPowerGLM", "frequency by vehicle power", "GLM", "fitGLM2")

# VehBrand
p3 <- plot_freq(test, "VehBrand", "frequency by vehicle brand", "GLM", "fitGLM2")

# VehAge
p4 <- plot_freq(test, "VehAgeGLM", "frequency by vehicle age", "GLM", "fitGLM2")

gridExtra::grid.arrange(p1, p2, p3, p4)


## -----------------------------------------------------------------------------
exec_time <- system.time(
  glm3 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM +
                        VehGas + DensityGLM + Region,
              data = train, offset = log(Exposure), family = poisson()))
exec_time[1:5]
summary(glm3)


## -----------------------------------------------------------------------------
# needs sufficient resources!
drop1(glm3, test = "LRT")


## -----------------------------------------------------------------------------
# needs sufficient resources!
anova(glm3)


## -----------------------------------------------------------------------------
# Predictions
train$fitGLM3 <- fitted(glm3)
test$fitGLM3 <- predict(glm3, newdata = test, type = "response")
dat$fitGLM3 <- predict(glm3, newdata = dat2, type = "response")


## -----------------------------------------------------------------------------
# in-sample and out-of-sample losses (in 10^(-2))
sprintf("100 x Poisson deviance GLM (train): %s", PoissonDeviance(train$fitGLM3, train$ClaimNb))
sprintf("100 x Poisson deviance GLM (test): %s", PoissonDeviance(test$fitGLM3, test$ClaimNb))

# Overall estimated frequency
sprintf("average frequency (test): %s", round(sum(test$fitGLM3) / sum(test$Exposure), 4))


## -----------------------------------------------------------------------------
df_cmp[4, ] <- list("GLM3", round(exec_time[[3]], 0), length(coef(glm3)), round(AIC(glm3), 0),
                   round(PoissonDeviance(train$fitGLM3, as.vector(unlist(train$ClaimNb))), 4),
                   round(PoissonDeviance(test$fitGLM3, as.vector(unlist(test$ClaimNb))), 4),
                   round(sum(test$fitGLM3) / sum(test$Exposure), 4))
knitr::kable(df_cmp)


## -----------------------------------------------------------------------------
# Region
p1 <- plot_freq(test, "Region", "frequency by area", "GLM", "fitGLM3")

# VehPowerGLM
p2 <- plot_freq(test, "VehPowerGLM", "frequency by vehicle power", "GLM", "fitGLM3")

# DriveAgeGLM
p3 <- plot_freq(test, "DrivAgeGLM", "frequency by vehicle brand", "GLM", "fitGLM3")

# VehAgeGLM
p4 <- plot_freq(test, "VehAgeGLM", "frequency by vehicle age", "GLM", "fitGLM3")

gridExtra::grid.arrange(p1, p2, p3, p4)


## -----------------------------------------------------------------------------
knitr::kable(df_cmp)


## -----------------------------------------------------------------------------
plot_freq_2 <- function(xvar, title) {
  out <- test %>% group_by(!!sym(xvar)) %>% summarize(obs = sum(ClaimNb) / sum(Exposure),
                                             glm1 = sum(fitGLM1) / sum(Exposure),
                                             glm2 = sum(fitGLM2) / sum(Exposure),
                                             glm3 = sum(fitGLM3) / sum(Exposure))
  
  ggplot(out, aes(x = !!sym(xvar), group = 1)) + 
    geom_point(aes(y = obs, colour = "observed")) + geom_line(aes(y = obs, colour = "observed"), linetype = "dashed") +
    geom_point(aes(y = glm1, colour = "GLM1")) + geom_line(aes(y = glm1, colour = "GLM1"), linetype = "dashed") +
    geom_point(aes(y = glm2, colour = "GLM2")) + geom_line(aes(y = glm2, colour = "GLM2"), linetype = "dashed") +
    geom_point(aes(y = glm3, colour = "GLM3")) + geom_line(aes(y = glm3, colour = "GLM3"), linetype = "dashed") +
    ylim(0, 0.35) + labs(x = xvar, y = "frequency", title = title) + theme(legend.position = "bottom")
}

# Area
p1 <- plot_freq_2("Area", "frequency by Area")

# VehPower
p2 <- plot_freq_2("VehPower", "frequency by VehPower")

# VehBrand
p3 <- plot_freq_2("VehBrand", "frequency by VehBrand")

# VehAgeGLM
p4 <- plot_freq_2("VehAgeGLM", "frequency by VehAgeGLM")

gridExtra::grid.arrange(p1, p2, p3, p4)


## -----------------------------------------------------------------------------
axis_min <- log(max(test$fitGLM1, test$fitGLM2))
axis_max <- log(min(test$fitGLM1, test$fitGLM2))

ggplot(test, aes(x = log(fitGLM1), y = log(fitGLM2), colour = Exposure)) + geom_point() +
  geom_abline(colour = "#000000", slope = 1, intercept = 0) +
  xlim(axis_max, axis_min) + ylim(axis_max, axis_min) +
  labs(x = "GLM1", y = "GLM2", title = "Claims frequency prediction (log-scale)") +
  scale_colour_gradient(low = "green", high = "red")


## -----------------------------------------------------------------------------
axis_min <- log(max(test$fitGLM1, test$fitGLM3))
axis_max <- log(min(test$fitGLM1, test$fitGLM3))

ggplot(test, aes(x = log(fitGLM1), y = log(fitGLM3), colour = Exposure)) + geom_point() +
  geom_abline(colour = "#000000", slope = 1, intercept = 0) +
  xlim(axis_max, axis_min) + ylim(axis_max, axis_min) +
  labs(x = "GLM1", y = "GLM3", title = "Claims frequency prediction (log-scale)") +
  scale_colour_gradient(low = "green", high = "red")


## -----------------------------------------------------------------------------
sessionInfo()

