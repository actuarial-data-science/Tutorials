#===============================================
# Peeking into the Black Box
# Modeling
# Author: Michael Mayer
# Version from: May 8, 2020
#===============================================

set.seed(1)

library(MetricsWeighted)  # 0.5.0
library(flashlight)       # 0.7.2
library(ggplot2)          # 3.3.0

fillc <- "#E69F00"

#===============================================
# Setting up explainers
#===============================================

fl_glm <- flashlight(
  model = fit_glm, label = "GLM", 
  predict_function = function(fit, X) predict(fit, X, type = "response")
)

fl_nn <- flashlight(
  model = fit_nn, label = "NNet", 
  predict_function = function(fit, X) 
    predict(fit, prep_nn_calib(X, x), type = "response")
)

fl_xgb <- flashlight(
  model = fit_xgb, label = "XGBoost", 
  predict_function = function(fit, X) predict(fit, prep_xgb(X, x))
)

# Combine them and add common elements like reference data
metrics <- list(`Average deviance` = deviance_poisson, 
                `Relative deviance reduction` = r_squared_poisson)
fls <- multiflashlight(list(fl_glm, fl_nn, fl_xgb), data = test, 
                       y = y, w = w, metrics = metrics)

# Version on canonical scale
fls_log <- multiflashlight(fls, linkinv = log)

# ===============================================
# Performance
# ===============================================

perf <- light_performance(fls)
perf
plot(perf, geom = "point") +
  labs(x = element_blank(), y = element_blank())

# ===============================================
# Importance
# ===============================================

imp <- light_importance(fls, v = x)
plot(imp, fill = fillc, color = "black")

# ===============================================
# Effects
# ===============================================

# ICE (uncentered and centered, without log and with)
plot(light_ice(fls, v = "DrivAge", n_max = 200, seed = 3), alpha = 0.1)
plot(light_ice(fls, v = "DrivAge", n_max = 200, seed = 3, 
               center = "middle"), alpha = 0.03)
plot(light_ice(fls_log, v = "DrivAge", n_max = 200, seed = 3), alpha = 0.1)
plot(light_ice(fls_log, v = "DrivAge", n_max = 200, seed = 3, 
               center = "middle"), alpha = 0.03)

# Partial dependence curves
plot(light_profile(fls, v = "VehAge", pd_evaluate_at = 0:20))
plot(light_profile(fls, v = "DrivAge", n_bins = 25))
plot(light_profile(fls, v = "logDensity"))
plot(light_profile(fls, v = "VehGas"))

# ALE versus partial dependence
ale_DrivAge <- light_effects(fls, v = "DrivAge", counts_weighted = TRUE,
                             v_labels = FALSE, n_bins = 20, cut_type = "quantile")
plot(ale_DrivAge, use = c("pd", "ale"), show_points = FALSE)

# Classic diagnostic plots
plot(light_profile(fls, v = "VehAge", type = "predicted"))
plot(light_profile(fls, v = "VehAge", type = "residual")) +
  geom_hline(yintercept = 0)
plot(light_profile(fls, v = "VehAge", type = "response"))

# Multiple aspects combined
eff_DrivAge <- light_effects(fls, v = "DrivAge", counts_weighted = TRUE)
p <- plot(eff_DrivAge, show_points = FALSE)
plot_counts(p, eff_DrivAge, alpha = 0.3)

# ===============================================
# Interactions
# ===============================================

# Interaction (relative)
interact_rel <- light_interaction(
  fls_log, 
  v = most_important(imp, 4), 
  take_sqrt = FALSE,
  pairwise = TRUE, 
  use_linkinv = TRUE,
  seed = 61
)
plot(interact_rel, color = "black", fill = fillc, rotate_x = TRUE)

# Interaction (absolute)
interact_abs <- light_interaction(
  fls_log, 
  v = most_important(imp, 4), 
  normalize = FALSE,
  pairwise = TRUE, 
  use_linkinv = TRUE,
  seed = 61
)
plot(interact_abs, color = "black", fill = fillc, rotate_x = TRUE)

# Filter on largest three brands
sub_data <- test %>% 
  filter(VehBrand %in% c("B1", "B2", "B12"))

# Strong interaction
pdp_vehAge_Brand <- light_profile(fls_log, v = "VehAge", by = "VehBrand", 
                                  pd_seed = 50, data = sub_data)
plot(pdp_vehAge_Brand)

# Weak interaction
pdp_DrivAge_Gas <- light_profile(fls_log, v = "DrivAge", 
                                 by = "VehGas", pd_seed = 50)
plot(pdp_DrivAge_Gas)

# ===============================================
# Global surrogate tree
# ===============================================

# In order to make the trees easy visible on the screen,
# we plot them separately (unlike in the paper)
surr_nn <- light_global_surrogate(fls_log$NNet, v = x)
plot(surr_nn)

surr_xgb <- light_global_surrogate(fls_log$XGBoost, v = x)
plot(surr_xgb)

# ===============================================
# Individual predictions
# ===============================================

new_obs <- test[1, ]
new_obs[, x]
unlist(predict(fls, data = new_obs))

# Breakdown
bd <- light_breakdown(fls$XGBoost, new_obs = new_obs, 
                      v = x, n_max = 1000, seed = 20)
plot(bd)

# Extract same order of variables for visualization only
v <- setdiff(bd$data$variable, c("baseline", "prediction"))

# Approximate SHAP
shap <- light_breakdown(fls$XGBoost, new_obs, 
                        visit_strategy = "permutation",
                        v = v, n_max = 1000, seed = 20)
plot(shap)

# ===============================================
# Derive global model properties from local
# ===============================================

fl_with_shap <- add_shap(fls$XGBoost, v = x, n_shap = 500, 
                         n_perm = 12, n_max = 1000, seed = 100)
# saveRDS(fl_with_shap, file = "fl_with_shap.rds")

plot(light_importance(fl_with_shap, v = x, type = "shap"), 
     fill = fillc, color = "black")
plot(light_scatter(fl_with_shap, v = "DrivAge", type = "shap"), alpha = 0.3)

# ===============================================
# Improve GLM
# ===============================================

fit_glm2 <- glm(Freq ~ VehPower + VehBrand * VehGas + PolicyRegion + 
                  ns(DrivAge, 5) + VehBrand * ns(VehAge, 5) + 
                  ns(logDensity, 5), 
                data = train, 
                family = quasipoisson(), 
                weights = train[[w]])

# Setting up expainers
fl_glm2 <- flashlight(
  model = fit_glm2, label = "Improved GLM", 
  predict_function = function(fit, X) predict(fit, X, type = "response")
)

# Combine them and add common elements like reference data
fls2 <- multiflashlight(list(fl_glm, fl_glm2, fl_nn, fl_xgb), 
                        metrics = metrics, data = test, y = y, w = w)
fls2_log <- multiflashlight(fls2, linkinv = log)

# Some results
plot(light_performance(fls2), geom = "point", rotate_x = TRUE)
plot(light_importance(fls2, v = x), fill = fillc, color = "black", top_m = 4)
plot(light_profile(fls2, v = "logDensity"))
interact_rel_improved <- light_interaction(
  fls2_log, v = most_important(imp, 4), take_sqrt = FALSE,
  pairwise = TRUE,  use_linkinv = TRUE, seed = 61)
plot(interact_rel_improved, color = "black", fill = fillc, top_m = 4)

