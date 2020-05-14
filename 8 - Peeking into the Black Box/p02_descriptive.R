#===============================================
# Peeking into the Black Box
# Descriptive analysis
# Author: Michael Mayer
# Version from: May 8, 2020
#===============================================

library(dplyr)    # 0.8.5
library(forcats)  # 0.5.0
library(reshape2) # 1.4.3
library(corrplot) # 0.84
library(ggplot2)  # 3.3.0

str(dat)
head(dat[, c(x, w, y)])
summary(dat[, c(x, w, y)])

# Univariate description
melted <- dat[c("Freq", "Exposure", "DrivAge", "VehAge", "VehPower", "logDensity")] %>% 
  stack() %>% 
  filter(ind != "Freq" | values > 0) %>% 
  mutate(ind = fct_recode(ind, 
                          `Driver's age` = "DrivAge", 
                          `Vehicle's age` = "VehAge", 
                          `Vehicle power` = "VehPower", 
                          `Logarithmic density` = "logDensity"))

ggplot(melted, aes(x=values)) +
  geom_histogram(bins = 19, fill = "#E69F00") +
  facet_wrap(~ind, scales = "free") +
  labs(x = element_blank(), y = element_blank()) +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

# Bivariate description
cor_mat <- dat %>% 
  select_at(c(x, "BonusMalus")) %>% 
  select_if(is.numeric) %>% 
  cor() %>% 
  round(2)
corrplot(cor_mat, method = "square", type = "lower", diag = FALSE, title = "",
         addCoef.col = "black", tl.col = "black")

# Boxplots
th <- theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

# BonusMalus nach DrivAge
dat %>% 
  mutate(DrivAge = cut(DrivAge, c(17:24, seq(25, 85, 10)), 
                       labels = c(18:25, "26-35", "36-45", "46-55", "56-65", "66-75", "76+"),
                       include.lowest = TRUE),
         DrivAge = fct_recode(DrivAge)) %>% 
ggplot(aes(x = DrivAge, y = BonusMalus)) +
  geom_boxplot(outlier.shape = NA, fill = "#E69F00") +
  coord_cartesian(ylim = c(50, 125))

# Brand/vehicle age
dat %>% 
  ggplot(aes(x = VehBrand, y = VehAge)) +
  geom_boxplot(outlier.shape = NA, fill = "#E69F00") +
  th

# Density/Area
dat %>% 
  ggplot(aes(x = AreaCode, y = logDensity)) +
  geom_boxplot(fill = "#E69F00") +
  th

# Density/Region
dat %>% 
  ggplot(aes(x = Region, y = logDensity)) +
  geom_boxplot(outlier.shape = NA, fill = "#E69F00") +
  th
