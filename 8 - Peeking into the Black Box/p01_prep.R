#===============================================
# Peeking into the Black Box
# Prepare data
# Author: Michael Mayer
# Version from: May 8, 2020
#===============================================

## install CASdatasets
# missing_packages <- setdiff(c("xts", "sp", "lattice"), installed.packages()[, 1])
# if (length(missing_packages)) {
#   install.packages(missing_packages)
# }
# R version 3.6.3
library(CASdatasets)  # 1.0.6
library(dplyr)        # 0.8.5

data(freMTPL2freq)
str(freMTPL2freq)

head(freMTPL2freq, 9)

# Grouping id
distinct <- freMTPL2freq %>% 
  distinct_at(vars(-c(IDpol, Exposure, ClaimNb))) %>% 
  mutate(group_id = row_number())

# Preprocessing
dat <- freMTPL2freq %>% 
  left_join(distinct) %>% 
  mutate(Exposure = pmin(1, Exposure),
         Freq = pmin(15, ClaimNb / Exposure),
         VehPower = pmin(12, VehPower),
         VehAge = pmin(20, VehAge),
         VehGas = factor(VehGas),
         DrivAge = pmin(85, DrivAge),
         logDensity = log(Density),
         VehBrand = factor(VehBrand, levels = 
                             paste0("B", c(12, 1:6, 10, 11, 13, 14))),
         PolicyRegion = relevel(Region, "R24"),
         AreaCode = Area)

table(table(dat[, "group_id"]))
dat[dat$group_id == 283967, ] # 22 times the same row

nrow(dat)

# Covariables, Response, Weight
x <- c("VehPower", "VehAge",  "VehBrand", "VehGas", "DrivAge",
       "logDensity", "PolicyRegion")
y <- "Freq"
w <- "Exposure"
