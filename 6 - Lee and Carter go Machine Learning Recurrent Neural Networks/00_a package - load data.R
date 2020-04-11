#### Purpose: package to load data
#### Authors: Ronald Richman and Mario Wuthrich
#### Data: The data were sourced from the HMD
#### Date: August 12, 2019

options(keras.view_metrics = FALSE)
require(data.table)
require(dplyr)
require(ggplot2)
require(data.table)
require(reshape2)
require(HMDHFDplus)
require(gnm)
require(stringr)
require(ggpubr)

set.seed(1234)

##################################################################
#### Load data
##################################################################

all_mort <- fread(path.data)                      # load data from csv file
all_mort$Gender <- as.factor(all_mort$Gender)

# The data has been downloaded from the Human Mortality Database (HMD).
# We have applied some pre-processing to this data so that Lee-Carter can be applied.
# Values that differ from the HMD have received a flag in the csv file


####################################################################################################################################

