##########################################
#########  Data Analysis French MTPL
#########  Install packages
#########  Author: Mario Wuthrich
#########  Version March 02, 2020
##########################################

##########################################
#########  install R packages
##########################################

install.packages("CASdatasets", repos = "http://dutangc.free.fr/pub/RRepos/", type="source")
 
require(MASS)
library(CASdatasets)
?CASdatasets
 
data(freMTPL2freq)
str(freMTPL2freq)


