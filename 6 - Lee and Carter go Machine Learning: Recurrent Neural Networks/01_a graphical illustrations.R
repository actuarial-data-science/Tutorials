#### Purpose: Illustrated selected data
#### Authors: Ronald Richman and Mario Wuthrich
#### Date: August 12, 2019


# select parameters
path.data <- "CHE_mort.csv"           # path and name of data file
region <- "CHE"                    # country to be loaded (code is for one selected country)

# load corresponding data
source(file="00_a package - load data.R")
str(all_mort)

##################################################################
#### Heatmap of selected data
##################################################################

gender <- "Male"
#gender <- "Female"

m0 <- c(min(all_mort$logmx), max(all_mort$logmx))
# rows are calendar year t, columns are ages x
logmx <- t(matrix(as.matrix(all_mort[which(all_mort$Gender==gender),"logmx"]), nrow=100, ncol=67))
image(z=logmx, useRaster=TRUE,  zlim=m0, col=rev(rainbow(n=60, start=0, end=.72)), xaxt='n', yaxt='n', main=list(paste("Swiss ",gender, " raw log-mortality rates", sep=""), cex=1.5), cex.lab=1.5, ylab="age x", xlab="calendar year t")
axis(1, at=c(0:(2016-1950))/(2016-1950), c(1950:2016))                   
axis(2, at=c(0:49)/50, labels=c(0:49)*2)                   
lines(x=rep((1999-1950+0.5)/(2016-1950), 2), y=c(0:1), lwd=2)

