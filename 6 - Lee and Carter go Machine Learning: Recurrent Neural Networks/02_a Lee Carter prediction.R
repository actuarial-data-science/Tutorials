#### Purpose: Fit the Lee-Carter model to one population
#### Authors: Ronald Richman and Mario Wuthrich
#### Date: August 12, 2019


# select parameters
path.data <- "CHE_mort.csv"           # path and name of data file
region <- "CHE"                    # country to be loaded (code is for one selected country)

# load corresponding data
source(file="00_a package - load data.R")
str(all_mort)



##################################################################
#### Fit Lee Carter to the selected country and extrapolate with Random Walk with Drift
##################################################################

ObsYear <- 1999
gender <- "Female"
train <- all_mort[Year<=ObsYear][Gender == gender]
min(train$Year)
    
### fit via SVD
train[,ax:= mean(logmx), by = (Age)]
train[,mx_adj:= logmx-ax]  
rates_mat <- as.matrix(train %>% dcast.data.table(Age~Year, value.var = "mx_adj", sum))[,-1]
svd_fit <- svd(rates_mat)
    
ax <- train[,unique(ax)]
bx <- svd_fit$u[,1]*svd_fit$d[1]
kt <- svd_fit$v[,1]
      
c1 <- mean(kt)
c2 <- sum(bx)
ax <- ax+c1*bx
bx <- bx/c2
kt <- (kt-c1)*c2
    
### extrapolation and forecast
vali  <- all_mort[Year>ObsYear][Gender == gender]    
t_forecast <- vali[,unique(Year)] %>% length()
forecast_kt  =kt %>% forecast::rwf(t_forecast, drift = T)
kt_forecast = forecast_kt$mean
 

##################################################################
#### Illustrate and back-test Lee-Carter prediction
##################################################################

# illustration selected drift
plot_data <- c(kt, kt_forecast)
plot(plot_data, pch=20, col="red", cex=2, cex.lab=1.5, xaxt='n', ylab="values k_t", xlab="calendar year t", main=list(paste("estimated process k_t for ",gender, sep=""), cex=1.5)) 
points(kt, col="blue", pch=20, cex=2)
axis(1, at=c(1:length(plot_data)), labels=c(1:length(plot_data))+1949)                   
abline(v=(length(kt)+0.5), lwd=2)

    
# in-sample and out-of-sample analysis    
fitted = (ax+(bx)%*%t(kt)) %>% melt
train$pred_LC_svd = fitted$value %>% exp
fitted_vali = (ax+(bx)%*%t(kt_forecast)) %>% melt
vali$pred_LC_svd =   fitted_vali$value %>% exp
round(c((mean((train$mx-train$pred_LC_svd)^2)*10^4) , (mean((vali$mx-vali$pred_LC_svd)^2)*10^4)),4)

