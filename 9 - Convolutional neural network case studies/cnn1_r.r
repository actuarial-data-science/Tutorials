## -----------------------------------------------------------------------------
options(encoding = 'UTF-8')

# Loading all the necessary packages
library("repr")  # not needed in the Rmarkdown version, only for Jupyter notebook
library("abind")
library("pROC")
library("grid")
library("fields")
library("ggplot2")
library("plotly")
library("keras")
library("tensorflow")


## -----------------------------------------------------------------------------
knitr::opts_chunk$set(fig.width = 9, fig.asp = 1)
#options(repr.plot.width=4, repr.plot.height=10)


## -----------------------------------------------------------------------------
pops <- c('AUS','AUT','BEL','BGR','BLR','CAN','CHE','CHL','CZE',
        'DEU','DNK','ESP','EST','FIN','FRA','GBR','GRC','HKG',
        'HRV','HUN','IRL','ISL','ISR','ITA','JPN','KOR','LTU',
        'LUX','LVA','NLD','NOR','NZL','POL','PRT','RUS','SVK',
        'SVN','SWE','TWN','UKR','USA')
nx <- 10 # window size in terms of ages
nt <- 10 # window size in terms of years
sx <- 5 # step width of windows in terms of ages
st <- 5 # step width of windows in terms of years
minAge <- 21
maxAge <- 80
testRatio <- 0.15
validationRatio <- 0.15
thresholdQ <- 0.95 # defines migration/error in terms of a quantile threshold
filterSize <- 5
numberFilters <- 16
filterSize1 <- 3
numberFilters1 <- 16
filterSize2 <- 3
numberFilters2 <- 32
filterSize3 <- 3
numberFilters3 <- 64
numberEpochs <- 800
rxm <- list()
rxf <- list()
X <- list()
Y <- list()
dataRoot <- "../../data"


## -----------------------------------------------------------------------------
qxm <- as.matrix(read.csv(file.path(dataRoot, "cnn1", "GBR_M.txt"), skip = 1, sep = "", header = TRUE))
knitr::kable(head(qxm))

fig <- plot_ly(z = matrix(as.numeric(qxm[, 4]), nrow = 111)[1:110, ]) %>%
        layout(title = 'Mortality rates GBR males', scene = list(
          xaxis = list(title = 'Year'),
          yaxis = list(title = 'Age'),
          zaxis = list(title = 'qx')
        )) %>%
        add_surface()
fig


## -----------------------------------------------------------------------------
E <- as.matrix(read.csv(file.path(dataRoot, "cnn1", "GBR.txt"), skip = 1, sep = "", header = TRUE))
knitr::kable(head(E))

fig <- plot_ly(z = matrix(as.numeric(E[, 4]), nrow = 111)[1:110, ]) %>%
        layout(title = 'Exposures GBR males', scene = list(
          xaxis = list(title = 'Year'),
          yaxis = list(title = 'Age'),
          zaxis = list(title = 'E')
        )) %>%
        add_surface()
fig


## -----------------------------------------------------------------------------
for (jPop in 1:length(pops)) {
    pop = pops[jPop]
    
    lqxm = as.matrix(read.csv(paste0(dataRoot, "/cnn1/logit_qx_", pops[jPop], "_m.csv"), sep = ",", header = FALSE))
    lqxf = as.matrix(read.csv(paste0(dataRoot, "/cnn1/logit_qx_", pops[jPop], "_f.csv"), sep = ",", header = FALSE))
       
    rxm[[pop]] = as.matrix(read.csv(paste0(dataRoot, "/cnn1/residuals_", pops[jPop], "_m.csv"), sep = ",", header = FALSE))
    rxf[[pop]] = as.matrix(read.csv(paste0(dataRoot, "/cnn1/residuals_", pops[jPop], "_f.csv"), sep = ",", header = FALSE))
    
    if (is.element(pop, c('JPN','RUS','USA'))) {
        image.plot(t(rxm[[pop]]))
        mtext(line = 2, side = 1, paste(pop, 'males'))
    }
    
    mx <- floor(floor((maxAge - minAge + 1 - nx) / sx + 1))
    mt <- floor(floor((nrow(rxm[[pop]]) - nt) / st + 1))
    
    X[[pop]] <- array(0, dim = c(mx * mt, nx, nt, 3))
    Y[[pop]] <- array(0, dim = c(mx * mt))
    
    for (j in 0:(mx-1)) {
        for (k in 0:(mt-1)) {                                
            # set up logit qx windows of size nt x nx for each population as input X
            # (population x year buckets x age buckets x sex)
            # logit qx of males as first channel:
            X[[pop]][k*mx+j+1, , , 1] <- lqxm[(k*st+1):(k*st+nt), (j*sx+1):(j*sx+nx)]
            # logit qx of females as second channel:
            X[[pop]][k*mx+j+1, , , 2] <- lqxf[(k*st+1):(k*st+nt), (j*sx+1):(j*sx+nx)]
            # logit qx of females less males as third channel:
            X[[pop]][k*mx+j+1, , , 3] <- X[[pop]][(k*mx+j+1), , , 2] - X[[pop]][k*mx+j+1, , , 1]
            # define output Y as the maximum absolute value of normalized residuals over each window of size nt x nx
            Y[[pop]][k*mx+j+1] <- max(0.5 * abs(
              rxm[[pop]][(k*st+1):(k*st+nt), (j*sx+1):(j*sx+nx)] + rxf[[pop]][(k*st+1):(k*st+nt), (j*sx+1):(j*sx+nx)]
            ))
        }
    }
}


## -----------------------------------------------------------------------------
# normalize X, Y
for (pop in pops) {
    minX1 <- min(X[[pop]][,,,1])
    maxX1 <- max(X[[pop]][,,,1])
    minX2 <- min(X[[pop]][,,,2])
    maxX2 <- max(X[[pop]][,,,2])
    minX3 <- min(X[[pop]][,,,3])
    maxX3 <- max(X[[pop]][,,,3])
    minY <- min(Y[[pop]])
    maxY <- max(Y[[pop]])    
    X[[pop]][,,,1] <- (X[[pop]][,,,1] - minX1) / (maxX1 - minX1)
    X[[pop]][,,,2] <- (X[[pop]][,,,2] - minX2) / (maxX2 - minX2)
    X[[pop]][,,,3] <- (X[[pop]][,,,3] - minX3) / (maxX3 - minX3)
    Y[[pop]] <- (Y[[pop]] - minY) / (maxY - minY)
}
grid.newpage()
grid.raster(X[['GBR']][2,,,], interpolate = FALSE)  # plot as RBG image


## -----------------------------------------------------------------------------
selectedPop <- c('AUS', 'BGR', 'BLR', 'CAN', 'CZE', 'ESP', 'EST', 'FIN',
          'GBR', 'GRC', 'HKG', 'ISL', 'ITA', 'JPN', 'LTU', 'NZL',
          'POL', 'PRT', 'RUS', 'SVK', 'TWN', 'UKR')

allX <- array(numeric(), c(0,10,10,3))
allY <- array(numeric(), c(0))
for (kPop in selectedPop) {
    allX <- abind(allX, X[[kPop]], along = 1)
    allY <- abind(allY, Y[[kPop]], along = 1)
}

set.seed(0)
tf$random$set_seed(0)

testIdx <- runif(length(allY)) < testRatio
testX <- allX[testIdx,,,]
testY <- allY[testIdx]
trainX <- allX[!testIdx,,,]
trainY <- allY[!testIdx]

cnn <- keras_model_sequential() %>% 
  layer_batch_normalization() %>%
  layer_conv_2d(filters = numberFilters, kernel_size = c(filterSize1, filterSize1),
              strides = c(1,1), padding = 'valid', data_format = 'channels_last') %>% 
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_conv_2d(filters = numberFilters, kernel_size = c(filterSize2, filterSize2),
                strides = c(1,1), padding = 'valid', data_format = 'channels_last') %>% 
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_conv_2d(filters = numberFilters, kernel_size = c(filterSize3, filterSize3),
                strides = c(1,1), padding = 'valid', data_format = 'channels_last') %>% 
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_flatten() %>%
  layer_dense(1) %>%
  layer_activation('sigmoid') %>%
  compile(loss = 'mean_squared_error', optimizer = 'sgd')

summary <- cnn %>% fit(
  x = trainX,
  y = trainY,
  epochs = numberEpochs / 4,
  validation_split = validationRatio,
  sample_weight = (0.2 + trainY) / 1.2,
  batch_size = 64,
  verbose = 0
)

plot(summary)

migErr <- testY >= quantile(testY, thresholdQ)
testPred <- predict(cnn, testX)

plot(testPred, testY - testPred[, 1], col = migErr + 5, main = 'Test set of combined populations',
     xlab = 'Prediction P', ylab = 'Residuals Y-P')
plot(testPred, testY, col = migErr + 5, main = 'Test set of combined populations',
     xlab = 'Prediction P', ylab = 'Output Y')

rocobj <- plot.roc(1 * migErr, testPred[, 1], main = "ROC, AUC", ci = TRUE, print.auc = TRUE)
ciobj <- ci.se(rocobj, specificities = seq(0, 1, 0.01))
plot(ciobj, type = "shape")
plot(ci(rocobj, of = "thresholds", thresholds = "best"))
summary(cnn)


## -----------------------------------------------------------------------------
allPred <- predict(cnn, allX)
df <- setNames(data.frame(
        rep(1:(length(allY)/11), each = 11),
        rep(1:11, length(allY)/11),
        allY,
        allPred[, 1],
        allY - allPred[, 1]
      ), c('x','y','z1','z2','z3'))

ggplot(df, aes(y, x, fill = z1)) + geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  ggtitle('Output Y') + xlab('Age buckets') + ylab('Years/countries')

ggplot(df, aes(y, x, fill = z2)) + geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  ggtitle('Prediction P') + xlab('Age buckets') + ylab('Years/countries')

ggplot(df, aes(y, x, fill = z3)) + geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  ggtitle('Residuals Y-P') + xlab('Age buckets') + ylab('Years/countries')

