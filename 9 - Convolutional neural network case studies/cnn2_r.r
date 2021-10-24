## -----------------------------------------------------------------------------
options(encoding = 'UTF-8')

# Loading all the necessary packages
library("repr")  # not needed in the Rmarkdown version, only for Jupyter notebook
library("ggplot2")
library("keras")
library("tensorflow")
library("OpenImageR")


## -----------------------------------------------------------------------------
knitr::opts_chunk$set(fig.width = 9, fig.height = 7)
#options(repr.plot.width=4, repr.plot.height=10)


## -----------------------------------------------------------------------------
validationRatio <- 0.15
filterSize1     <- 3
numberFilters1  <- 10
filterSize2     <- 3
numberFilters2  <- 20
filterSize3     <- 3
numberFilters3  <- 40
numberEpochs    <- 10

dataRoot <- "../../data"


## -----------------------------------------------------------------------------
load_image_file <- function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

load_label_file <- function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

trainX <- load_image_file(file.path(dataRoot, "cnn2", "train-images.idx3-ubyte"))
testX  <- load_image_file(file.path(dataRoot, "cnn2", "t10k-images.idx3-ubyte"))

train_Y <- as.factor(load_label_file(file.path(dataRoot, "cnn2", "train-labels.idx1-ubyte")))
test_Y  <- as.factor(load_label_file(file.path(dataRoot, "cnn2", "t10k-labels.idx1-ubyte")))

trainX <- array_reshape(data.matrix(trainX) / 255, c(dim(trainX)[1], 28, 28, 1))
testX <- array_reshape(data.matrix(testX) / 255, c(dim(testX)[1], 28, 28, 1))
trainY <- to_categorical(train_Y, 10)
testY <- to_categorical(test_Y, 10)

par(mfrow = c(2, 4))
for (j in 1:8) {
    image(aperm(trainX[j, 28:1, , 1], c(2, 1)), col = gray(12:1 / 12))
    title(train_Y[j])
}


## -----------------------------------------------------------------------------
set.seed(0)
tf$random$set_seed(0)

cnn <- keras_model_sequential() %>%
  layer_conv_2d(filters = numberFilters1, kernel_size = c(filterSize1, filterSize1),
                strides = c(1,1), padding = 'valid', input_shape = c(28, 28, 1)) %>%
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(2,2), padding = 'valid') %>%
  
  layer_conv_2d(filters = numberFilters2, kernel_size = c(filterSize2, filterSize2),
                strides = c(1,1), padding = 'valid') %>%
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(1,1), padding = 'valid') %>%
  
  layer_conv_2d(filters = numberFilters3, kernel_size = c(filterSize3, filterSize3),
                strides = c(1,1), padding = 'valid') %>%
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(2,2)) %>%
  
  layer_flatten() %>%
  layer_dense(10) %>%
  layer_activation('softmax', name = 'softmax') %>%
  compile(loss = loss_categorical_crossentropy, optimizer = optimizer_adadelta(), metrics = c('accuracy'))

# RSc: below took ~22 minutes with 1CPU / 8GB / 40 epochs
summary <- cnn %>% fit(
  x = trainX,
  y = trainY,
  epochs = numberEpochs,
  validation_split = validationRatio,
  batch_size = 64,
  verbose = 1
)
summary(cnn)


## -----------------------------------------------------------------------------
plot(summary)
print(summary)

#testP <- cnn %>% predict_classes(testX)  # This is deprecated in keras/tf 2.6. In our case, below is applicable instead.
testP <- cnn %>% predict(testX) %>% k_argmax()
testP <- as.array(testP)

confusion_matrix <- as.data.frame(table(testP, test_Y))

ggplot(data = confusion_matrix, aes(x = testP, y = test_Y)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "blue", trans = "log")


## -----------------------------------------------------------------------------
incorrectIdx <- which(test_Y != testP)
par(mfrow = c(2, 4))
for (j in 1:8) {
    image(aperm(testX[incorrectIdx[j], 28:1, , 1], c(2,1)), col = gray(12:1 / 12))
    title(paste0('A: ', test_Y[incorrectIdx[j]], ', P:', testP[incorrectIdx[j]]))
}

print(paste(length(incorrectIdx), "incorrectly classified digits (out of 10'000 digits)"))


## -----------------------------------------------------------------------------
layerModel <- keras_model(input = cnn$input, outputs = get_layer(cnn, 'softmax')$output)  # the softmax activation layer
img <- trainX[19, 28:1, , ]
par(mfrow = c(2, 4))
for (j in seq(0, 315, 45)) {
    image(aperm(rotateImage(img, j), c(2,1)), col = gray(12:1 / 12))
    title(j)
}

activationSoftMax <- matrix(0, 360, 10)
for (j in 1:360) {
    imgRotated <- img
    imgRotated <- rotateImage(img, j)[28:1, ]
    activationSoftMax[j, ] <- layerModel %>% predict(array_reshape(imgRotated, c(1, 28, 28, 1)))
}

par(mfrow = c(1, 1))
plot(1:360, activationSoftMax[, 7], type = "l", col = "blue", xlab = "Rotation angle", ylab = "Output of softmax layer")
lines(1:360, activationSoftMax[, 10], col = "red")
lines(1:360, activationSoftMax[, 9], col = "orange")
lines(1:360, activationSoftMax[, 6], col = "magenta")
legend("topright", legend = c("7", "10", "9", "6"), fill = c("blue", "red", "orange", "magenta"))


## -----------------------------------------------------------------------------
activationSoftMax <- array(0, c(121, 10, 18))
par(mfrow = c(3, 6))
for (i in 1:18) {
    img <- trainX[i, , , ]
    for (j in 1:121) {
        shiftRows <- j %% 11 - 5
        shiftCols <- floor(j / 11) - 5
        if (shiftRows != 0 && shiftCols != 0)
          imgShifted <- translation(img, shift_rows = shiftRows, shift_cols = shiftCols)
        else
          imgShifted <- img
        activationSoftMax[j, , i] <- layerModel %>% predict(array_reshape(imgShifted, c(1, 28, 28, 1)))
        if (j == 1) {
            lowerRight <- imgShifted
        }
    }
    image(aperm(lowerRight[28:1, ], c(2, 1)), col = gray(12:1 / 12))
}
par(mfrow = c(3, 6))
for (i in 1:18) {
    image(array_reshape(activationSoftMax[, as.numeric(train_Y[i]), i], c(11, 11)), col = gray(12:1 / 12))
}


## -----------------------------------------------------------------------------
activationSoftMax <- array(0, c(121, 10, 18))
par(mfrow = c(3, 6))
for (i in 1:18) {
    img <- trainX[i, , , ]
    for (j in 1:121) {
        imgZoomed <- cropImage(
          resizeImage(img, height = round(28*((j%%11)/20+1)), width = round(28*((floor(j/11))/20+1)), method = "bilinear"),
          new_height = 1:28,
          new_width = 1:28,
          type = "user_defined"
        )
        activationSoftMax[j, , i] <- layerModel %>% predict(array_reshape(imgZoomed, c(1, 28, 28, 1)))
        if (j == 48) {
            selectedImgZoom <- imgZoomed
        }
    }
    image(aperm(selectedImgZoom[28:1, ], c(2, 1)), col = gray(12:1 / 12))
}
par(mfrow = c(3, 6))
for (i in 1:18) {
    image(array_reshape(activationSoftMax[, as.numeric(train_Y[i]), i], c(11, 11)), col = gray(12:1 / 12))
}

