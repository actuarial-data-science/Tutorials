##########################################################
### Authors: Simon Rentzmann and Mario Wüthrich
### Date: August 9, 2019
### Tutorial: Unsupervised learning: What is a Sports Car?
##########################################################

source("00_a functions and tools.R")

#####################################
##### load data
#####################################

dat1 <- read.table(file="SportsCars.csv", header=TRUE, sep=";")
str(dat1)
dat2 <- dat1   
dat2$x1 <- log(dat2$weight/dat2$max_power)
dat2$x2 <- log(dat2$max_power/dat2$cubic_capacity)
dat2$x3 <- log(dat2$max_torque)
dat2$x4 <- log(dat2$max_engine_speed)
dat2$x5 <- log(dat2$cubic_capacity)
dat2 <- dat2[, c("x1","x2","x3","x4","x5")]

# normalization of design matrix
X01 <- dat2-colMeans(dat2)[col(dat2)]
X <- X01/sqrt(colMeans(X01^2))[col(X01)]

##########################################################################
#########  K-means
##########################################################################

# initialize
Kaverage <- colMeans(X)
K0 <- 10
TWCD <- array(NA, c(K0))  # total within-cluster dissimilarity
Classifier <- array(1, c(K0, nrow(X)))
(TWCD[1] <- sum(colSums(as.matrix(X^2))))

# run K-means algorithm
set.seed(100)
for (K in 2:K0){ 
   if (K==2){(K_res <- kmeans(X,K) )}
   if (K>2){(K_res  <- kmeans(X,K_centers) )}
   TWCD[K] <- sum(K_res$withins)
   Classifier[K,] <- K_res$cluster
   K_centers <- array(NA, c(K+1, ncol(X)))
   K_centers[K+1,] <- Kaverage
   K_centers[1:K,] <- K_res$centers 
                }

# plot losses                
xtitle <- "decrease in total within-cluster dissimilarity "
plot(x=c(1:K0), y=TWCD, ylim=c(0, max(TWCD)), main=list(xtitle, cex=1.5), col="blue", cex=1.5, pch=20, ylab="total within-cluster dissimilarity", xlab="hyperparameter K", cex.lab=1.5)
lines(x=c(1:K0), y=TWCD, col="blue", lty=3)


##########################################################################
##### plot K-means versus principal component analysis
##########################################################################

# singular value decomposition
SVD <- svd(as.matrix(X))
pca <- c(1,2)
dat3 <- dat1
dat3$v1 <- as.matrix(X) %*% SVD$v[,pca[1]]
dat3$v2 <- as.matrix(X) %*% SVD$v[,pca[2]]

lim0 <- 7

plot(x=dat3$v1, y=dat3$v2, col="orange",pch=20, ylim=c(-lim0,lim0), xlim=c(-lim0,lim0), ylab=paste("principal component ", pca[2], sep=""),xlab=paste("principal component ", pca[1], sep=""),, main=list("K-means vs. PCA", cex=1.5), cex.lab=1.5)
dat0 <- dat3[which(Classifier[4,]==4),]
points(x=dat0$v1, y=dat0$v2, col="blue",pch=20)
dat0 <- dat3[which(Classifier[4,]==1),]
points(x=dat0$v1, y=dat0$v2, col="red",pch=20)
dat0 <- dat3[which(Classifier[4,]==3),]
points(x=dat0$v1, y=dat0$v2, col="magenta",pch=20)
legend("bottomleft", c("cluster 1", "cluster 2", "cluster 3", "cluster 4"), col=c("red", "orange", "magenta", "blue"), lty=c(-1,-1,-1,-1), lwd=c(-1,-1,-1,-1), pch=c(20,20,20,20))


##########################################################################
#########  K-medoids
##########################################################################

set.seed(100)
(K_res <- pam(X, k=4, metric="manhattan", diss=FALSE))

# plot K-medoids versus PCA
plot(x=dat3$v1, y=dat3$v2, col="orange",pch=20, ylim=c(-lim0,lim0), xlim=c(-lim0,lim0), ylab=paste("principal component ", pca[2], sep=""),xlab=paste("principal component ", pca[1], sep=""),, main=list("K-medoids vs. PCA", cex=1.5), cex.lab=1.5)
dat0 <- dat3[which(K_res$cluster==4),]
points(x=dat0$v1, y=dat0$v2, col="red",pch=20)
dat0 <- dat3[which(K_res$cluster==3),]
points(x=dat0$v1, y=dat0$v2, col="blue",pch=20)
dat0 <- dat3[which(K_res$cluster==2),]
points(x=dat0$v1, y=dat0$v2, col="magenta",pch=20)
points(x=dat3[K_res$id.med,"v1"],y=dat3[K_res$id.med,"v2"], col="black",pch=20, cex=2)
legend("bottomleft", c("cluster 1", "cluster 2", "cluster 3", "cluster 4"), col=c("red", "orange", "magenta", "blue"), lty=c(-1,-1,-1,-1), lwd=c(-1,-1,-1,-1), pch=c(20,20,20,20))


