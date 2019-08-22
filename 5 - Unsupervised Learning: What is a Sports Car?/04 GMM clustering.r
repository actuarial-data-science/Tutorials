##########################################################
### Authors: Simon Rentzmann and Mario Wüthrich
### Date: August 9, 2019
### Tutorial: Unsupervised learning: What is a Sports Car?
##########################################################

require(MASS)
library(plyr)
library(stringr)
library(plotrix)
library(matrixStats)
library(ClusterR)
#library(mclust)


##########################################################################
##### load data
##########################################################################

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
#########  K GMM  # only diagonal covariance matrices!
##########################################################################

seed <- 100
set.seed(seed)
K_res <- GMM(X, gaussian_comps=4, dist_mode="eucl_dist", seed_mode="random_subset", em_iter=5, seed=seed)
summary(K_res)
clust <- predict_GMM(X, K_res$centroids, K_res$covariance_matrices, K_res$weights)$cluster_labels

K_res$centroids

##########################################################################
##### principal component analysis versus GMM clustering
##########################################################################

# singular value decomposition
SVD <- svd(as.matrix(X))

pca <- c(1,2)
dat3 <- dat1
dat3$v1 <- as.matrix(X) %*% SVD$v[,pca[1]]
dat3$v2 <- as.matrix(X) %*% SVD$v[,pca[2]]

(kk1 <- K_res$centroids %*% SVD$v[,pca[1]])
(kk2 <- K_res$centroids %*% SVD$v[,pca[2]])

lim0 <- 7

plot(x=dat3$v1, y=dat3$v2, col="orange",pch=20, ylim=c(-lim0,lim0), xlim=c(-lim0,lim0), ylab=paste("principal component ", pca[2], sep=""),xlab=paste("principal component ", pca[1], sep=""),, main=list("GMM(diagonal) vs. PCA", cex=1.5), cex.lab=1.5)
dat0 <- dat3[which(clust==0),]
points(x=dat0$v1, y=dat0$v2, col="red",pch=20)
dat0 <- dat3[which(clust==3),]
points(x=dat0$v1, y=dat0$v2, col="blue",pch=20)
dat0 <- dat3[which(clust==1),]
points(x=dat0$v1, y=dat0$v2, col="magenta",pch=20)
points(x=kk1,y=kk2, col="black",pch=20, cex=2)
legend("bottomleft", c("cluster 1", "cluster 2", "cluster 3", "cluster 4"), col=c("red", "orange", "magenta", "blue"), lty=c(-1,-1,-1,-1), lwd=c(-1,-1,-1,-1), pch=c(20,20,20,20))

