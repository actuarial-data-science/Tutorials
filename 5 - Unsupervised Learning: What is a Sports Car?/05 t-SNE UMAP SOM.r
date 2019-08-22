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
library(tsne)
library(umap)
library(kohonen)

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
#########  t SNE
##########################################################################

seed <- 100
set.seed(seed)
# it takes roughly 50 seconds
# KL divergence 0.427991566501587
{t1 <- proc.time()
   (K_res <- tsne(X, k=2, initial_dim=ncol(X), perplexity=30))
proc.time()-t1}


i1 <- 2
i2 <- 1
sign0 <- 1

K_res1 <- K_res[,c(i1,i2)]
K_res1 <- sign0 * K_res1
plot(K_res1, pch=20, col="blue", cex.lab=1.5, ylab=paste("component ", i2, paste=""), xlab=paste("component ", i1, paste=""), main=list(paste("t-SNE with seed ", seed, sep=""), cex=1.5))
dat0 <- K_res1[which(dat1$tau<21),]
points(dat0, col="green",pch=20)
dat0 <- K_res1[which(dat1$tau<17),]
points(dat0, col="red",pch=20)
legend("bottomleft", c("tau>=21", "17<=tau<21", "tau<17 (sports car)"), col=c("blue", "green", "red"), lty=c(-1,-1,-1), lwd=c(-1,-1,-1), pch=c(20,20,20))


##########################################################################
#########  UMAP
##########################################################################

kNeighbors <- 15     # default is 15
seed <- 100
set.seed(seed)
min_dist <- .1

umap.param <- umap.defaults
umap.param$n_components <- 2
umap.param$n_neighbors <- kNeighbors  
umap.param$random_state <- seed 
umap.param$min_dist <- min_dist 

{t1 <- proc.time()
   (K_res <- umap(X, config=umap.param, method="naive"))
proc.time()-t1}

i1 <- 2
i2 <- 1
sign0 <- 1

K_res1 <- K_res$layout[,c(i1,i2)]
K_res1 <- sign0 * K_res1
plot(K_res1, pch=20, col="blue", cex.lab=1.5, ylab=paste("component ", i2, paste=""), xlab=paste("component ", i1, paste=""), main=list(paste("UMAP (k=", kNeighbors, " NN and min_dist=", min_dist,")", sep=""), cex=1.5))
dat0 <- K_res1[which(dat1$tau<21),]
points(dat0, col="green",pch=20)
dat0 <- K_res1[which(dat1$tau<17),]
points(dat0, col="red",pch=20)
legend("bottomright", c("tau>=21", "17<=tau<21", "tau<17 (sports car)"), col=c("blue", "green", "red"), lty=c(-1,-1,-1), lwd=c(-1,-1,-1), pch=c(20,20,20))

##########################################################################
#########  Kohonen map
##########################################################################

n1 <- 2
n2 <- 10

set.seed(100)
{t1 <- proc.time()
 som.X <- som(as.matrix(X), grid = somgrid(xdim=n1, ydim=n2, topo="rectangular"), 
                            rlen= 100, dist.fcts="sumofsquares")
proc.time()-t1}


summary(som.X)

plot(som.X,c("changes"), main=list("training progress", cex=1.5), col="blue", cex.lab=1.5)            # training progress

plot(som.X,c("counts"), main="allocation counts to neurons", cex.lab=1.5)

dat1$tau2 <- dat1$sports_car+as.integer(dat1$tau<21)+1
plot(som.X,c("mapping"), classif=predict(som.X), col=c("blue","green","red")[dat1$tau2], pch=19, main="allocation of cases to neurons")


