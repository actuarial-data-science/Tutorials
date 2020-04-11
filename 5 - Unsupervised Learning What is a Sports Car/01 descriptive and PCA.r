##########################################################
### Authors: Simon Rentzmann and Mario Wüthrich
### Date: August 9, 2019
### Tutorial: Unsupervised learning: What is a Sports Car?
##########################################################

#############################################
### load packages and data
#############################################

source("00_a functions and tools.R")                                                     

d.data.org<-read.table("SportsCars.csv",sep=";",header=TRUE)
d.data<-d.data.org
str(d.data)

#############################################
### transform data for descriptive plots
#############################################

# change to log scale
d.data$W_l    <- log(d.data$weight)
d.data$MP_l   <- log(d.data$max_power)
d.data$CC_l   <- log(d.data$cubic_capacity)
d.data$MT_l   <- log(d.data$max_torque)
d.data$MES_l  <- log(d.data$max_engine_speed)
d.data$S100_l <- log(d.data$seconds_to_100)
d.data$TS_l   <- log(d.data$top_speed)

# data transform according to Ingenbleek-Lemaire (ASTIN Bulletin 1988)
d.data$x1s  <- d.data$W_l-d.data$MP_l
d.data$x2s  <- d.data$MP_l-d.data$CC_l
d.data$x3s  <- d.data$MT_l
d.data$x4s  <- d.data$MES_l
d.data$x5s  <- d.data$CC_l

#############################################
### data illustration
#############################################

# scatter plots with  QQ plots
t.data.streu<-d.data[!is.na(d.data$S100_l),c("W_l","MP_l","CC_l","MT_l","MES_l","S100_l","TS_l")]
pairs(t.data.streu,diag.panel=panel.qq,upper.panel=panel.cor)

t.data.streu<-d.data[,c("x1s","x2s","x3s","x4s","x5s")]
pairs(t.data.streu,diag.panel=panel.qq,upper.panel=panel.cor)

# scatter plots with histogram
t.data.streu<-d.data[,c("x1s","x2s","x3s","x4s","x5s")]
pairs(t.data.streu,diag.panel=panel.hist.norm,upper.panel=panel.cor)

# empirical density plots
t.data.streu <-d.data[,c("x1s","x2s","x3s","x4s","x5s")]
(m0 <- colMeans(t.data.streu))
X01 <- t.data.streu-colMeans(t.data.streu)[col(t.data.streu)]
(sds <- sqrt(colMeans(X01^2))*sqrt(nrow(t.data.streu)/(nrow(t.data.streu)-1)))

i1 <- 1  # should be in 1:5 for xs1 to xs5
position <- c("topleft","topright","topleft","topleft","topleft")
plot(density(t.data.streu[,i1]), col="orange", lwd=2, ylab="empirical density", xlab=paste("x",i1, "s", sep=""), main=list(paste("empirical density variable x", i1, "s", sep=""), cex=1.5), cex.lab=1.5)
lines(density(t.data.streu[,i1])$x, dnorm(density(t.data.streu[,i1])$x, mean=m0[i1], sd=sds[i1]), col="blue", lwd=2, lty=2) 
legend(position[i1], c("empirical density", "Gaussian approximation"), col=c("orange", "blue"), lty=c(1,2), lwd=c(2,2), pch=c(-1,-1))
        

#############################################
### principal components analysis
#############################################

# standardize matrix
X <- X01/sqrt(colMeans(X01^2))[col(X01)]

# eigenvectors and eigenvalues
X1 <- as.matrix(X)
A <-  t(X1) %*% X1
sqrt(eigen(A)$value)      # singular values
eigen(A)$value/nrow(X1)   # scaled eigenvalues

# singular value decomposition
SVD <- svd(X1)
SVD$d                       # singular values
rbind(SVD$v[,1],SVD$v[,2])  # first two right singular vectors

# PCA Sports Cars weights
alpha <- SVD$v[,1]/sds
(alpha_star <- c(alpha[1],alpha[2]-alpha[1], alpha[3], alpha[4], alpha[5]-alpha[2])/alpha[1])

# plot first two principal components
dat3 <- d.data 
dat3$v1 <- X1 %*% SVD$v[,1]
dat3$v2 <- X1 %*% SVD$v[,2]

plot(x=dat3$v1, y=dat3$v2, col="blue",pch=20, ylim=c(-7,7), xlim=c(-7,7), ylab="2nd principal component", xlab="1st principal component", main=list("principal components analysis", cex=1.5), cex.lab=1.5)
dat0 <- dat3[which(dat3$tau<21),]
points(x=dat0$v1, y=dat0$v2, col="green",pch=20)
dat0 <- dat3[which(dat3$tau<17),]
points(x=dat0$v1, y=dat0$v2, col="red",pch=20)
legend("bottomleft", c("tau>=21", "17<=tau<21", "tau<17 (sports car)"), col=c("blue", "green", "red"), lty=c(-1,-1,-1), lwd=c(-1,-1,-1), pch=c(20,20,20))


# reconstruction error
reconstruction.PCA <- array(NA, c(5))

for (p in 1:5){
  Xp <- SVD$v[,1:p] %*% t(SVD$v[,1:p]) %*% t(X)
  Xp <- t(Xp)
  reconstruction.PCA[p] <- sqrt(sum(as.matrix((X-Xp)^2))/nrow(X))
               }
round(reconstruction.PCA,2)               


# PCA with package PCA
t.pca <- princomp(X1,cor=TRUE)
t.pca$loadings          
summary(t.pca)

# scatter plot
switch_sign <- -1           # switch sign of the first component to make svd and princomp compatible
tt.pca <- t.pca$scores
tt.pca[,1] <- switch_sign *tt.pca[,1]
pairs(tt.pca,diag.panel=panel.qq,upper.panel=panel.cor)


# biplot
tt.pca <- t.pca
tt.pca$scores[,1] <-  switch_sign * tt.pca$scores[,1]
tt.pca$loadings[1:5,1] <- switch_sign * tt.pca$loadings[1:5,1] 
biplot(tt.pca,choices=c(1,2),scale=0, expand=2, xlab="1st principal component", ylab="2nd principal component", cex=c(0.4,1.5), ylim=c(-7,7), xlim=c(-7,7))
  