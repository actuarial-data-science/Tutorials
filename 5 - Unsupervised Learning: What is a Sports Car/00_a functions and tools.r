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
library(cluster)
#library(ClusterR)
#library(mclust)

#############################################
### Graphic tools
#############################################


panel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
   par(usr = c(usr[1:2], 0, 1.5) )
    h <- hist(x, plot = FALSE)
    breaks <- h$breaks; nB <- length(breaks)
    y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}

panel.hist.norm <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
    h <- hist(x, plot = FALSE)
    breaks <- h$breaks; nB <- length(breaks)
    y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
    t.delta.breaks<-breaks[-1]-breaks[-nB]
    t.area<-t(t.delta.breaks)%*%y
    t.mean<-mean(x);t.sd<-sd(x)
    t.x.abs.sort<-sort(abs(x))
    t.boolean.0<-(t.x.abs.sort==0)
    t.x.NachKommaStellen<-floor((1-t.boolean.0)*t.x.abs.sort[1]+t.boolean.0*t.x.abs.sort[2])
    t.x.new<-round(x,abs(t.x.NachKommaStellen-1))
    t.range<-range(t.x.new)
    t.x.seq<-seq(t.range[1],t.range[2],length.out=101)
    t.norm<-dnorm(t.x.seq,mean=t.mean,sd=t.sd)
    t.norm<-c(t.norm)*c(t.area)
  lines(t.x.seq,t.norm, col=2,lty=1)
}

panel.qq <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(-3.3,5,-3.3,5) )
    t.col<-"cyan"
    t.mean<-mean(x);t.sd<-sd(x)
    t.x.sort<-sort((x-t.mean)/t.sd)
    t.l<-length(x)
    t.qq<-(1:t.l)/(1+t.l)
    t.qnorm<-qnorm(t.qq,0,1)
  points(t.x.sort,t.qnorm,col=t.col)
  abline(c(0,1),col=t.col,lty=1)
}

panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  #txt <- paste0("Cor=", txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}


#https://stackoverflow.com/questions/36964404/histogram-on-main-diagonal-of-pairs-function-in-r
#https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/pairs.html
