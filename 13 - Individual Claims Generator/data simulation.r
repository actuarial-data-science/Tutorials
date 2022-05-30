#####################################################################
#### Authors Melantha Wang & Mario Wuthrich
#### Date 28/05/2022  
#### Generation of individual claim data
#####################################################################

## This simulator is based on the following R package:
## https://cran.r-project.org/web/packages/SynthETIC/index.html
## Vignette:
## https://cran.r-project.org/web/packages/SynthETIC/vignettes/SynthETIC-demo.html

#####################################################################
#### load packages
#####################################################################

library(SynthETIC)  
library(plyr)
library(locfit)
library(dplyr)

ref_claim <- 1          # currency is 1
time_unit <- 1/12       # we consider monthly claims development
set_parameters(ref_claim=ref_claim, time_unit=time_unit)

years <- 10               # number of occurrence years
I <- years/time_unit      # number of development periods

source("./Tools/functions simulation.R")
dest_path <- "" # plot to be saved in ...

#####################################################################
#### generate claim data
#####################################################################

# generate individual claims: may take 60 seconds
claims_list <- data.generation(seed=1000, future_info=TRUE)

# save individual output for easier access
claims <- claims_list[[1]]
paid <- claims_list[[2]]

## is only returned if future_info=TRUE
#full_claims <- claims_list[[3]]
#full_paid <- claims_list[[4]]
reopen <- claims_list[[5]]

# uncomment to save RData files
PathData <- "./Data/"
#save(claims, file=paste(PathData, "claims.rda", sep=""))
#save(paid, file=paste(PathData, "paid.rda", sep=""))
#load(file=paste(PathData, "claims.rda", sep=""))
#load(file=paste(PathData, "paid.rda", sep=""))

str(claims)
str(paid)
 
#####################################################################
#### claims count and reporting
#####################################################################

source("./Tools/functions plotting.R")

# get reported claims
rep_claims <- claims %>% 
  dplyr::filter(!is.na(RepDate)) %>%
  dplyr::mutate(
    AccWeek = ceiling((as.integer(difftime(AccDate, as.Date("2011-12-31"), units="days"))-.5)/7),
    RepWeek = ceiling((as.integer(difftime(RepDate, as.Date("2011-12-31"), units="days"))-.5)/7))

# plot accident dates vs reporting delays
save.yes <- 1
save.yes <- 0
for (type in c(1:6)) {
  plot_triangle(
    rep_claims, type, x_lim=200, save.yes=save.yes, 
    path=paste(dest_path, "AccRep_", type, ".pdf", sep=""))   
}

# plot claim counts per claim type
plot_data <- rep_claims %>%
  dplyr::group_by(Type, AccMonth) %>%
  dplyr::summarise(count = n(), .groups = "drop") %>%
  dplyr::ungroup()
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  # open canvas
  pdf(file=paste(dest_path, "ClaimCounts.pdf", sep=""))
}
col_type <- rainbow(n=6, start=0, end=.75)
with(
  plot_data %>% filter(Type == 1),
  plot(
    x=AccMonth, y=count, ylim=range(plot_data$count),
    type='l', main=list("claim counts per claim type", cex=1.5), col=col_type[1], 
    xlab="accident date (in monthly units)", ylab="claim counts", cex.lab=1.5))
polygon(x=c(120-36, 120, 120, 120-36), y=c(0,0,250,250), border=FALSE, col=adjustcolor("gray", alpha.f=0.3)) 
for (jj in 1:6){
  with(plot_data %>% filter(Type == jj),
       lines(x=AccMonth, y=count, col=col_type[jj]))
}
abline(v=c(0:10)*12, lty=3, col="darkgray") # add gridlines
legend(x="topleft", col=col_type, lwd=rep(6,1), legend=paste("claim type ", c(1:6), sep="")) # add legend
if (save.yes==1){
  # close graphic device
  dev.off()
}


#####################################################################
#### ultimate claims sizes (this information is usually not available for all claims)
#####################################################################
# we take all claims here, i.e., the following plots are usually not
# available in a typical claims reserving situation

range(claims$Ultimate)

# claim size density plot
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "Density.pdf", sep=""))
}
plot(
  density(claims$Ultimate, from=0, to=50000), 
  col="blue", lwd=2, 
  main=list("empirical density of ultimates", cex=1.5), cex.lab=1.5, 
  ylab="empirical density", xlab="ultimate claim sizes")
if (save.yes==1) {
  dev.off()
}

# log-log plot of ultimate claim size
logp_data <- claims %>% filter(Ultimate > 0) # exclude zero-claims for logged plots
pp <- ecdf(logp_data$Ultimate)
set.seed(100)
ll <- sample(x=c(1:nrow(logp_data)), size=2000)
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "LogLog.jpeg", sep=""))
}
plot(
  x=log(logp_data[ll,]$Ultimate), y=log(1-pp(logp_data[ll,]$Ultimate)), 
  main=list("log-log plot of ultimate claims", cex=1.5), cex.lab=1.5, 
  xlab="logged ultimate claim sizes", ylab="logged survival probability")
if (save.yes==1) {
  dev.off()
}

# plot of ultimate claims per claim type
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "UltimatesType.pdf", sep=""))
}
boxplot(
  Ultimate ~ Type, data = claims, col=col_type, 
  main=list("ultimate claims per claim type", cex=1.5), 
  ylab="ultimate claim sizes", xlab="claim type", cex.lab=1.5)
if (save.yes==1) {
  dev.off()
}

# plot of ultimate claims per claim type (log-scale)
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "UltimatesTypeLog.pdf", sep=""))
}
boxplot(
  log(Ultimate) ~ Type, data = logp_data, col=col_type, 
  main=list("ultimate claims per claim type (log-scale)", cex=1.5), 
  ylab="logged ultimate claim sizes", xlab="claim type", cex.lab=1.5)
if (save.yes==1) {
  dev.off()
}

# summary table of claim size by claim type
claims %>%
  filter(Ultimate > 0) %>%
  dplyr::group_by(Type) %>%
  dplyr::summarise(
    mean_ultimate = mean(Ultimate),
    sd_ultimate = sd(Ultimate),
    min_ultimate = min(Ultimate),
    max_ultimate = max(Ultimate)) %>%
  round()

save.yes <- 1
save.yes <- 0
# plot of (age, type) interaction
# refer to "functions plotting.R" for plotting code
for (type in c(1:6)) {
  UltimatePerClaimType(filter(claims, Ultimate > 0), type, save.yes, paste(dest_path, "Age_Type", type, ".pdf", sep=""))
}

# log ultimate vs accident date (in months) for claim type 5
type <- 5
UltimatePerClaimAccMonth(filter(claims, Ultimate > 0), type, save.yes, paste(dest_path, "AccMonth_Type", type, ".pdf", sep=""))

# log ultimate vs weekday for claim type 6
type <- 6
UltimatePerClaimAccWeekday(filter(claims, Ultimate > 0), type, save.yes, paste(dest_path, "AccWeekday_Type", type, ".pdf", sep=""))



#####################################################################
#### analyzing reporting delays
#####################################################################
# we only consider reported claims, which typically induces a bias

# density of reporting delays
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "ReportingDelay.pdf", sep=""))
}
plot(
  density(rep_claims$RepDelDays, from=0), col="blue", lwd=2, 
  main=list("empirical density of reporting delay", cex=1.5), 
  ylab="empirical density", xlab="reporting delay (in days)", cex.lab=1.5)
if (save.yes==1) {
  dev.off()
}

# boxplot of reporting delays per claim type
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "ReportingDelayClaimType.pdf", sep=""))
}
boxplot(
  RepDelDays ~ Type, data=rep_claims, col=col_type, 
  ylab="reporting delay (in days)", xlab="claim type", cex.lab=1.5, 
  main=list("reporting delays per claim type", cex=1.5))
if (save.yes==1) {
  dev.off()
}

# reporting delay vs binned claim size
rep_claims$UltiClass <- pmin(round_any(rep_claims$Ultimate, 1000, f = ceiling), 100000)
col_claimsize <- rainbow(n=length(unique(rep_claims$UltiClass)), start=0, end=.5)
save.yes <- 1
save.yes <- 0
if (save.yes==1){
  pdf(file=paste(dest_path, "ReportingDelayClaimSize1.pdf", sep=""))
}
boxplot(
  RepDelDays ~ UltiClass, data=rep_claims, col=col_claimsize, 
  ylab="reporting delay (in days)", xlab="ultimate claim size", cex.lab=1.5, 
  main=list("reporting delays per ultimate claim size", cex=1.5))
if (save.yes==1) {
  dev.off()
}

# reporting delay vs claim size (line graph)
plot_data <- rep_claims %>%
  dplyr::group_by(UltiClass) %>%
  dplyr::summarise(
    mean_RepDel = mean(RepDelDays),
    sd_RepDel = sd(RepDelDays))
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "ReportingDelayClaimSize2.pdf", sep=""))
}
with(
  plot_data, {
    plot(x=UltiClass, y=mean_RepDel, ylim=range(mean_RepDel + sd_RepDel, mean_RepDel - sd_RepDel, 0), type='l', ylab="reporting delay (in days)", xlab="ultimate claim size", cex.lab=1.5, main=list("reporting delays per ultimate claim size", cex=1.5))
    lines(x=UltiClass, y=mean_RepDel+sd_RepDel, col="darkgray")
    lines(x=UltiClass, y=mean_RepDel-sd_RepDel, col="darkgray")
    lines(x=UltiClass, y=predict(locfit(mean_RepDel ~ UltiClass, alpha=.5, deg=2), newdata = UltiClass), col="orange", lwd=2) 
    abline(h=mean(mean_RepDel), lty=3)
    legend(x="topright", cex=1.25,  lty=rep(1,3), lwd=c(1,1,2), col=c("black", "darkgray", "orange"), legend=c("empirical mean", "1 std.dev.", "spline fit"))
  }
)
if (save.yes==1) {
  dev.off()
}


#####################################################################
#### analyzing settlement delays
#####################################################################

closed_claims <- claims %>% filter(!is.na(SetMonth))

# density plot of settlement delay
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "SettlementDelay.pdf", sep=""))
}
plot(
  density(closed_claims$SetDelMonths, from=0), col="blue", lwd=2, 
  main=list("empirical density of settlement delay", cex=1.5), 
  xlab="settelement delay (monthly units)", ylab="empirical density", cex.lab=1.5)
if (save.yes==1) {
  dev.off()
}

# boxplots of settlement delays per claim type
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "SettlementDelayClaimType.pdf", sep=""))
}
boxplot(
  SetDelMonths ~ Type, data=closed_claims, col=col_type, 
  ylab="settlement delay (in months)", xlab="claim type", cex.lab=1.5, 
  main=list("settlement delays per claim type", cex=1.5))
if (save.yes==1) {
  dev.off()
}

# boxplots of settlement delay vs binned claim size
closed_claims$UltiClass <- pmin(round_any(closed_claims$Ultimate, 1000, f = ceiling), 100000)
col_claimsize <- rainbow(n=length(unique(closed_claims$UltiClass)), start=0, end=.5)
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "SettlementDelayClaimSize1.pdf", sep=""))
}
boxplot(
  SetDelMonths ~ UltiClass, data=closed_claims, col=col_claimsize, 
  ylab="settlement delay (in months)", xlab="ultimate claim size", cex.lab=1.5, 
  main=list("settlement delays per ultimate claim size", cex=1.5))
if (save.yes==1) {
  dev.off()
}

# settlement delay vs claim size (line graph)
plot_data <- closed_claims %>%
  dplyr::group_by(UltiClass) %>%
  dplyr::summarise(
    mean_SetDel = mean(SetDelMonths),
    sd_SetDel = sd(SetDelMonths))
save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path, "SettlementDelayClaimSize2.pdf", sep=""))
}
with(
  plot_data, {
    plot(x=UltiClass, y=mean_SetDel, ylim=range(mean_SetDel + sd_SetDel, mean_SetDel - sd_SetDel, 0), type='l', ylab="settlement delay (in months)", xlab="ultimate claim size", cex.lab=1.5, main=list("settlement delays per ultimate claim size", cex=1.5))
    lines(x=UltiClass, y=mean_SetDel+sd_SetDel, col="darkgray")
    lines(x=UltiClass, y=mean_SetDel-sd_SetDel, col="darkgray")
    lines(x=UltiClass, y=predict(locfit(mean_SetDel ~ UltiClass, alpha=.5, deg=2), newdata = UltiClass), col="orange", lwd=2) 
    abline(h=mean(mean_SetDel), lty=3)
    legend(x="topright", cex=1.25,  lty=rep(1,3), lwd=c(1,1,2), col=c("black", "darkgray", "orange"), legend=c("empirical mean", "1 std.dev.", "spline fit"))
  }
)
if (save.yes==1) {
  dev.off()
}



#####################################################################
#### number of payments
#####################################################################

pay_count <- claims %>%
  dplyr::mutate(PayCount_new = pmin(PayCount, 3)) %>%
  dplyr::group_by(PayCount_new) %>%
  dplyr::summarise(count = n()) %>%
  dplyr::mutate(prop = count / sum(count))
pay_count


#####################################################################
#### reopenings
#####################################################################

claims_by_type <- claims %>%
  dplyr::group_by(Type) %>%
  dplyr::summarise(count = n())
plot_data <- reopen %>%
  # add claim type to the re-opening data
  merge(dplyr::select(claims, Id, Type), by = "Id") %>%
  dplyr::mutate(mths_after_close = EventMonth - SetMonth) %>%
  dplyr::group_by(Type, mths_after_close) %>%
  dplyr::summarise(reopen_claims = n(), .groups = "drop") %>%
  merge(claims_by_type, by = "Type") %>%
  dplyr::mutate(reopen_prop = reopen_claims / count)

save.yes <- 1
save.yes <- 0
if (save.yes==1) {
  pdf(file=paste(dest_path,"Reopening.pdf", sep=""))
}
col_type <- rainbow(n=6, start=0, end=.75)
with(
  plot_data %>% filter(Type == 1),
  plot(
    x=mths_after_close, y=reopen_prop,
    type='l', main=list("re-openings per claim type", cex=1.5), col=col_type[1], 
    xlab="months after first claim closure", ylab="proportion of re-openings", cex.lab=1.5,
    ylim = c(0, max(plot_data$reopen_prop) + 0.001)
  ) 
)
for (jj in 1:6){
  with(plot_data %>% filter(Type == jj),
    points(x=mths_after_close, y=reopen_prop, col=col_type[jj]))
  with(plot_data %>% filter(Type == jj),
       lines(x=mths_after_close, y=reopen_prop, col=col_type[jj]))
}
abline(v=c(0:12), lty=3, col="darkgray")
legend(x="topright", col=col_type, lwd=rep(6,1), legend=paste("claim type ", c(1:6), sep=""))
if (save.yes==1) {
  dev.off()
}


#####################################################################
#### chain ladder method
#####################################################################

### triangles
library(reshape2)
library(ChainLadder)

# payment triangle
payments <- merge(paid, claims, by="Id")

# project to annual grid
payments$AccYear <- ceiling((payments$AccMonth-1/2)/12)
payments$RepYear <- ceiling((payments$RepMonth-1/2)/12)
payments$PayYear <- ceiling((payments$EventMonth-1/2)/12)
payments$PayDelay <- payments$PayYear - payments$AccYear

# construct upper paid triangle (past)
paid_triangle <- dcast(payments, AccYear ~ PayDelay, sum, value.var="Paid")
paid_triangle <- paid_triangle[, -1] # remove (duplicate) AccYear column
paid_triangle <- paid_triangle / 1000 # give numbers in thousands
J <- ncol(paid_triangle)

# true payments (lower triangle)
True <- claims %>%
  dplyr::mutate(AccYear = ceiling((AccMonth-1/2)/12)) %>%
  dplyr::group_by(AccYear) %>%
  dplyr::summarise(CC = sum(Ultimate) / 1000) # divide by 1,000 to get numbers in thousands

# convert increment to cumulative paid
for (jj in 2:ncol(paid_triangle)){
  paid_triangle[, jj] <- paid_triangle[, jj] + paid_triangle[, jj-1]
  paid_triangle[(ncol(paid_triangle) - jj + 2):ncol(paid_triangle), jj] <- NA
}

# apply Mack chain-ladder
units1 <- 1
M <- MackChainLadder(paid_triangle/units1, est.sigma="Mack")
tt <- round(cbind(M$FullTriangle[, J], True$CC, M$FullTriangle[, J] - True$CC, M$Mack.S.E[, J]))
tt <- cbind(tt, round(100 * abs(tt[,3]/tt[,4]), 1)) # % error
tt <- data.frame(
  rbind(
    tt, 
    c(colSums(tt[,1:3]), round(M$Total.Mack.S.E), round(100 * sum(tt[,3])/M$Total.Mack.S.E, 1)))
)
names(tt) <- c("ChainLadder", "True", "Difference", "RMSEP", "%")
tt



