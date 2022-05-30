#####################################################################
#### Authors Melantha Wang & Mario Wuthrich
#### Date 28/05/2022  
#### Auxiliary functions for descriptive analysis                                       
#####################################################################


#####################################################################
#### claim counts plotting
#####################################################################
# Plot accident weeks against reporting weeks of all claims
plot_triangle <- function(portfolio, claim_type, x_lim, save.yes, path) {
  
  # get data: claim count triangle (long format data)
  claim_count_tri <- portfolio %>%
    dplyr::filter(Type == claim_type) %>%
    dplyr::group_by(AccWeek, RepWeek) %>%
    dplyr::summarise(count = n(), .groups = "drop")
  
  # plot
  if (save.yes == 1) {
    pdf(file=path)
  }
  y_lim = max(portfolio$AccWeek)
  with(
    claim_count_tri,
    plot(
      x=RepWeek-AccWeek, y=y_lim-AccWeek, cex=0.8, cex.lab=1.5, pch=1, yaxt='n', 
      xlim=c(0, x_lim), ylim=c(0, y_lim), 
      xlab="reporting delay (in weeks)", ylab="accident date", 
      main=list(paste("claims reporting of Type ", claim_type, sep=""), cex=1.5)))
  
  # colour overlapping points
  # colour blue if no less than 2 claims, green if no less than 4
  with(
    claim_count_tri %>% dplyr::filter(count > 2),
    points(x=RepWeek-AccWeek, y=y_lim-AccWeek, cex=0.8, pch=19, col="blue"))
  with(
    claim_count_tri %>% dplyr::filter(count > 4),
    points(x=RepWeek-AccWeek, y=y_lim-AccWeek, cex=0.8, pch=19, col="green"))
  
  # label y-axis
  axis(2, at=c(0:10)*365.25/7, label=rev(c(2012:2022)), cex=1.5, las=2)
  
  # draw a diagonal line
  abline(a=0, b=1, col="orange", lwd=2)  
  if (save.yes==1) {
    dev.off()
  }
}


#####################################################################
#### ultimate claims
#####################################################################

# plot of log ultimate vs age, for a given claim type
UltimatePerClaimType <- function(data, type, save.yes, path) {
  col <- rainbow(n=length(unique(data$Age)), start=0, end=.5)
  if (save.yes == 1) {
    pdf(file=path)
  }
  
  filtered_data = filter(data, Type == type)
  # boxplot of log ultimate vs age, for a given claim type
  boxplot(
    log(Ultimate) ~ Age, data=filtered_data, col=col, 
    ylab="logged ultimate claim sizes", xlab="age of injured", cex.lab=1.5, 
    main=paste("ultimates claim type ", type, sep=""))
  # plot average log claim size as a horizontal line
  abline(h=mean(log(filtered_data$Ultimate)), col="magenta")
  
  if (save.yes==1) {
    dev.off()
  }
}

# log ultimate vs accident date (in months), for a given claim type
UltimatePerClaimAccMonth <- function(data, type, save.yes, path) {
  col <- rainbow(n=12, start=0, end=.5)
  if (save.yes==1) {
    pdf(file=path)
  }
  
  filtered_data = filter(data, Type == type)
  # boxplot of log ultimate vs accident month
  boxplot(
    log(Ultimate) ~ AccMonth, data=filtered_data, col=col, 
    ylab="logged ultimate claim sizes", xlab="accident date (monthly units)", 
    main=paste("ultimates of claim type ", type, sep=""), cex.lab=1.5)
  # plot average log claim size as a horizontal line
  abline(h=mean(log(filtered_data$Ultimate)), col="magenta")
  # plot vertical gridlines
  abline(v=c(1:9)*12)
  
  if (save.yes==1){
    dev.off()
  }
}

# log ultimate vs accident weekday, for a given claim type
UltimatePerClaimAccWeekday <- function(data, type, save.yes, path) {
  col <- rainbow(n=7, start=0, end=.5)
  if (save.yes==1) {
    pdf(file=path)
  }
  
  filtered_data = filter(data, Type == type)
  # boxplot of log ultimate vs accident weekday
  boxplot(
    log(Ultimate) ~ AccWeekday, data=filtered_data, col=col, 
    ylab="logged ultimate claim sizes", xlab="accident weekday", 
    main=paste("ultimates of claim type ", type, sep=""), cex.lab=1.5)
  # plot average log claim size as a horizontal line
  abline(h=mean(log(filtered_data$Ultimate)), col="magenta")
  
  if (save.yes==1) {
    dev.off()
  }
}
  
    