## -----------------------------------------------------------------------------
library(rgdal)
# library(rgeos)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(corrplot)


## -----------------------------------------------------------------------------
# plotting parameters in R Markdown notebook
knitr::opts_chunk$set(fig.width = 9, fig.height = 9)
# plotting parameters in Jupyter notebook
library(repr)  # only needed for Jupyter notebook
options(repr.plot.width = 9, repr.plot.height = 9)


## -----------------------------------------------------------------------------
options(encoding = 'UTF-8')


## -----------------------------------------------------------------------------
summarize <- function(...) suppressMessages(dplyr::summarize(...))


## -----------------------------------------------------------------------------
load_data <- function(file) {
  load(file.path("../0_data/", file), envir = parent.frame(1))
}


## -----------------------------------------------------------------------------
runMultiPlot <- function(dat, VarName) {
  dat <- rename(dat, "VarName" = all_of(VarName))
  out_sum <- dat %>%
    group_by(VarName) %>% 
    summarize(NrObs = length(Exposure),
              Exp = sum(Exposure),
              Nr.Claims = sum(ClaimNb),
              Freq = sum(ClaimNb) / sum(Exposure),
              StDev = sqrt(sum(ClaimNb)) / sum(Exposure))
  # Plot 1
  p1 <- ggplot(out_sum, aes(x = VarName, y = Exp, fill = VarName)) +
    geom_bar(stat = "identity") +
    geom_text(stat = 'identity', aes(label = round(Exp, 0), color = VarName), vjust = -0.5, size = 2.5) +
    labs(x = VarName, y = "Exposure in years", title = "exposure") + theme(legend.position = "none")
  
  # Plot 2
  p2 <- ggplot(out_sum, aes(x = VarName, group = 1)) + geom_point(aes(y = Freq, colour = "observed")) +
    geom_line(aes(y = Freq, colour = "observed"), linetype = "dashed") +
    geom_line(aes(x = as.numeric(VarName), y = pf_freq), color = "red") +
    geom_line(aes(x = as.numeric(VarName), y = Freq + 2 * StDev), color = "red", linetype = "dotted") +
    geom_line(aes(x = as.numeric(VarName), y = Freq - 2 * StDev), color = "red", linetype = "dotted") +
    ylim(0, 0.35) + 
    labs(x = paste(VarName, "groups"), y = "frequency", title = "observed frequency") + theme(legend.position = "none")
  
  # Plot 3
  p3 <- ggplot(out_sum) + geom_bar(stat = "identity", aes(x = VarName, y = Freq, fill = VarName)) +
    geom_line(aes(x = as.numeric(VarName), y = pf_freq), color = "red") + guides(fill = FALSE) +
    labs(x = paste(VarName, "groups"),  y = "frequency", title = "observed frequency") + theme(legend.position = "bottom")
  
  grid.arrange(p1, p2, p3, ncol = 2)
}

plot_2dim_contour <- function(data, VarX, VarY, LabelX, LabelY) {
  data <- rename(data, "VarX" = all_of(VarX), "VarY" = all_of(VarY))
  df_plt <- data %>%
    group_by(VarX, VarY) %>%
    summarize(Exp = sum(Exposure),
              Freq = sum(ClaimNb) / sum(Exposure),
              Pol = n())
  p <- ggplot(df_plt, aes(
    x = as.numeric(VarX),
    y = as.numeric(VarY),
    z = Exp
  )) + geom_contour_filled() + labs(x = LabelX, y = LabelY)
}

plotMap <- function(area_points, Var, label, clow, chigh) {
  area_points <- rename(area_points, "Var" = all_of(Var))
  ggplot(area_points, aes(long, lat, group=group)) +
    ggtitle(paste(label, "by region", sep = " ")) +
    geom_polygon(aes(fill = Var)) +
    scale_fill_gradient(low = clow, high = chigh, name = label) +
    xlab("Longitude") + ylab("Latitude")
}


## -----------------------------------------------------------------------------
load_data("freMTPL2freq.RData")


## -----------------------------------------------------------------------------
str(freMTPL2freq)


## -----------------------------------------------------------------------------
knitr::kable(head(freMTPL2freq))


## -----------------------------------------------------------------------------
dat <- freMTPL2freq %>% 
  mutate(ClaimNb = as.integer(ClaimNb),
         VehAge = pmin(VehAge, 20),
         DrivAge = pmin(DrivAge, 90),
         BonusMalus = round(pmin(BonusMalus, 150) / 10, 0) * 10,
         Density = round(log(Density), 0),
         VehGas = factor(VehGas))


## -----------------------------------------------------------------------------
knitr::kable(head(dat))


## -----------------------------------------------------------------------------
str(dat)


## -----------------------------------------------------------------------------
summary(dat)


## -----------------------------------------------------------------------------
p1 <- ggplot(dat, aes(Exposure)) + geom_histogram()
p2 <- ggplot(dat, aes(x = "Exposure", y = Exposure)) + geom_boxplot() +
      labs(x = "Exposure", y = "frequency", title = "boxplot of exposure")
p3 <- ggplot(dat, aes(ClaimNb)) + geom_histogram() +
      labs(x = "number of claims", y = "frequency", title = "histogram of claims number")
grid.arrange(p1, p2, p3, ncol = 2)


## -----------------------------------------------------------------------------
dat %>% 
  group_by(ClaimNb) %>% 
  summarize(n = n(), Exposure = round(sum(Exposure), 0))


## -----------------------------------------------------------------------------
# calculate portfolio claims frequency
pf_freq <- sum(dat$ClaimNb) / sum(dat$Exposure)

# portfolio claims frequency (homogeneous estimator)
sprintf("Portfolio claim frequency:  %s", round(pf_freq, 4))


## -----------------------------------------------------------------------------
runMultiPlot(dat, VarName = "Area")


## -----------------------------------------------------------------------------
runMultiPlot(dat, VarName = "VehPower")


## -----------------------------------------------------------------------------
runMultiPlot(dat, VarName = "VehAge")


## -----------------------------------------------------------------------------
runMultiPlot(dat, VarName = "DrivAge")


## -----------------------------------------------------------------------------
runMultiPlot(dat, VarName = "BonusMalus")


## -----------------------------------------------------------------------------
runMultiPlot(dat, VarName = "VehBrand")


## -----------------------------------------------------------------------------
runMultiPlot(dat, VarName = "VehGas")


## -----------------------------------------------------------------------------
runMultiPlot(dat, VarName = "Density")


## -----------------------------------------------------------------------------
runMultiPlot(dat, VarName = "Region")


## -----------------------------------------------------------------------------
p1 <- plot_2dim_contour(dat, "Area", "BonusMalus", "area group",  "bonus-malus group")
p2 <- plot_2dim_contour(dat, "VehPower", "DrivAge", "vehicle power group",  "driver age group")
p3 <- plot_2dim_contour(dat, "VehPower", "BonusMalus", "vehicle power group",  "bonus-malus group")
p4 <- plot_2dim_contour(dat, "VehAge", "DrivAge", "vehicle age group",  "driver age group")

grid.arrange(p1, p2, p3, p4, ncol = 2)


## -----------------------------------------------------------------------------
df_cor <- dat %>% 
  select(Area, VehPower, VehAge, DrivAge, BonusMalus, Density) 
df_cor$Area <- as.numeric(df_cor$Area)
df_cor$VehPower <- as.numeric(df_cor$VehPower)


## -----------------------------------------------------------------------------
M <- round(cor(df_cor, method = "pearson"), 2)
knitr::kable(M)
corrplot(M, method = "color")


## -----------------------------------------------------------------------------
M <- round(cor(df_cor, method = "spearman"), 2)
knitr::kable(M)
corrplot(M, method = "color")


## -----------------------------------------------------------------------------
reg_sum <- dat %>% 
  group_by(Region) %>% 
  mutate(VehGas = factor(VehGas)) %>% 
  mutate_at(c("Area", "VehPower", "VehGas"), as.numeric) %>% 
  summarize(NrObs = length(Exposure),
            Exp = sum(Exposure),
            Freq = sum(ClaimNb) / sum(Exposure),
            Area = mean(Area),
            VehPower = mean(VehPower),
            VehAge = mean(VehAge),
            DrivAge = mean(DrivAge),
            BonusMalus = mean(BonusMalus),
            VehGas = mean(VehGas),
            Density = mean(Density))

knitr::kable(head(reg_sum, n = 10))


## -----------------------------------------------------------------------------
# Downloaded shapefiles from http://www.diva-gis.org/gData and extracted all the files from the zip file.
area <- rgdal::readOGR(file.path("../../data/shapefiles", "FRA_adm2.shp"))


## -----------------------------------------------------------------------------
reg_sum$id <- sapply(reg_sum$Region, substr, 2, 3)
area_points <- fortify(area, region = "ID_1")  # convert to data.frame


## -----------------------------------------------------------------------------
area_points$id <- recode(
  area_points$id,
  "1"="42","2"="72","3"="83","4"="11","5"="25","6"="26","7"="53","8"="24","9"="21",
  "10"="94","11"="43","12"="23","13"="91","14"="74","15"="41","16"="73","17"="31",
  "18"="52","19"="22","20"="54","21"="93","22"="82"
)


## -----------------------------------------------------------------------------
area_points <- merge(
  area_points,
  reg_sum[, c("id","Exp","Freq","Area","VehPower","VehAge","DrivAge","BonusMalus","VehGas","Density")],
  by.x = "id",
  by.y = "id",
  all.x = TRUE
)
area_points <- area_points[order(area_points$order), ]  # Has to be ordered correctly to plot.


## -----------------------------------------------------------------------------
plotMap(area_points, "Exp", "Exposure", "blue", "red")


## -----------------------------------------------------------------------------
plotMap(area_points, "Freq", "Observed frequencies", "green", "red")


## -----------------------------------------------------------------------------
plotMap(area_points, "VehGas", "Average diesel ratio", "green", "red")


## -----------------------------------------------------------------------------
plotMap(area_points, "VehAge", "average vehicle age", "green", "red")


## -----------------------------------------------------------------------------
sessionInfo()

