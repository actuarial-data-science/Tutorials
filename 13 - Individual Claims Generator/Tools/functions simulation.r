#####################################################################
#### Authors Melantha Wang & Mario Wuthrich
#### Date 28/05/2022  
#### Auxiliary functions for individual claim generation of different types                                       
#####################################################################

## This simulator is based on the following R package:
## https://cran.r-project.org/web/packages/SynthETIC/index.html
## Vignette:
## https://cran.r-project.org/web/packages/SynthETIC/vignettes/SynthETIC-demo.html
## The following code is mainly taken from this vignette.

#####################################################################
#### auxiliary functions: mainly taken from SynthETIC Vignette
#####################################################################

# helper functions
# define sampling functions for ultimates of different claim types
claims.type1 <- function(n_vector, age) {
  mu <- exp(8 + 0.0175*(age-18) + 0.0005*(age-30)^2)
  cc <- ceiling(rgamma(sum(n_vector), shape=1.25, rate=1.25/mu))
  to_SynthETIC(cc, n_vector)
}

claims.type2 <- function(n) {
  s <- actuar::rinvgauss(n, mean = 50000, dispersion = 1.5e-5)
  while (any(s < 30 | s > 1000000)) {
    for (j in which(s < 30 | s > 1000000)) {
      # for rejected values, resample
      s[j] <- actuar::rinvgauss(1, mean = 25000, dispersion = 1e-4)
    }
  }
  return(s) 
}

claims.type3 <- function(n_vector, age) {
  mu <- (9 - 0.02*(age-18))  
  cc <- pmin(exp(rgamma(sum(n_vector), shape=50, rate=50/mu)), 1000000)
  to_SynthETIC(cc, n_vector)
}

claims.type4 <- function(n_vector, age) {
  mu <- exp(8 - 0.6 * sin((age-18)/(65-18)*2*pi)) # cyclic in age
  cc <- rgamma(sum(n_vector), shape=1.2, rate=1.2/mu)
  to_SynthETIC(cc, n_vector)
}

claims.type5 <- function(n_vector, AccDate) {
  mu <- exp(9 - sin(AccDate/12*2*pi - pi/2)) # cyclic in accident date
  cc <- rgamma(sum(n_vector), shape=.5, rate = .5/mu) + 500
  to_SynthETIC(cc, n_vector)
}

claims.type6A <- function(n) {
  rgamma(n, shape=.75, rate = .75/3000) + 50
}
claims.type6B <- function(n) {
  rgamma(n, shape=.75, rate = .75/6000) + 50
}


# define reporting delay parameters
# taken from vignette with changed parameters
notidel_param <- function(claim_size, occurrence_period) {
  target_mean <- pmax(0.01, 1-log(claim_size/ref_claim)/15) / time_unit
  target_cv <- 1.5
  shape <- 1 / sqrt(target_cv)
  scale <- target_mean / shape
  c(shape = shape, scale = scale)
}


# define settlement delay parameters
# taken from vignette with slightly changed parameters
setldel_param <- function(claim_size, occurrence_period) {
  if (claim_size < (1000 * ref_claim) & occurrence_period >= 21) {
    a <- pmin(0.85, 0.65 + 0.02 * (occurrence_period - 21))
  } else {
    a <- pmax(0.85, 1 - 0.0075 * occurrence_period)
  }
  mean_quarter <- a * pmin(25, pmax(1, 6 + 4*log(claim_size/(1000 * ref_claim))))
  target_mean <- mean_quarter / 8 / time_unit
  target_cv <- 0.60
  c(shape = get_Weibull_parameters(target_mean, target_cv)[1, ],
    scale = get_Weibull_parameters(target_mean, target_cv)[2, ])
}

# define sampling function for number of payments
# taken from vignette with slightly changed parameters
rmixed_payment_no <- function(n, claim_size, claim_size_benchmark_1, claim_size_benchmark_2) {
  test_1 <- (claim_size_benchmark_1 < claim_size & claim_size <= claim_size_benchmark_2)
  test_2 <- (claim_size > claim_size_benchmark_2)
  no_pmt <- sample(c(1, 2), size = n, replace = T, prob = c(5/6, 1/6))
  no_pmt[test_1] <- sample(c(1, 2, 3), size = sum(test_1), replace = T, prob = c(1/2, 1/4, 1/4))
  no_pmt_mean <- pmin(8, 4 + log(claim_size/claim_size_benchmark_2))
  prob <- 1 / (no_pmt_mean - 3)
  no_pmt[test_2] <- stats::rgeom(n = sum(test_2), prob = prob[test_2]) + 2
  no_pmt
}

# define sampling function for individual payment sizes of multiple payments
# taken from vignette
rmixed_payment_size <- function(n, claim_size) {
  if (n >= 4) {
    p_mean <- 1 - pmin(0.95, 0.75 + 0.04*log(claim_size/(1000 * ref_claim)))
    p_CV <- 0.20
    p_parameters <- get_Beta_parameters(target_mean = p_mean, target_cv = p_CV)
    last_two_pmts_complement <- stats::rbeta(
      1, shape1 = p_parameters[1], shape2 = p_parameters[2])
    last_two_pmts <- 1 - last_two_pmts_complement
    q_mean <- 0.9
    q_CV <- 0.03
    q_parameters <- get_Beta_parameters(target_mean = q_mean, target_cv = q_CV)
    q <- stats::rbeta(1, shape1 = q_parameters[1], shape2 = q_parameters[2])
    p_second_last <- q * last_two_pmts
    p_last <- (1-q) * last_two_pmts
    p_unnorm_mean <- last_two_pmts_complement/(n - 2)
    p_unnorm_CV <- 0.10
    p_unnorm_parameters <- get_Beta_parameters(
      target_mean = p_unnorm_mean, target_cv = p_unnorm_CV)
    amt <- stats::rbeta(
      n - 2, shape1 = p_unnorm_parameters[1], shape2 = p_unnorm_parameters[2])
    amt <- last_two_pmts_complement * (amt/sum(amt))
    amt <- append(amt, c(p_second_last, p_last))
    amt <- claim_size * amt
    
  } else if (n == 2 | n == 3) {
    p_unnorm_mean <- 1/n
    p_unnorm_CV <- 0.10
    p_unnorm_parameters <- get_Beta_parameters(
      target_mean = p_unnorm_mean, target_cv = p_unnorm_CV)
    amt <- stats::rbeta(
      n, shape1 = p_unnorm_parameters[1], shape2 = p_unnorm_parameters[2])
    amt <- claim_size * amt/sum(amt)

  } else {
    amt <- claim_size
  }
  return(amt)
}

# define sampling function for payment times of multiple payments
# taken from vignette
r_pmtdel <- function(n, claim_size, setldel, setldel_mean) {
  result <- c(rep(NA, n))
  if (n >= 4) {
    unnorm_d_mean <- (1 / 4) / time_unit
    unnorm_d_cv <- 0.20
    parameters <- get_Weibull_parameters(target_mean = unnorm_d_mean, target_cv = unnorm_d_cv)
    result[n] <- stats::rweibull(1, shape = parameters[1], scale = parameters[2])
    for (i in 1:(n - 1)) {
      unnorm_d_mean <- setldel_mean / n
      unnorm_d_cv <- 0.35
      parameters <- get_Weibull_parameters(target_mean = unnorm_d_mean, target_cv = unnorm_d_cv)
      result[i] <- stats::rweibull(1, shape = parameters[1], scale = parameters[2])
    }
  } else {
    for (i in 1:n) {
      unnorm_d_mean <- setldel_mean / n
      unnorm_d_cv <- 0.35
      parameters <- get_Weibull_parameters(target_mean = unnorm_d_mean, target_cv = unnorm_d_cv)
      result[i] <- stats::rweibull(1, shape = parameters[1], scale = parameters[2])
    }
  }
  result[1:n-1] <- (setldel/sum(result)) * result[1:n-1]
  result[n] <- setldel - sum(result[1:n-1])
  return(result)
}

param_pmtdel <- function(claim_size, setldel, occurrence_period) {
  if (claim_size < (0.10 * ref_claim) & occurrence_period >= 61) {
    a <- pmin(0.85, 0.65 + 0.02 * (occurrence_period - 61))
  } else {
    a <- pmax(0.85, 1 - 0.0075 * occurrence_period)
  }
  mean_quarter <- a * pmin(25, pmax(1, 6 + 4*log(claim_size/(1000 * ref_claim))))
  target_mean <- mean_quarter / 4 / time_unit
  c(claim_size = claim_size,
    setldel = setldel,
    setldel_mean = target_mean)
}

# define base inflation
# taken from vignette
demo_rate <- (1 + 0.02)^(1/4) - 1
base_inflation_past <- rep(demo_rate, times = 40)
base_inflation_future <- rep(demo_rate, times = 40)
base_inflation_vector <- c(base_inflation_past, base_inflation_future)



#####################################################################
#### generating claim data
#####################################################################
data.generation.type <- function(type, exposure, seed){
  set.seed(seed)                         
  
  ### choice of monthly exposures and frequencies
  ### exposure is annualized
  if (type==1) {
    E <- c(rep(exposure, I))
    lambda <- c(rep(.05, I))
  }
  if (type==2) {
    E <- c(rep(exposure, I))
    lambda <- c(rep(.01, I))}
  if (type==3) {
    E <- exposure + c(1:I) * exposure / I
    lambda <- c(rep(.02, I))}
  if (type==4) {
    E <- exposure - c(1:I) * exposure/ (2*I)
    lambda <- c(rep(.05, I))}
  if (type==5) {
    E <- c(rep(exposure, I))
    lambda <- .01 + (sin(c(1:I)/12*2*pi) + 1)/2 * 0.06}
  if (type==6) {
    E <- exposure + c(1:I) * exposure / (2*I)
    lambda <- c(rep(.05, I))}
  
  ### simulation of numbers of claims and accident dates   
  n_vector <- claim_frequency(I=I, E=E, freq=lambda)
  acc_time <- claim_occurrence(frequency_vector=n_vector)
  
  ### set up data frame: Id, Type, Age, AccDate, AccMonth, AccWeekday
  n <- sum(n_vector)
  get_weekday <- function(Date) {
    days_since_start <- 1 + as.integer(difftime(Date, as.Date("2011-12-31"), units="days"))
    i <- (days_since_start - 1) %% 7 + 1
    # return
    c("Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri")[i]
  }
  get_weekday <- Vectorize(get_weekday)
  
  claims_ <- data.frame(Id = c(1:n), Type = type) %>% dplyr::mutate(
    Age = sample(size=n, x=c(18:65), replace=TRUE),
    AccDate = as.Date(ceiling(unlist(acc_time) * 365.25 / 12), origin = "2011-12-31"),
    AccMonth = 12 * (as.integer(substr(AccDate, 1, 4)) - 2012) + as.integer(substr(AccDate, 6, 7)),
    AccWeekday = factor(get_weekday(AccDate), levels=c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
  )
  
  ### simulation of ultimate claim sizes
  ### choosing an age dependent regression structure
  if (type==1) {Ultimate <- claims.type1(n_vector, claims_$Age)}
  if (type==2) {Ultimate <- claim_size(frequency_vector = n_vector, simfun = claims.type2)}
  if (type==3) {Ultimate <- claims.type3(n_vector, claims_$Age)}
  if (type==4) {Ultimate <- claims.type4(n_vector, claims_$Age)}
  if (type==5) {Ultimate <- claims.type5(n_vector, unlist(acc_time))}
  if (type==6) {
     UltimateA <- unlist(claim_size(frequency_vector = n_vector, simfun = claims.type6A))
     UltimateB <- unlist(claim_size(frequency_vector = n_vector, simfun = claims.type6B))
     # replace weekend values by type 6B
     UltimateA[which(claims_$AccWeekday %in% c("Sat", "Sun"))] <- UltimateB[which(claims_$AccWeekday %in% c("Sat", "Sun"))]
     Ultimate <- to_SynthETIC(UltimateA, n_vector)
  }
  
  ### reporting delay
  RepDel <- claim_notification(n_vector, Ultimate, rfun = rgamma, paramfun = notidel_param)
  
  ### settlement delay
  SetDel <- claim_closure(n_vector, Ultimate, rfun = rweibull, paramfun = setldel_param)
  
  ### number of payments
  PayCount <- claim_payment_no(n_vector, Ultimate, rfun = rmixed_payment_no, claim_size_benchmark_1 = 3000 * ref_claim, claim_size_benchmark_2 = 10000 * ref_claim)
  
  ### payment sizes of individual payments
  Paid <- claim_payment_size(n_vector, Ultimate, PayCount, rfun = rmixed_payment_size)
  
  ### payment times of individual payments
  PayDel <- claim_payment_delay(n_vector, Ultimate, PayCount, SetDel, rfun = r_pmtdel, paramfun = param_pmtdel, occurrence_period = rep(1:I, times = n_vector))
  PayTimes <- claim_payment_time(n_vector, acc_time, RepDel, PayDel)
  
  ### collecting (newly simulated) claims information
  claims_info <- generate_claim_dataset(
    frequency_vector = n_vector, occurrence_list = acc_time, 
    claim_size_list = Ultimate, notification_list = RepDel, 
    settlement_list = SetDel, no_payments_list = PayCount
  ) %>% dplyr::mutate(
    RepDate = as.Date(ceiling((occurrence_time + notidel) * 365.25 / 12), origin = "2011-12-31"),
    RepMonth = 12 * (as.integer(substr(RepDate, 1, 4)) - 2012) + as.integer(substr(RepDate, 6, 7)),
    RepDelDays = as.integer(difftime(RepDate, claims_$AccDate, units="days")),
    SetDate = as.Date(ceiling((occurrence_time + notidel + setldel) * 365.25 / 12), origin = "2011-12-31"),
    SetMonth = 12 * (as.integer(substr(SetDate, 1, 4)) - 2012) + as.integer(substr(SetDate, 6, 7)),
    SetDelMonths = SetMonth - RepMonth
  ) %>% dplyr::rename(
    PayCount = no_payment
  )
  
  ### inflation adjustment
  Inflation <- claim_payment_inflation(n_vector, Paid, PayTimes, acc_time, Ultimate, base_inflation_vector)
  
  ### collecting the individual payments
  ClaimsPaid <- claims(
    frequency_vector = n_vector, occurrence_list = acc_time, 
    claim_size_list = Ultimate, notification_list = RepDel, 
    settlement_list = SetDel, no_payments_list = PayCount, 
    payment_size_list = Paid, payment_delay_list = PayDel, 
    payment_time_list = PayTimes, payment_inflated_list = Inflation)
  
  ClaimsPaid <- generate_transaction_dataset(ClaimsPaid, adjust = FALSE) %>%
    dplyr::select(claim_no, pmt_no, payment_time, payment_inflated) %>%
    dplyr::rename(
      # rename columns: new_name = old_name
      Id = claim_no,
      PayId = pmt_no,
      PayTime = payment_time,
      Paid = payment_inflated) %>%
    dplyr::mutate(
      Paid = round(Paid),
      PayDate = as.Date(ceiling(PayTime * 365.25 / 12), origin = "2011-12-31"),
      PayMonth = 12 * (as.integer(substr(PayDate, 1, 4)) - 2012) + as.integer(substr(PayDate, 6, 7))) %>%
    dplyr::select(Id, PayId, PayMonth, Paid) # We drop PayDate, PayTime

  ### rounding some payments to 1'000's
  Paid0 <- ceiling(ClaimsPaid$Paid / 200)
  selected_pmts <- which(Paid0 %in% c(5, 10, 15, 25, 50, 75, 100, 125, 150, 250))
  selected_pmts <- sort(sample(x=selected_pmts, size=floor(.7 * length(selected_pmts))))
  ClaimsPaid[selected_pmts, "Paid"] <- Paid0[selected_pmts] * 200
  
  ### combining all claim features into the `claims` dataset
  claims <- cbind(claims_, claims_info) %>%
    # add (inflated) Ultimate column to claims dataset
    merge(setNames(aggregate(Paid ~ Id, ClaimsPaid, sum), c("Id", "Ultimate")), by = "Id") %>%
    dplyr::select(Id, Type, Age, AccDate, AccMonth, AccWeekday, RepDate, RepMonth, RepDelDays, SetMonth, SetDelMonths, Ultimate, PayCount)

  ### zero-claims (being closed without payments)
  r <- c(0.10, 0.05, 0.05, 0.10, 0.10, 0.10)[type]
  Id <- claims[(claims$PayCount == 1) & (claims$SetDel <= 24), "Id"]
  selected_clms <- sort(sample(x=Id, size=floor(r*length(Id))))
  claims[claims$Id %in% selected_clms, "Ultimate"] <- 0
  claims[claims$Id %in% selected_clms, "PayCount"] <- 0
  # remove those zero-claims from the ClaimsPaid data
  ClaimsPaid <- ClaimsPaid[!(ClaimsPaid$Id %in% selected_clms), ]
  
  ### adding recovery payments (either last or second last payment)
  Id <- claims[claims$PayCount > 2, "Id"]
  r <- c(0.05, 0.15, 0.15, 0.10, 0.05, 0.10)[type]
  selected_clms <- sort(sample(x=Id, size=floor(r*length(Id)))); n_RR <- length(selected_clms)
  selected <- claims$Id %in% selected_clms
  # simulate size of recoveries
  RR <- (claims[selected, "PayCount"] - 1.75) + runif(n_RR)
  Recov <- round(claims[selected, "Ultimate"] / RR)
  # simulate time of recoveries (1 to 5 months of delay)
  RecovDel <- sample(x=c(1:5), size=length(selected_clms), prob=rep(0.2, 5), replace=TRUE)
  # update the `claims` data to reflect the recovery payments
  claims[selected, "Ultimate"] <- claims[selected, "Ultimate"] - Recov
  claims[selected, "PayCount"] <- claims[selected, "PayCount"] + 1
  claims[selected, "SetDelMonths"] <- claims[selected, "SetDelMonths"] + RecovDel
  claims[selected, "SetMonth"] <- claims[selected, "SetMonth"] + RecovDel    
  # update the `ClaimsPaid` data
  # recovery may occur as the last or second last payment
  PC <- claims[selected, "Id"] * 100 + (claims[selected, "PayCount"] - 1)
  Paid_Id <- ClaimsPaid$Id * 100 + ClaimsPaid$PayId
  # ClaimsPaid1 includes all payments for claims without recoveries
  ClaimsPaid1 <- ClaimsPaid[!(Paid_Id %in% PC), ]
  # ClaimsPaid2 is for the second last payment for claims with recoveries
  ClaimsPaid2 <- ClaimsPaid[Paid_Id %in% PC, ]
  # ClaimsPaid3 is for the last payment for claims with recoveries
  ClaimsPaid3 <- ClaimsPaid2
  ClaimsPaid3$PayId <- claims[selected, "PayCount"]
  ClaimsPaid3$PayMonth <- claims[selected, "SetMonth"]
  # half of the recoveries would occur in the penultimate payment and half in last payment
  at_penultimate <- sort(sample(x=c(1:length(selected_clms)), size=floor(0.5*length(selected_clms))))
  ClaimsPaid2[at_penultimate, ]$Paid <- - Recov[at_penultimate]
  ClaimsPaid3[-at_penultimate, ]$Paid <- - Recov[-at_penultimate]
  # re-merge the individual datasets to get ClaimsPaid
  ClaimsPaid <- rbind(ClaimsPaid1, ClaimsPaid2, ClaimsPaid3) %>%
    dplyr::arrange(Id) %>%
    dplyr::select(Id, PayId, PayMonth, Paid)
  
  ### additional settlement delays (so far claims are closed with a final payment)
  Id <- claims[claims$PayCount > 2, "Id"]
  # 75% probability closed with final payment, 25% probability closed with up to 5 months delay
  DelayPlus <- sample(x=c(0:5), size=length(Id), prob=c(.75, rep(0.05, 5)), replace=TRUE)
  claims[claims$Id %in% Id, "SetDelMonths"] <- claims[claims$Id %in% Id, "SetDelMonths"] + DelayPlus
  claims[claims$Id %in% Id, "SetMonth"] <- claims[claims$Id %in% Id, "SetMonth"] + DelayPlus
  
  ### reopenings
  source('./Tools/reopening.r', local = TRUE)
  
  list(
    claims = claims, 
    paid = ClaimsPaid_new, 
    reopen = reopen_rows %>% dplyr::select(Id, SetMonth, EventMonth, OpenInd)
  )
}



data.generation <- function(seed, future_info = FALSE){
  
  exposure = c(40000, 30000, 10000, 40000, 20000, 20000)
  seeds = seed + c(0:5)
  
  for (type in c(1:6)) {
    if (type == 1) {
      # Initialise loop
      data_list <- data.generation.type(type, exposure[type], seeds[type])
      # get individual data components
      claims <- data_list$claims
      paid   <- data_list$paid
      reopen <- data_list$reopen
    } else {
      data_list <- data.generation.type(type, exposure[type], seeds[type])
      # get individual data components
      data_list$claims$Id <- data_list$claims$Id + nrow(claims)
      data_list$paid$Id <- data_list$paid$Id + nrow(claims)
      data_list$reopen$Id <- data_list$reopen$Id + nrow(claims)
      claims <- rbind(claims, data_list$claims)
      paid   <- rbind(paid, data_list$paid)
      reopen <- rbind(reopen, data_list$reopen)
    }
  }
  
  # tidy up data format
  # sort by reporting date and assign new claim Id
  claims <- dplyr::arrange(claims, RepDate)
  Id_map <- data.frame(Id = claims$Id, Id_new = c(1:nrow(claims)))
  claims <- merge(Id_map, claims, by = "Id", all.y = T) %>%
    dplyr::select(-Id) %>% # remove old Id
    dplyr::rename(Id = Id_new) %>%
    dplyr::arrange(Id)
  paid <- merge(Id_map, paid, by = "Id", all.y = T) %>%
    dplyr::select(-Id) %>% # remove old Id
    dplyr::rename(Id = Id_new) %>%
    dplyr::arrange(Id, EventId)
  reopen <- merge(Id_map, reopen, by = "Id", all.y = T) %>%
    dplyr::select(-Id) %>% # remove old Id
    dplyr::rename(Id = Id_new) %>%
    dplyr::arrange(Id)
  
  # impose maximal reporting delay of 3 years
  claims <- claims[claims$RepDelDays <= 365 * 3, ]
  paid <- paid[paid$Id %in% claims$Id, ]
  reopen <- reopen[reopen$Id %in% claims$Id, ]
  
  # save full simulated claims and paid data, if future_info == TRUE
  if (future_info == FALSE) {
    full_claims <- NULL
    full_paid <- NULL
    reopen <- NULL
  } else {
    full_paid <- paid
    full_claims <- claims %>% 
      dplyr::mutate(ReopenInd = as.numeric(SetMonth > SetMonth_old)) %>%
      dplyr::select(-SetMonth_old, -SetDelMonths_old)
  }
  
  # get a stopped view of claim status/set months as of the cutoff date
  claims$Status <- "Closed"
  cutoff.date <- years / time_unit
  claims$SetMonth <- dplyr::if_else(
    # if the claim reopens and has a revised SetMonth that is after the
    # cutoff date, we will consider the original SetMonth
    claims$SetMonth > cutoff.date, claims$SetMonth_old,
    claims$SetMonth)
  claims$SetDelMonths <- claims$SetMonth - claims$RepMonth
  claims <- dplyr::select(claims, -SetMonth_old, -SetDelMonths_old)
  claims[claims$SetMonth > cutoff.date, "Status"]       <- "RBNS"
  claims[claims$SetMonth > cutoff.date, "SetDelMonths"] <- NA
  claims[claims$SetMonth > cutoff.date, "SetMonth"]     <- NA
  claims[claims$RepMonth > cutoff.date, "Status"]       <- "IBNR"
  claims[claims$RepMonth > cutoff.date, "RepDate"]      <- NA
  claims[claims$RepMonth > cutoff.date, "RepDelDays"]   <- NA
  claims[claims$RepMonth > cutoff.date, "RepMonth"]     <- NA
  claims$Status <- factor(claims$Status, levels=c("Closed", "RBNS", "IBNR"))
    
  # censored paid data (output payment history as of the cutoff date)
  paid <- paid[paid$EventMonth <= cutoff.date, ]
  
  # compute cumpaid and add that to `claims`
  cumpaid <- paid %>%
    dplyr::group_by(Id) %>%
    dplyr::summarise(CumPaid = sum(Paid))
  claims <- dplyr::left_join(claims, cumpaid, by = "Id") %>%
    tidyr::replace_na(list(CumPaid = 0))
  
  # reset index
  rownames(claims) <- NULL
  rownames(paid) <- NULL
  rownames(full_claims) <- NULL
  rownames(full_paid) <- NULL
  rownames(reopen) <- NULL
  list(
    # censored paid data, (not-so-fully-censored) claims data
    claims = claims, paid = paid,
    # full simulated data (to be returned only if future_info == TRUE)
    full_claims = full_claims, full_paid = full_paid, reopen = reopen
  )
}

