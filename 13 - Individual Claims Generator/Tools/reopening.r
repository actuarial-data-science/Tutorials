#####################################################################
#### Authors Melantha Wang & Mario Wuthrich
#### Date 28/05/2022  
#### Auxiliary functions for claim re-openings
#####################################################################

# For use within the data.generation.type() function...
# Goal: Simulate claim status indicator process I = (PayInd, OpenInd)
# For simplicity, we will assume that:
# > A claim can only reopen once after the first claim closure.
# > Only claims exceeding 20000 can re-open, and they re-open with probability 0.1.
# > Re-opening within 12 months after closing, with exponential decay after closing.
# > The re-opening may (with probability 0.2) or may not (with probability 0.8) occur with a payment:
#   - If the re-opening occurs with a payment, we will simulate a payment, and close the claim 1 month after that.
#   - If the re-opening occurs without a payment, we will assume at most 1 payment within 6 months of reopening, and an immediate closure (i.e. same month) after or with the payment. 

# First we do some labelling on our current data:
ClaimsPaid_new <- ClaimsPaid %>%
  # Add SetMonth, RepMonth, Ultimate information to the paid data
  merge(dplyr::select(claims, Id, SetMonth, RepMonth, Ultimate), by = "Id") %>%
  dplyr::mutate(
    # Label payments
    PayInd = 1,
    # Label claim status (at month-end)
    # (ps this is why we need to reconcile the PayMonth calc earlier)
    OpenInd = ifelse(PayMonth == SetMonth, 0, 1)) %>%
  dplyr::ungroup()

# Add rows for claims that closed after a final payment (PayInd = 0, OpenInd = 0)
late_Id <- Id[DelayPlus > 0] # get the Ids for such claims
late_closure_rows <- data.frame(
  Id = late_Id, 
  PayId = NA,
  RepMonth = claims[claims$Id %in% late_Id, "RepMonth"],
  SetMonth = claims[claims$Id %in% late_Id, "SetMonth"],
  EventMonth = claims[claims$Id %in% late_Id, "SetMonth"],
  Ultimate = claims[claims$Id %in% late_Id, "Ultimate"],
  Paid = 0,
  PayInd = 0,
  OpenInd = 0
)

# Simulate reopenings
# Step 1: Randomly select a subset of claims that will reopen after the first 
# recorded claim closure
sim_reopen <- function(Ultimate) {
  if (Ultimate > 20000) {
    sample(c(0, 1), size = 1, prob = c(0.9, 0.1))
  } else {
    0
  }
}
sim_reopen <- Vectorize(sim_reopen)
reopen_Id <- claims$Id[as.logical(sim_reopen(claims$Ultimate))]

# Record reopen rows (to be concatenated with ClaimsPaid)
reopen_rows <- data.frame(
  Id = reopen_Id, 
  RepMonth = claims[claims$Id %in% reopen_Id, "RepMonth"],
  SetMonth = claims[claims$Id %in% reopen_Id, "SetMonth"],
  # EventMonth to be simulated in step 2.1
  Ultimate = claims[claims$Id %in% reopen_Id, "Ultimate"],
  # Paid, PayInd to be simulated in 2.2 (depends on whether the re-open occurs with a payment)
  OpenInd = 1
)

# Step 2: What will happen after reopening
# 2.1 When - Re-opening within 12 months after closing, with exponential decay
sim_reopen_mth <- function(SetMonth) {
  del <- rexp(1, rate = 1/6)
  while (del >= 12) {
    del <- rexp(1, rate = 1/6)
  }
  return(ceiling(del) + SetMonth)
}
sim_reopen_mth <- Vectorize(sim_reopen_mth)
reopen_rows$EventMonth <- sim_reopen_mth(reopen_rows$SetMonth)

# 2.2 Payment with/after reopening
# The re-opening may (prob = 0.2) or may not (prob = 0.8) occur with a payment
reopen_rows$PayInd <- sample(
  c(0, 1), size = nrow(reopen_rows), prob = c(0.8, 0.2), replace = T)

# Case 1: If the re-opening does occur with a payment
# We will activate the PayInd, and close the claim 1 month after.
sim_reopen_paid <- function(PayInd, Ultimate) {
  beta_params <- get_Beta_parameters(target_mean = 0.15, target_cv = 0.2)
  if (PayInd == 1) {
    Ultimate * rbeta(1, shape1 = beta_params[1], shape2 = beta_params[2])
  } else {
    0
  }
}
sim_reopen_paid <- Vectorize(sim_reopen_paid)
reopen_rows$Paid <- round(sim_reopen_paid(reopen_rows$PayInd, reopen_rows$Ultimate))
reopen_rows$PayId <- ifelse(
  reopen_rows$PayInd == 1,
  claims[claims$Id %in% reopen_Id, "PayCount"] + 1, NA
)

# Close the claim 1 month after re-opening
reopen_wp_Id <- reopen_rows[reopen_rows$PayInd == 1, "Id"]
if (length(reopen_wp_Id) > 0) {
  reopen_closure_rows <- data.frame(
    Id = reopen_wp_Id, 
    PayId = NA,
    RepMonth = claims[claims$Id %in% reopen_wp_Id, "RepMonth"],
    SetMonth = claims[claims$Id %in% reopen_wp_Id, "SetMonth"],
    EventMonth = reopen_rows[reopen_rows$Id %in% reopen_wp_Id, "EventMonth"] + 1,
    Ultimate = claims[claims$Id %in% reopen_wp_Id, "Ultimate"],
    Paid = 0,
    PayInd = 0,
    OpenInd = 0
  )
} else {
  reopen_closure_rows <- NULL
}

# Case 2: If the re-opening does not occur with a payment
# We will assume at most 1 payment within 6 months of reopening, and an immediate
# (i.e. same month-end) closure after the payment.
reopen_np_Id <- reopen_rows[reopen_rows$PayInd == 0, "Id"] # reopen w/o payment
paid_del <- data.frame(
  Id = reopen_np_Id,
  paid_del = ceiling(rexp(length(reopen_np_Id), rate = 1/3))
)

for (row in 1:nrow(paid_del)) {
  id <- paid_del[row, "Id"]
  del <- paid_del[row, "paid_del"]
  if (del <= 6) {
    # If del <=6, add a payment row, close the claim in the same month
    new_row <- data.frame(
      Id = id,
      PayId = claims[claims$Id == id, "PayCount"] + 1,
      RepMonth = claims[claims$Id == id, "RepMonth"],
      SetMonth = claims[claims$Id == id, "SetMonth"],
      EventMonth = reopen_rows[reopen_rows$Id == id, "EventMonth"] + del,
      Ultimate = claims[claims$Id == id, "Ultimate"],
      PayInd = 1,
      OpenInd = 0 # immediate closure with the payment
    )
  } else {
    # If del >=7, close the claim without payment
    new_row <- data.frame(
      Id = id,
      PayId = NA,
      RepMonth = claims[claims$Id == id, "RepMonth"],
      SetMonth = claims[claims$Id == id, "SetMonth"],
      EventMonth = reopen_rows[reopen_rows$Id == id, "EventMonth"] + del,
      Ultimate = claims[claims$Id == id, "Ultimate"],
      PayInd = 0,
      OpenInd = 0
    )
  }
  new_row$Paid <- round(sim_reopen_paid(new_row$PayInd, new_row$Ultimate))
  reopen_closure_rows <- rbind(reopen_closure_rows, new_row)
}

# Update claims dataset: "Ultimate", "SetMonth", "SetDelMonths", "PayCount"
reopen_new_info <- reopen_closure_rows %>%
  dplyr::arrange(Id) %>%
  dplyr::rename(SetMonth_new = EventMonth) %>%
  dplyr::mutate(
    SetDelMonths_new = SetMonth_new - RepMonth,
    Ultimate_new = Ultimate + Paid + reopen_rows$Paid,
    PayCount_new = claims[claims$Id %in% reopen_Id, "PayCount"] + 
      as.numeric(Ultimate_new > Ultimate)) %>%
  dplyr::select(Id, Ultimate_new, SetMonth_new, SetDelMonths_new, PayCount_new)
# keep the first settlement dates
claims$SetMonth_old <- claims$SetMonth
claims$SetDelMonths_old <- claims$SetDelMonths
claims[claims$Id %in% reopen_Id, "SetMonth"] <- reopen_new_info$SetMonth_new
claims[claims$Id %in% reopen_Id, "SetDelMonths"] <- reopen_new_info$SetDelMonths_new
claims[claims$Id %in% reopen_Id, "Ultimate"] <- reopen_new_info$Ultimate_new
claims[claims$Id %in% reopen_Id, "PayCount"] <- reopen_new_info$PayCount_new

# Add the payment rows to the paid dataset
# ClaimsPaid includes only rows with a payment
reopen_paid_rows <- rbind(reopen_rows, reopen_closure_rows) %>%
  dplyr::filter(PayInd == 1) %>%
  dplyr::rename(PayMonth = EventMonth)
ClaimsPaid <- rbind(ClaimsPaid_new, reopen_paid_rows) %>%
  dplyr::arrange(Id, PayMonth) %>%
  dplyr::select(Id, PayId, PayMonth, Paid, PayInd, OpenInd) %>%
  dplyr::ungroup()

# ClaimsPaid_new includes rows for claim status changes
ClaimsPaid_new <- ClaimsPaid_new %>%
  # Rename column to "EventMonth" as we also want to track changes in claim status
  # Event = a payment or a change in claim status
  dplyr::rename(EventMonth = PayMonth) %>%
  rbind(late_closure_rows, reopen_rows, reopen_closure_rows) %>%
  dplyr::rename(EventId = PayId) %>%
  dplyr::select(Id, EventId, EventMonth, Paid, PayInd, OpenInd) %>%
  dplyr::arrange(Id, EventMonth)

# Aggregate payment by months
ClaimsPaid_new <- ClaimsPaid_new %>%
  dplyr::group_by(Id, EventMonth) %>%
  dplyr::summarise(
    Paid = sum(Paid),
    PayInd = as.numeric(sum(Paid) > 0),
    OpenInd = last(OpenInd), # latest claim status
    .groups = 'drop') %>%
  dplyr::ungroup() %>%
  dplyr::group_by(Id) %>%
  dplyr::mutate(EventId = dplyr::row_number()) %>%
  dplyr::select(Id, EventId, EventMonth, Paid, PayInd, OpenInd) %>%
  dplyr::ungroup() %>%
  as.data.frame()

PayCount_new <- ClaimsPaid_new %>%
  dplyr::group_by(Id) %>%
  dplyr::summarise(PayCount = sum(PayInd))
claims <- merge(claims, PayCount_new, by = "Id", suffixes = c("_old", ""), all.x = T) %>%
  dplyr::mutate(PayCount = ifelse(is.na(PayCount), PayCount_old, PayCount)) %>%
  dplyr::select(-PayCount_old)
