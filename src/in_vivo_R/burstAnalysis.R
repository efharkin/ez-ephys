sparky <- function(data, start_ISI = 0.1, end_ISI = 0.1, subset_data = F) {
  
  if (subset_data) {
    data <- data[["wavemark"]]
  }
  
  cond <- NULL #tryCatch error functions require a cond argument.
  
  first_spike <- data[-length(data)]
  second_spike <- data[-1]
  ISI <- second_spike - first_spike
  
  l.ISI <- length(ISI)
  bursts <- matrix(NA, ncol = 3, nrow = l.ISI)
  events <- vector("numeric", length = l.ISI + 1)
  i <- 1
  j <- 1
  k <- 1
  while (i <= l.ISI) {
    if (ISI[i] < start_ISI) {
      ISI_sub <- ISI[i:(i + 50)]
      burst_end.i <- tryCatch(Position(function(x) x > end_ISI, x = ISI_sub) - 1, error = function (cond) NULL)
      
      if (!is.null(burst_end.i)) {
        n_spikes.i <- burst_end.i + 1
        
        burst.i <- ISI_sub[1:burst_end.i]
        middle.i <- (first_spike[i] + second_spike[burst_end.i + i])/2
        mean_ISI <- .Internal(mean(burst.i))
        
        bursts[j, ] <- c(middle.i, n_spikes.i, mean_ISI)
        
        events[k] <- middle.i
        
        i <- i + burst_end.i + 1
        j <- j + 1
        k <- k + 1
      } else {
        break
      }
    } else {
      events[k] <- first_spike[i]
      i <- i + 1
      k <- k + 1
    }
  }
  
  #Clean up data to remove extra cells.
  bursts <- bursts[1:(j - 1), ]
  events <- events[1:(k - 1)]
  rm(i, j, k)
  
  
  #Create IBI and IEI vectors. Overall ISI and within-burst ISI vectors already exist.
  first_burst <- bursts[-nrow(bursts), 1]
  second_burst <- bursts[-1, 1]
  IBI <- second_burst - first_burst
  
  first_event <- events[-length(events)]
  second_event <- events[-1]
  IEI <- second_event - first_event
  
  rm(first_spike, second_spike, first_burst, second_burst, first_event, second_event)
  
  #Generate first couple of rows of interval output.
  #Order will be: overall ISI, ISI within bursts, IBI, IEI.
  #First row will be means, second row will be standard deviations. This is so the calculations to convert these to rates can be vectorized.
  interval_mean <- c(.Internal(mean(ISI)), .Internal(mean(bursts[, 3])), .Internal(mean(IBI)), .Internal(mean(IEI)))
  interval_SD <- c(sd(ISI), sd(bursts[, 3]), sd(IBI), sd(IEI))
  
  #Same output as rates.
  Hz_mean <- 1/interval_mean
  Hz_SD <- c(sd(1/ISI), sd(1/bursts[, 3]), sd(1/IBI), sd(1/IEI))
  
  #Assemble interval output.
  interval.out <- data.frame(rbind(interval_mean, interval_SD, Hz_mean, Hz_SD))
  colnames(interval.out) <- c("firing_overall", "firing_burst", "burst", "discharge")
  
  #General output.
  #N spikes, N bursts, N events, mean spikes per burst, sd spikes per burst, % spikes in burst, and % bursting.
  general.out <- c("n_spikes" = length(data),
                   "n_bursts" = nrow(bursts),
                   "n_discharges" = length(events),
                   "mean_spikes/burst" = .Internal(mean(bursts[, 2])),
                   "SD_spikes/burst" = sd(bursts[, 2]),
                   "pct_spikes_in_burst" = sum(bursts[, 2])*100/length(data),
                   "pct_bursting" = nrow(bursts)*100/length(events))
  
  #Assemble output into a list.
  output <- list(general.out, interval.out)
  names(output) <- c("general", "intervals_rates")
  
  return(output)
}

