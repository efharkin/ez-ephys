tonic.activation <- function(data, drugs, vehicle = "saline") {
  
  keyboard <- data[["keyboard"]]
  
  drug_times <- c()
  names.drugs <- names(drugs)
  if (is.null(names.drugs)) {
    for (j in seq_along(drugs)) {
      drug_times <- c(drug_times, keyboard[keyboard[, 2] == drugs[j], 1])
    }
  } else {
    for (j in seq_along(drugs)) {
      drug_times.j <- keyboard[keyboard[, 2] == drugs[j], 1]
      
      names.drugs.j <- names.drugs[j]
      if (length(drug_times.j) > 1) {
        names.tmp <- vector(mode = "numeric", length = length(drug_times.j))
        for (k in seq_along(drug_times.j)) {
          names.tmp[k] <-paste(names.drugs.j, k, sep = "_")
        }
      } else {
        names.tmp <- names.drugs.j
      }
      
      names(drug_times.j) <- names.tmp
      drug_times <- c(drug_times, drug_times.j)
    }
  }
  drug_times <- sort(drug_times)
  
  rm(j, drug_times.j)
  
  wavemark <- data[["wavemark"]]
  
  wavemark.j <- wavemark
  initial <- 1
  
  cut_recording <- vector(mode = "list", length = length(drug_times) + 1)
  if (is.null(names(drug_times))) {
    names(cut_recording) <- LETTERS[1:(length(drug_times) + 1)]
  } else {
    names(cut_recording) <- c("pre", names(drug_times))
  }
  
  
  #From the end of the previous injection to before each injection. Non-overlapping
  for (j in 1:length(drug_times)) {
    wavemark.j <- wavemark.j[initial:length(wavemark.j)]
    final <- Position(function(x) x >= drug_times[j], x = wavemark.j) - 1
    cut_recording[[j]] <- wavemark.j[1:final]
    initial <- final + 1
  }
  
  #From last injection to end
  cut_recording[[length(cut_recording)]] <- wavemark.j[initial:length(wavemark.j)]
  
  
  minus.vehicle <- drug_times[!(names(drug_times) %in% vehicle)]
  time_interval <- (minus.vehicle[length(minus.vehicle)] - minus.vehicle[1])/(length(minus.vehicle) - 1)
  
  output <- matrix(data = NA, nrow = length(cut_recording), ncol = 4)
  
  
  t.start <- drug_times[1] - time_interval
  rate <- sum(cut_recording[[1]] >= t.start)/time_interval
  t <- time_interval
  output[1, 2:3] <- c(rate, t) 
  
  t.start <- drug_times[2] - time_interval
  rate <- sum(cut_recording[[2]] >= t.start)/time_interval
  t <- time_interval
  pct.rate <- (rate - output[1, 2])*100/output[1, 2]
  output[2, 2:4] <- c(rate, t, pct.rate)
  
  for (i in 1:(length(minus.vehicle) - 1) + 2) {
    t <- minus.vehicle[i - 1] - minus.vehicle[i - 2]
    rate <- length(cut_recording[[i]])/t
    pct.rate <- (rate - output[1, 2])*100/output[1, 2]
    output[i, 2:4] <- c(rate, t, pct.rate)
  }
  
  t.stop <- minus.vehicle[length(minus.vehicle)] + time_interval
  rate <- sum(cut_recording[[length(cut_recording)]] < t.stop)/time_interval
  t <- time_interval
  pct.rate <- (rate - output[1, 2])*100/output[1, 2]
  output[nrow(output), 2:4] <- c(rate, t, pct.rate)
  
  output <- data.frame(output)
  output[, 1] <- c("baseline", names(cut_recording)[-1])
  colnames(output) <- c("level", "firing_rate", "dt", "pct.change")
  
  return(output)
}