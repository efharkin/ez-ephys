import <- function(data, pumps = NULL, summarize_only = T) {
  
  aliquot <- readChar(data, nchars = 30)
  if (grepl("\r\n", aliquot)) {
    linebreak <- "\r\n"
  } else {
    if (grepl("\r", aliquot)) {linebreak <- "\r"}
    if (grepl("\n", aliquot)) {linebreak <- "\n"}
  }
  
  size <- file.info(data)[["size"]]
  data <- readChar(data, size, useBytes = T)
  data <- .Internal(strsplit(data, linebreak, fixed = T, perl = F, useBytes = T))[[1]]
  
  
  stops <- grep("CHANNEL", data, fixed = T, useBytes = T) - 2
  
  p.names <- names(pumps)
  
  ejections.out <- vector("list", length(p.names))
  pumps.out <- vector("list", length(p.names))
  
  if(!is.null(pumps)) {
    
    for (i in seq_along(p.names)) {
      pump.name <- p.names[i]
      start.point <- grep(pump.name, data, fixed = T, useBytes = T)
      start.point <- start.point[length(start.point)] + 4
      
      stop.point <- stops[stops > start.point]
      stop.point <- stop.point[1]
      
      
      if (!is.na(stop.point)) {
        waveform <- as.numeric(data[start.point:stop.point])
      } else {
        waveform <- as.numeric(data[start.point:(length(data) - 1)])
      }
      
      
      #Grab ejections if thresholds/charges are specified.
      if (length(pumps[[pump.name]]) == 3) {
        threshold.i <- pumps[[pump.name]][2]
        
        if (pumps[[pump.name]][3] == "+") {
          threshold.ind <- .Internal(which(waveform >= threshold.i))
        } else {
          threshold.ind <- .Internal(which(waveform <= threshold.i))
        }
        
        #This finds any discontinuities in the indexes of above-threshold currents: that is, the edges of ejections. 
        n1.threshold.ind <- threshold.ind[-1]
        nl.threshold.ind <- threshold.ind[-length(threshold.ind)]
        d.threshold.ind <- n1.threshold.ind - nl.threshold.ind
        edges <- .Internal(which(d.threshold.ind != 1))
        
        starting.i <- c(threshold.ind[1], n1.threshold.ind[edges])
        stopping.i <- c(nl.threshold.ind[edges], threshold.ind[length(threshold.ind)])
        
        
        if (length(starting.i) != length(stopping.i)) {
          warning('Unequal number of starts and stops in ', pump.name,'. Ejections not retrieved.')
          next
        }
        
        line. <- strsplit(data[start.point - 1], ",")[[1]]
        sampling_rate <- as.numeric(line.[3])
        offset. <- round(as.numeric(line.[2]), 2)
        
        t_start <- starting.i*sampling_rate + offset.
        t_stop <- stopping.i*sampling_rate + offset.
        t_ejection <- t_stop - t_start
        avg_current <- vector("numeric", length(starting.i))
        
        
        for (j in seq_along(starting.i)) {
          avg_current[j] <- .Internal(mean(waveform[starting.i[j]:stopping.i[j]]))
        }
        
        #Assemble output, grouping with other pumps containing the same thing if possible.
        if (!(pumps[[pump.name]][1] %in% names(ejections.out))) {
          ejections.out[[i]] <- data.frame(t_start, t_stop, t_ejection, avg_current)
          names(ejections.out)[i] <- pumps[[pump.name]][1]
          colnames(ejections.out[[i]]) <- c("t_start", "t_stop", "t_ejection", "avg_current")
        } else {
          ejections.out[[pumps[[pump.name]][1]]] <- rbind(ejections.out[[pumps[[pump.name]][1]]], data.frame(t_start, t_stop, t_ejection, avg_current))
          rownames(ejections.out[[pumps[[pump.name]][1]]]) <- NULL
          colnames(ejections.out[[pumps[[pump.name]][1]]]) <- c("t_start", "t_stop", "t_ejection", "avg_current")
        }
        
        #Go to the next iteration of the loop if only_summarize is on, skipping saving the waveform.
        if (summarize_only) {next}
      }
      
      line. <- strsplit(data[start.point - 1])[[1]]
      sampling_rate <- as.numeric(line.[3])
      offset. <- round(as.numeric(line.[2]), 2)
      
      time <- seq(from = offset., by = sampling_rate, length.out = length(waveform)) 
      
      pumps.out[[i]] <- data.frame(time, waveform)
      names(pumps.out)[i] <- pumps[[pump.name]][1]
      colnames(pumps.out[[i]]) <- c("time", "waveform")
    }
    
    nulls <- vector("logical", length(ejections.out))
    for (i in seq_along(ejections.out)) {
      nulls[i] <- is.null(ejections.out[[i]])
    }
    ejections.out <- ejections.out[!nulls]
    
    nulls <- vector("logical", length(pumps.out))
    for (i in seq_along(pumps.out)) {
      nulls[i] <- is.null(pumps.out[[i]])
    }
    pumps.out <- pumps.out[!nulls]
  } else {
    ejections.out <- NULL
    pumps.out <- NULL
  }
  
  if (summarize_only) {pumps.out <- NULL}
  
  #Retrieve the wavemark channel
  start.point <- grep("wavemark", data, fixed = T, useBytes = T)
  start.point <- start.point[length(start.point)] + 2
  
  stop.point <- stops[stops > start.point]
  stop.point <- stop.point[1]
  
  if(!is.na(stop.point)) {
    wavemark.out <- data[start.point:stop.point]
  } else {
    wavemark.out <- data[start.point:(length(data) - 1)]
  }
  wavemark.out <- as.numeric(wavemark.out)
  
  
  #Retrieve the keyboard channel
  start.point <- grep("code", data, fixed = T, useBytes = T)
  
  if (length(start.point) > 0) {
    start.point <- start.point[length(start.point)] + 2
    
    stop.point <- stops[stops > start.point]
    stop.point <- stop.point[1]
    
    if(!is.na(stop.point)) {
      keyboard.out <- data[start.point:stop.point]
    } else {
      keyboard.out <- data[start.point:(length(data) - 1)]
    }
    
    keyboard.out <- strsplit(keyboard.out, ",")
    keyboard.out <- matrix(unlist(keyboard.out), ncol = 6, byrow = T)[, 1:2]
    keyboard.out <- data.frame(as.numeric(keyboard.out[, 1]), substring(keyboard.out[, 2], 1, 1), stringsAsFactors = F)
    colnames(keyboard.out) <- c("time", "keystroke")
  } else {
    keyboard.out <- NULL
  }
  
  output <- list(keyboard.out, ejections.out, pumps.out, wavemark.out)
  names(output) <- c("keyboard", "ejections", "pumps", "wavemark")
  
  return(output)
}