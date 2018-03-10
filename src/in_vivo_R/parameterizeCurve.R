analyze.curve <- function(data, drugs = NULL, RT50_method = "median", time_interval = 50, stop_character = "/", skip = 10, resolution = 0.5, width = 5, bare_bones = T, do_plot = F) {
  
  #Check that the RT50 method has been appropriately specified.
  if (!(RT50_method == "median" | RT50_method == "regression" | RT50_method == "both")) {
    stop('RT50_method must be "median", "regression", or "both"')
  }
  
  #For use in generating a centred rolling mean.
  hw <- width/2
  
  #Extract the relevant data.
  wavemark <- data[["wavemark"]]
  ejections <- data[["ejections"]]
  keyboard <- data[["keyboard"]]
  
  #Plot in a 4x4 grid.
  if (do_plot) {
    par(mfrow = c(2,2))
  }
  
  #Functions needed for regression analysis.
  if(RT50_method == "regression" | RT50_method == "both") {
    
    #The four parameter logistic function describing y over ln(x) solved for x.
    solvd <- function(A = coef(mod)["A"], B = coef(mod)["B"], med = coef(mod)["med"], m = coef(mod)["m"], Y = medn) {
      X <- med*((A - B)/(Y - B) - 1)^(-1/m)
      return(X)
    }
    
    #Returns NULL for the model and prints a warning if tryCatch determines that the curve cannot be fitted.
    nullfn <- function(cond) {
      message("Curve ", paste(ejections.names.i, j), " could not be fitted.")
      return(NULL)
    }
    
  }
  
  #Generate a list of each time a drug is applied.
  if (!is.null(drugs)) {
    drug_times <- vector(mode = "list", length = length(drugs))
    for (i in 1:length(drugs)) {
      for (j in 1:length(drugs[[i]])) {
        drug_times[[i]] <- c(drug_times[[i]], keyboard[keyboard[, 2] == drugs[[i]][j], 1])
      }
      drug_times[[i]] <- sort(drug_times[[i]])
    }
    rm(i, j)
  }
  
  #For use in finding the end of RT50 curves
  slashes <- keyboard[which(keyboard[, 2] == "/"), 1]
  
  ejections.names <- names(ejections)
  output <- vector(mode = "list", length = length(ejections))
  
  #For each thing ejected.
  for (i in 1:length(ejections.names)) {
    
    ejections.names.i <- ejections.names[i]
    
    if (bare_bones) {
      output[[i]] <- data.frame(matrix(NA, nrow = nrow(ejections[[i]]), ncol = 4 + length(drugs)))
    } else {
      output[[i]] <- data.frame(matrix(NA, nrow = nrow(ejections[[i]]), ncol = 9 + length(drugs)))
    }
    
    wavemark.ij <- wavemark
    
    #For each individual ejection.
    for (j in 1:nrow(ejections[[i]])) {
      #Define variables that are called multiple times in order to avoid repeated indexing
      ejections.ij.tstart <- ejections[[i]][j, "t_start"]
      ejections.ij.tstop <- ejections[[i]][j, "t_stop"]
      
      
      wavemark.ij <- wavemark.ij[Position(function(x) x >= (ejections.ij.tstart - time_interval), x = wavemark.ij):length(wavemark.ij)]
      curve.ij <- wavemark.ij[1:Position(function(x) x >= (slashes[slashes > ejections.ij.tstop][1]) - 1, x = wavemark.ij)]
      
      #Calculate spikes suppressed per nC using the average current recorded during that ejection.
      #NB: this calculation assumes that the baseline and ejection times are equal to the specified "time_interval" (default is 50s).
      #Baseline spikes are included in the output so that the assumption that the baseline does not vary significantly between replicates can be tested.
      baseline_spikes <- sum(curve.ij >= (ejections.ij.tstart - time_interval) & curve.ij < ejections.ij.tstart)
      ejection_spikes <- sum(curve.ij >= ejections.ij.tstart & curve.ij < ejections.ij.tstop)
      suppression <- (baseline_spikes - ejection_spikes)/(ejections[[i]][j, "avg_current"]*time_interval)
      
      #Calculate a rolling mean firing rate for use in determining the RT50.
      centres <- seq(from = as.numeric(ejections.ij.tstop) - 20, to = slashes[slashes > ejections.ij.tstop][1] - hw, by = resolution)
      lowers <- centres - hw
      uppers <- centres + hw
      rolled <- matrix(NA, nrow = length(centres), ncol = 2)
      
      for (k in 1:length(centres)) {
        rolled[k, ] <- c(centres[k], sum(curve.ij >= lowers[k] & curve.ij < uppers[k])/width)
      }
      rm(k, centres, lowers, uppers)
      
      #Terms used in both RT50 calculations
      medn <- baseline_spikes/(2*time_interval)
      domain.start <- rolled[, 1] >= (ejections.ij.tstop + skip)
      
      #Calculate the RT50 as a simple median.
      #This equation finds the y-coordinate closest to half the baseline firing rate 
      #-- searching only values occurring after stopping ejection (plus "skip") but before the recovery curve reaches its maximum --
      #and returns the corresponding x-coordinate.
      RT50.med <- NA
      if (RT50_method == "median" | RT50_method == "both") {
        domain <- domain.start & 1:nrow(rolled) <= which.max(rolled[domain.start, 2])
        rolled.sub <- rolled[domain, ]
        rm(domain)
        coords.med <- rolled.sub[which.min(abs(rolled.sub[, 2] - medn)), ]
        RT50.med <- as.numeric(coords.med[1] - ejections.ij.tstop)
      }
      
      #Calculate the RT50 using nonlinear least-squares regression.
      #The equation is a four-parameter logistic function which describes y over ln(x).
      RT50.reg <- NA
      if (RT50_method == "regression" | RT50_method == "both") {
        block.reg <- ((Position(function(x) x >= ejections.ij.tstop - 5, x = rolled[, 1]) - 1) : (Position(function(x) x > 0, x = domain.start) - 1)) # rolled[, 1] >= ejections.ij.tstop & rolled[, 1] <= domain.start)
        Y <- rolled[-block.reg, 2]
        X <- rolled[-block.reg, 1]
        A.est <- mean(rolled[1:nrow(rolled) >= (which.max(rolled[domain.start, 2]) - 20), 2])                            #Estimates the top of the recovery curve as the mean firing rate from 20s before the maximum *occuring after the skip period*.
        B.est <- mean(rolled[rolled[, 1] <= (ejections.ij.tstop - hw), 2])                                                                                           #Estimates the bottom of the curve as the mean firing rate before one half-width before ejection stops. The half width is included to avoid contamination from a burst immediately following ejection stop.
        m.est <- 12                                                                                                                                                           #I'm not sure how to efficiently estimate slope, so currently it is fixed at twelve.
        
        if (exists("rolled.sub")) {
          med.est <- coords.med[1]
        } else {
          domain <- domain.start & 1:nrow(rolled) <= which.max(rolled[domain.start, 2])
          rolled.sub <- rolled[domain, ]
          rm(domain)
          med.est <- rolled.sub[which.min(abs(rolled.sub[, 2] - medn)), 1]
        }
        
        mod <- tryCatch(nls(Y ~ B + (A - B)/(1 + (X/med)^-m), start = c(A = A.est, B = B.est, med = med.est, m = m.est)), error = nullfn)                                     #tryCatch returns NULL for the model if nls is unable to fit the curve due to an error.
        
        #This step takes the NULL model returned by tryCatch and returns NA for the RT50 if the recovery curve cannot be fitted. It also prints an alert that this particular curve had to be skipped.
        if (is.null(mod)) {
          coord.reg <- NA
          RT50.reg <- NA
        } else {
          coord.reg <- solvd() 
          RT50.reg <- coord.reg - ejections.ij.tstop
        }
        
      }
      
      
      
      #Generate output
      if (bare_bones) {
        row <- c(baseline_spikes, suppression, RT50.med, RT50.reg)
        output[[i]][j, 1:4] <- row
      } else {
        row <- c(ejections.ij.tstart, ejections.ij.tstop, ejections[[i]][j, "t_ejection"], ejections[[i]][j, "avg_current"], baseline_spikes, ejection_spikes, suppression, RT50.med, RT50.reg)
        output[[i]][j, 1:9] <- row
      }
      rm(ejection_spikes, suppression, RT50.med, RT50.reg)
      rm(row)
      
      #Generate plots with the relevant RT50 values for error checking.
      if (do_plot) {
        lowres.rolled <- rolled[seq(from = 1, to = nrow(rolled), by = resolution*2), ]
        ymax <- max(lowres.rolled[, 2])
        ymax <- ymax + 0.1*ymax
        plot(x = lowres.rolled[, 1], y = lowres.rolled[, 2], type = "l", lwd = 3, ylab = "Firing rate (Hz)", xlab = "Time (s)", main = paste(ejections.names.i, j), ylim = c(0,ymax))
        abline(h = baseline_spikes/time_interval, lty = 2)
        abline(v = ejections.ij.tstop, lty = 2)
        
        if (RT50_method == "regression" | RT50_method == "both") {
          if (!is.null(mod)) {
            lines(x = X, y = fitted(mod), col = 4, lwd = 1.5)
            points(x = coord.reg,
                   y = medn,
                   cex = 1.8, pch = 21, bg = scales::alpha(4,0.7))
          }
        }
        
        if (RT50_method == "median" | RT50_method == "both") {
          points(x = coords.med[1],
                 y = coords.med[2],
                 cex = 1.8, pch = 21, bg = scales::alpha(2,0.7))
        }
        
        suppressWarnings(rm(ymax, rolled, coords, coord.reg, medn, rolled.sub, baseline_spikes))
      } else {
        suppressWarnings(rm(rolled, coords, coord.reg, medn, rolled.sub, baseline_spikes))
      }
    }
    
    #Name columns
    if (bare_bones) {
      colnames(output[[i]]) <- c("baseline_spikes", "suppression", "RT50.med", "RT50.reg", names(drugs))
    } else {
      colnames(output[[i]]) <- c("t_start", "t_stop", "t_ejection", "avg_current", "baseline_spikes", "ejection_spikes", "suppression", "RT50.med", "RT50.reg", names(drugs))
    }
    
    
    #Make drug columns into ordered factors with appropriate labels/levels
    if (!is.null(drugs)) {
      for (k in 1:length(drugs)) {
        breaks <- c(0, drug_times[[k]], max(wavemark) + 300)
        labels <- LETTERS[1:(length(drug_times[[k]]) + 1)]
        drug.k <- cut(as.numeric(ejections[[i]][, "t_start"]), breaks = breaks, labels = labels, right = F)
        output[[i]][, names(drugs)[k]] <- ordered(drug.k, levels = labels)
      }
      rm(k)
    }
    
    #Remove unnecessary RT50 columns
    if (all(is.na(output[[i]][, "RT50.med"]))) {output[[i]][, "RT50.med"] <- NULL}
    if (all(is.na(output[[i]][, "RT50.reg"]))) {output[[i]][, "RT50.reg"] <- NULL}
    
    #Not actually useful here, but should be incorporated into statistical analysis functions
    #error <- (sd(output[[i]][, "baseline_spikes"])*100)/(sqrt(length(output[[i]][, "baseline_spikes"]))*mean(output[[i]][, "baseline_spikes"]))
    #if (error >= 10) {
    #  warning(round(error,1), "% error on the number of baseline spikes in ", ejections.names.i, " curves. \n", sep = "")
    #} else {
    #  cat(round(error,1), "% error on the number of baseline spikes in ", ejections.names.i, " curves. \n", sep = "")
    #}
    #rm(error)
    
  }
  
  names(output) <- ejections.names
  
  return(output)
}