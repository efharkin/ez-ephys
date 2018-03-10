#Suggestions: 
#Have tonic activation be able to take manually specified timepoints.


setwd("/Users/Emerson/Desktop")

#Import the data.
dat.demo <- import("Tonic Activation/TA1.txt", pumps = list(`Pump 2` = c("Ser", 0, "+")))
dat.demo[["keyboard"]]
dat.demo[["ejections"]][[1]][, -c(2,3)]
dat.demo[["wavemark"]][1:10]

#Extract effect of ionto on firing rate.
analyze.curve(dat.demo, drugs = list(citalopram = "c", WAY = c("s", "w")))[[1]][, 1:3]
analyze.curve(dat.demo, drugs = list(citalopram = "c", WAY = c("s", "w")))[[1]][1:2, 3:4]
invisible <- analyze.curve(dat.demo, drugs = list(citalopram = "c", WAY = c("s", "w")), do_plot = T)
invisible <- analyze.curve(dat.demo, drugs = list(citalopram = "c", WAY = c("s", "w")), do_plot = T, RT50_method = "regression")

#Tonic activation.
tonic.activation(dat.demo, drugs = c(saline = "s", WAY = "w"))[, 1:2]
tonic.activation(dat.demo, drugs = c(saline = "s", WAY = "w"))[, c(1, 4)]


#Analyze whole recording for bursting.
sparky(dat.demo[["wavemark"]])[[1]][c(2, 4, 6, 7)]
sparky(dat.demo[["wavemark"]])[[2]][3, c(2, 3)]


#Speedy!
library(parallel)
cl <- makeCluster(detectCores())

t.start <- proc.time()
file_names <- list.files("speed test", full.names = T)

batch_data <- parLapply(cl, file_names,
                        import,
                        pumps = list(`Pump 2` = c("Ser", 0, "+")))
batch_RT50 <- parLapply(cl, batch_data,
                        analyze.curve,
                        drugs = list(citalopram = "c", WAY = c("s", "w")))
batch_TA <- parLapply(cl, batch_data,
                      tonic.activation,
                      drugs = c(saline = "s", WAY = "w"))
batch_burst <- parLapply(cl, batch_data,
                         sparky,
                         subset_data = T)

t.stop <- (proc.time() - t.start)[3]
t.stop[[1]]
(t.stop/length(file_names))[[1]]

stopCluster(cl)
rm(cl)

#Demonstrate plot.
p.data <- data.frame(matrix(NA, nrow = 0, ncol = 2))
for (i in seq_along(batch_RT50)) {
  p.data <- rbind(p.data, batch_RT50[[i]][[1]][1:2, c("RT50.med", "citalopram")])
}
rm(i)
par(mfrow = c(1,1))
boxplot(RT50.med ~ citalopram, data = p.data)

rm(p.data, batch_burst, batch_data, batch_RT50, batch_TA, dat.demo, file_names, invisible, t.start, t.stop)