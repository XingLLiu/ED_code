# NEED TO CHECK

# Date: July 31st, 2019

# Calculate number of patients in emergency department for every 15 minute interval of every year
# from 2008 - present based on WellSoft Data Start and End time of visit

# Input: processed_wellsoft.csv
#
# Output: flow_var_per_patient_wellSoft.rds: Number of patients at start of visit, total number of patients, 
#                                   CTAS1-5, 9 patients during duration of entire visit for each 
#                                   registration number (i.e. each patient visit)


#path <- "/home/lebo/data/lebo/data/"
library(data.table)
path <- "./data/"
wellSoft <- fread(paste0(path, "cleaned_wellSoft.csv"), integer64 = "numeric", na.strings = c('""', "", "NA", "NULL"))

wellSoft$DischargeWithUpdates <- ifelse(!is.na(wellSoft$Updated_Discharge_Time_596), wellSoft$Updated_Discharge_Time_596, wellSoft$Discharge_Time_276)

wellSoft$Arrival_Time_9 <- as.POSIXct(strptime(wellSoft$Arrival_Time_9, format="%Y-%m-%d %H:%M:%S"), tz="EST")
wellSoft$DischargeWithUpdates <- as.POSIXct(strptime(wellSoft$DischargeWithUpdates, format="%Y-%m-%d %H:%M:%S"), tz="EST")

wellSoft$LengthOfStayInMinutes <- as.numeric(difftime(wellSoft$DischargeWithUpdates, wellSoft$Arrival_Time_9, units = "mins"))
range(wellSoft$LengthOfStayInMinutes, na.rm = T)
wellSoft <- wellSoft[order(wellSoft$Arrival_Time_9),]


wellSoft$Ctas_326[!wellSoft$Ctas_326 %in% c("1", "2", "3", "4", "5", "9") ] <- NA

ordered.Arrival_Time_9 <- wellSoft$Arrival_Time_9

ordered.EndTime <- wellSoft$DischargeWithUpdates
ordered.EndTime <- na.exclude(ordered.EndTime[order(ordered.EndTime)])

first.time <- round(ordered.Arrival_Time_9[1], "hours"); print(first.time) 
last.time <- round(ordered.EndTime[length(ordered.EndTime)-1], "hours"); print(last.time)

num.mins <- as.integer(difftime(last.time, first.time, units='mins')) + 1; num.mins


date.sequence <- seq(first.time, last.time, by="1 hour"); head(date.sequence); tail(date.sequence)

pat.per.min <- seq(0, 0, length=length(date.sequence))
num.ones <- seq(0, 0, length=length(date.sequence)) # collect number of patients in ED with certain CTAS Scores
num.twos <- seq(0, 0, length=length(date.sequence))
num.threes <- seq(0, 0, length=length(date.sequence))
num.fours <- seq(0, 0, length=length(date.sequence))
num.fives <- seq(0, 0, length=length(date.sequence))
num.nines <- seq(0, 0, length=length(date.sequence))

names(pat.per.min) <- date.sequence
names(num.ones) <- date.sequence; names(num.twos) <- date.sequence; names(num.threes) <- date.sequence; names(num.fours) <- date.sequence
names(num.fives) <- date.sequence; names(num.nines) <- date.sequence
#
x <-Sys.time()
print(paste("Start time:", x))
for (pat.num in 1:nrow(wellSoft)) { # 
  pat.data <- wellSoft[pat.num,]
  print(paste("Patient", pat.num, "out of 960006"))
  if (!is.na(pat.data$DischargeWithUpdates)) {
    
    assign.dates <- ifelse(!is.na(pat.data$LengthOfStayInMinutes) & pat.data$LengthOfStayInMinutes > 0, TRUE, FALSE)
    
    if (assign.dates) {
      
      pat.end <- round(pat.data$DischargeWithUpdates, "hours")
      pat.start <- round(pat.data$Arrival_Time_9, "hours")

      
      pat.date.sequence <- seq(pat.start, pat.end, by="1 hour")
      # record patient was in ED during that time frame
      pat.per.min[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] <- pat.per.min[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] + 1
      
      
      # next, record the CTASScore of patients
      
      if (!is.na(pat.data$Ctas_326)) {
        pat.CTASScore <- as.numeric(pat.data$Ctas_326)
        
        if (pat.CTASScore == 1) {
          num.ones[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] <- num.ones[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] + 1
        } else if (pat.CTASScore == 2) {
          num.twos[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] <- num.twos[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] + 1
        } else if (pat.CTASScore == 3) {
          num.threes[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] <- num.threes[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] + 1
        } else if (pat.CTASScore == 4) {
          num.fours[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] <- num.fours[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] + 1
        } else if (pat.CTASScore == 5) {
          num.fives[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] <- num.fives[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] + 1
        } else if (pat.CTASScore == 9) {
          num.nines[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] <- num.nines[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] + 1
        }
        
      }

    }
    
    
  }
  
  
  
}

y <- Sys.time()
print(paste("End time:", y))
print(paste("It took", round(difftime(y, x, units='hours'), 3), "hours to complete"))


fwrite(x=data.frame("time"=names(pat.per.min), "numPatients"=pat.per.min), file = paste0(path, "flowVars/pat.per.min.csv"))
fwrite(x=data.frame("time"=names(num.ones), "numPatients"=num.ones), file = paste0(path, "flowVars/num.ones.csv"))
fwrite(x=data.frame("time"=names(num.twos), "numPatients"=num.twos), file = paste0(path, "flowVars/num.twos.csv"))
fwrite(x=data.frame("time"=names(num.threes), "numPatients"=num.threes), file = paste0(path, "flowVars/num.threes.csv"))
fwrite(x=data.frame("time"=names(num.fours), "numPatients"=num.fours), file = paste0(path, "flowVars/num.fours.csv"))
fwrite(x=data.frame("time"=names(num.fives), "numPatients"=num.fives), file = paste0(path, "flowVars/num.fives.csv"))
fwrite(x=data.frame("time"=names(num.nines), "numPatients"=num.nines), file = paste0(path, "flowVars/num.nines.csv"))
# Now, calculate statistics per patient

times <- as.POSIXct(strptime(names(pat.per.min), format="%Y-%m-%d %H:%M:%S"), tz="EST")
p1 <- ggplot(data=data.frame("times"=times, "pat.per.min"= data.frame(pat.per.min)$pat.per.min), 
       aes(x=times, y=pat.per.min)) + geom_line() + theme_bw() + 
  ggtitle("Number of Patients Arriving in the ED Per Hour") + xlab("Date") + ylab("Number of Patients")

ggsave(filename = paste0(path, "flowVars/visitsPerHour.png"), plot = p1, type = "cairo")


total.patients.in.er <- data.frame(matrix(0, nrow=nrow(wellSoft), ncol=9))
colnames(total.patients.in.er) <- c("RegistrationNumber", "NumberOfPatientsAtStart", "TotalNumberOfPatients",
                                    "NumberCTAS1", "NumberCTAS2", "NumberCTAS3", "NumberCTAS4",
                                    "NumberCTAS5","NumberCTAS9")

x <-Sys.time()
print(paste("Start time:", x))
for (pat.num in 1:nrow(wellSoft)) { 
  pat.data <- wellSoft[pat.num,]
  print(paste("Patient", pat.num, "out of 960006"))
  if (!is.na(pat.data$DischargeWithUpdates)) {
    
    assign.dates <- ifelse(!is.na(pat.data$LengthOfStayInMinutes) & pat.data$LengthOfStayInMinutes > 0, TRUE, FALSE)
    
    if (assign.dates) {
      
      pat.end <- round(pat.data$DischargeWithUpdates, "hours")
      pat.start <- round(pat.data$Arrival_Time_9, "hours")

      pat.date.sequence <- seq(pat.start, pat.end, by="1 hour")
      
      reg.num <- pat.data$Pt_Accnt_5
      total <- sum(na.exclude(pat.per.min[as.character(pat.date.sequence)]))
      at.start <- as.numeric(pat.per.min[as.character(pat.date.sequence[1], "%Y-%m-%d %H:%M:%S")])
      pat.num.ones <- sum(na.exclude(num.ones[as.character(pat.date.sequence)]))
      pat.num.twos <- sum(na.exclude(num.twos[as.character(pat.date.sequence)]))
      pat.num.threes <- sum(na.exclude(num.threes[as.character(pat.date.sequence)]))
      pat.num.fours <- sum(na.exclude(num.fours[as.character(pat.date.sequence)]))
      pat.num.fives <- sum(na.exclude(num.fives[as.character(pat.date.sequence)]))
      pat.num.nines <- sum(na.exclude(num.nines[as.character(pat.date.sequence)]))
      
      
      
      
      
      total.patients.in.er[pat.num,] <- c(RegistrationNumber=reg.num, 
                                          NumberOfPatientsAtStart=at.start, 
                                          TotalNumberOfPatients=total,
                                          NumberCTAS1=pat.num.ones, 
                                          NumberCTAS2=pat.num.twos, 
                                          NumberCTAS3=pat.num.threes, 
                                          NumberCTAS4=pat.num.fours,
                                          NumberCTAS5=pat.num.fives, 
                                          NumberCTAS9=pat.num.nines)
      
      
      
      
    } else {
      
      total.patients.in.er[pat.num,] <- c(RegistrationNumber=as.character(pat.data$Pt_Accnt_5), 
                                          NumberOfPatientsAtStart=NA, 
                                          TotalNumberOfPatients=NA,
                                          NumberCTAS1=NA, 
                                          NumberCTAS2=NA, 
                                          NumberCTAS3=NA, 
                                          NumberCTAS4=NA,
                                          NumberCTAS5=NA, 
                                          NumberCTAS9=NA)
      
    }
    
    
  } else {
    total.patients.in.er[pat.num,] <- c(RegistrationNumber=as.character(pat.data$Pt_Accnt_5), 
                                        NumberOfPatientsAtStart=NA, 
                                        TotalNumberOfPatients=NA,
                                        NumberCTAS1=NA, 
                                        NumberCTAS2=NA, 
                                        NumberCTAS3=NA, 
                                        NumberCTAS4=NA,
                                        NumberCTAS5=NA, 
                                        NumberCTAS9=NA)
  }
  
}

y <- Sys.time()
print(paste("End time:", y))
print(paste("It took", round(difftime(y, x, units='hours'), 3), "hours to complete"))

fwrite(total.patients.in.er, paste0(path, "flowVars/flow_var_per_patient_wellSoft.csv"))



