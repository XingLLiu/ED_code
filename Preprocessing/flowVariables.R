# Date: July 29th, 2019

library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)

# Calculate number of patients in emergency department for every 15 minute interval of every year
# from July 2018 - present based on EPIC Arrived (start time) and Disch.Date.Time (end time)

## NOTE !!  This is currently run using Disch.Date.Time --> this is discharge time from hospital, NOT ED
##          NEED TO RERUN THIS WITH ACTUAL DISCHARGE TIME! 
##          Alternatively, can rerun with Arrived + ED.Completed.Length.of.Stay..Minutes.

# Input: EPIC.csv
#
# Output: -- flowStats.csv: For each patient visit, the following is calculated: 
#                             - Number of patients at start of visit
#                             - Total number of patients during duration of entire visit
#                             - CTAS1-5 patients during duration of entire visit
#
#         -- flow_data.csv: For each 15min interval from July 1st 2018 to present, the following is calculated
#                             - Total number of patients during duration of 15min interval
#                             - Number CTAS1-5 patients during duration of 15min interval



path <- "./data/EPIC_DATA/"


# ===================== 1. Load Data =====================
EPIC <- fread(paste0(path, "EPIC.csv"))

# ===================== 2. Preprocess =====================
EPIC$Arrived <- as.POSIXct(EPIC$Arrived, tz="EST", format="%d/%m/%y %H%M")
EPIC <- EPIC[order(EPIC$Arrived),]
EPIC <- EPIC[!is.na(EPIC$Arrived),]
EPIC$Disch.Date.Time <- as.POSIXct(EPIC$Disch.Date.Time, tz="EST", format="%d/%m/%Y %H%M")


# ===================== 3. Create 15min accumulators for entire length of stays =====================
first.time <- lubridate::round_date(EPIC$Arrived[1], unit="15 minutes"); 
last.time <- lubridate::round_date(EPIC$Disch.Date.Time[length(EPIC$Disch.Date.Time)-1], unit="15 minutes")
print(paste("First Arrival Time:", first.time, "Last Discharge Time:", last.time))


date.sequence <- seq(first.time, last.time, by="15 min"); head(date.sequence); tail(date.sequence)

pat.per.min <- seq(0, 0, length=length(date.sequence))
num.ones <- seq(0, 0, length=length(date.sequence)) # collect number of patients in ED with certain CTAS Scores
num.twos <- seq(0, 0, length=length(date.sequence))
num.threes <- seq(0, 0, length=length(date.sequence))
num.fours <- seq(0, 0, length=length(date.sequence))
num.fives <- seq(0, 0, length=length(date.sequence))


names(pat.per.min) <- date.sequence
names(num.ones) <- date.sequence; names(num.twos) <- date.sequence; names(num.threes) <- date.sequence; names(num.fours) <- date.sequence
names(num.fives) <- date.sequence
#
x <-Sys.time()
print(paste("Start time:", x))
for (pat.num in 1:nrow(EPIC)) { # 
  pat.data <- EPIC[pat.num,]
  print(paste("Patient", pat.num, "out of", nrow(EPIC)))
  if (!is.na(pat.data$ED.Completed.Length.of.Stay..Minutes.)) {
      
      pat.end <- pat.data$Arrived + (pat.data$ED.Completed.Length.of.Stay..Minutes.*60) ## CAN CHANGE ONCE ACTUAL TIME STAMP IS FOUND
      pat.start <- pat.data$Arrived
      
      # --> If use Disch.Date.Time --> 6 patients have discharge times after arrival --> can't have - length of stays!
      # if (pat.start > pat.end) {# start time occurs AFTER end time
      #   pat.end <- round(pat.start + (pat.data$LengthOfStayInMinutes * 60), "mins")
      # }
      
      
      pat.start <- lubridate::round_date(pat.start, unit="15 minutes")
      pat.end <- lubridate::round_date(pat.end, unit="15 minutes")
      
      pat.date.sequence <- seq(pat.start, pat.end, by="15 min")
      
      # incremement all 15min intervals where patient was in ED for
      pat.per.min[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] <- pat.per.min[as.character(pat.date.sequence, "%Y-%m-%d %H:%M:%S")] + 1
      # next, record the CTASScore of patients
      
      if (!is.na(pat.data$CTAS)) {
        pat.CTASScore <- as.numeric(pat.data$CTAS)
        
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
        } 
        
      }
    
  }
  

  
}

y <- Sys.time()
print(paste("End time:", y))
print(paste("It took", round(difftime(y, x, units='hours'), 3), "hours to complete"))

all.data <- rbind(pat.per.min, num.ones, num.twos, num.threes, num.fours, num.fives)
all.data <- cbind("Type"=c("Total", "NumOnes", "NumTwos", "NumThrees", "NumFours", "NumFives"), all.data)
fwrite(x = all.data, "./flow_data.csv")


# ===================== 4. Calculate data per patient =====================
# Now, calculate statistics per patient


total.patients.in.er <- data.frame(matrix(0, nrow(EPIC), 9))
colnames(total.patients.in.er) <- c("CSN", "MRN", "NumberOfPatientsAtStart", "TotalNumberOfPatients",
                                    "NumberCTAS1", "NumberCTAS2", "NumberCTAS3", "NumberCTAS4",
                                    "NumberCTAS5")

x <-Sys.time()
print(paste("Start time:", x))
for (pat.num in 1:nrow(EPIC)) { 
  pat.data <- EPIC[pat.num,]
  print(paste("Patient", pat.num, "out of", nrow(EPIC)))
  
  if (!is.na(pat.data$ED.Completed.Length.of.Stay..Minutes.)) {
    
    pat.end <- pat.data$Arrived + (pat.data$ED.Completed.Length.of.Stay..Minutes.*60) ## CAN CHANGE ONCE ACTUAL TIME STAMP IS FOUND
    pat.start <- pat.data$Arrived
    
    pat.start <- lubridate::round_date(pat.start, unit="15 minutes")
    pat.end <- lubridate::round_date(pat.end, unit="15 minutes")
    
    pat.date.sequence <- seq(pat.start, pat.end, by="15 min")
    
    
    pat.CSN <- pat.data$CSN; pat.MRN <- pat.data$MRN
    
    total <- sum(na.exclude(pat.per.min[as.character(pat.date.sequence)]))
    at.start <- as.numeric(pat.per.min[as.character(pat.date.sequence[1], "%Y-%m-%d %H:%M:%S")])
    pat.num.ones <- sum(na.exclude(num.ones[as.character(pat.date.sequence)]))
    pat.num.twos <- sum(na.exclude(num.twos[as.character(pat.date.sequence)]))
    pat.num.threes <- sum(na.exclude(num.threes[as.character(pat.date.sequence)]))
    pat.num.fours <- sum(na.exclude(num.fours[as.character(pat.date.sequence)]))
    pat.num.fives <- sum(na.exclude(num.fives[as.character(pat.date.sequence)]))

      
      
    total.patients.in.er[pat.num,] <- c(CSN=pat.CSN,
                                        MRN=pat.MRN,
                                        NumberOfPatientsAtStart=at.start, 
                                        TotalNumberOfPatients=total,
                                        NumberCTAS1=pat.num.ones, NumberCTAS2=pat.num.twos, 
                                        NumberCTAS3=pat.num.threes, NumberCTAS4=pat.num.fours,
                                        NumberCTAS5=pat.num.fives)
      
      
      
      
    } else {
      
      total.patients.in.er[pat.num,] <- c(CSN=pat.data$CSN,
                                          MRN=pat.data$MRN,
                                          NumberOfPatientsAtStart=NA, 
                                          TotalNumberOfPatients=NA,
                                          NumberCTAS1=NA, NumberCTAS2=NA, NumberCTAS3=NA, NumberCTAS4=NA,
                                          NumberCTAS5=NA)
      
    }
 
}

y <- Sys.time()
print(paste("End time:", y))
print(paste("It took", round(difftime(y, x, units='hours'), 3), "hours to complete"))

fwrite(x = total.patients.in.er, "./flowStats.csv")

# ===================== 5. Visualize CTAS scores over time =====================

#all.data <- fread(paste0(path, "flow_data.csv"))
head(all.data[,1:5])
times <- colnames(all.data[,2:ncol(all.data)])
times <- as.POSIXct(times, format="%Y-%m-%d %H:%M:%S", tz="EST")

all.data <- t(all.data)
colnames(all.data) <- all.data[1,]
all.data <- all.data[2:nrow(all.data),]
all.data <- data.frame(all.data)
all.data$Arrived <- times
rownames(all.data) <- NULL
head(all.data)

all.data$Month <- month.name[month(all.data$Arrived)]; head(all.data)
all.data$NumOnes <-as.numeric(as.character(all.data$NumOnes))
all.data$NumTwos <- as.numeric(as.character(all.data$NumTwos))
all.data$NumThrees <-  as.numeric(as.character(all.data$NumThrees))
all.data$NumFours <-  as.numeric(as.character(all.data$NumFours))
all.data$NumFives <-  as.numeric(as.character(all.data$NumFives))
all.data$Total <- as.numeric(as.character(all.data$Total))


ggplot() + 
  geom_line(data=all.data, aes(x=Arrived, y=NumFives, colour="CTAS5")) +
  geom_line(data=all.data, aes(x=Arrived, y=NumFours, colour="CTAS4")) +
  geom_line(data=all.data, aes(x=Arrived, y=NumThrees, colour="CTAS3")) +
  geom_line(data=all.data, aes(x=Arrived, y=NumTwos, colour="CTAS2")) + 
  geom_line(data=all.data, aes(x=Arrived, y=NumOnes, colour="CTAS1")) + 
  ylab("Number of Patients") + 
  ggtitle("Number of Patients Arriving per 15min Interval by CTAS Score") + 
  theme_bw()

