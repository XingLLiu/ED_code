# July 11th 2019
# Preprocesses EPIC.csv (combined raw data) and outputs preprocessed_EPIC.csv (ready for training)

formatTimes <- function(data) {
  # Feature engineering based on arrival times
  data$Arrived <- as.POSIXct(EPIC$Arrived, format="%d/%m/%Y %H%M", tz = "EST")
  data$ArrivalMonth <- month.name[as.numeric(format(data$Arrived, "%m"))]
  data$ArrivalNumHoursSinceMidnight <- hour(as.ITime(data$Arrived))
  data$ArrivalNumHoursSinceMidnight[data$ArrivalNumHoursSinceMidnight < 0] <- NA
  data$ArrivalDayOfWeek <- weekdays(data$Arrived)
  data$Disch.Date.Time <- as.character(hour(as.POSIXct(data$Disch.Date.Time, 
                                            format="%d/%m/%Y %H%M", tz = "EST")))
  data$Arrival.to.Room <-hour(strptime(data$Arrival.to.Room, "%H:%M"))
  data$Roomed <- as.character(hour(as.POSIXct(data$Roomed, 
                                              format="%d/%m %H%M", tz = "EST")))
  
  return (data)
  
}

createReturnInd <- function(data) {
  repeats <- data[,c("MRN", "Arrived", "CSN", "Dispo")] %>% filter(MRN %in% c(data$MRN[duplicated(data$MRN)]))
  repeats <- repeats[order(repeats$Arrived),]
  repeats <- repeats %>% group_by(MRN) %>% mutate(diff=difftime(Arrived, lag(Arrived), units="hours"))
  repeats$WillReturn <- repeats$diff <= 72
  repeats$WillReturn[is.na(repeats$WillReturn)] <- 0 # treat those with NA for 'WillReturn' as not return
  
  # filter those who will return to only those who were discharged at initial visit
  repeats <- repeats[repeats$WillReturn==1 & repeats$Dispo=="Discharge",] 
  repeat.encounter.numbers <- repeats$CSN[repeats$WillReturn==1]
  data$WillReturn <- 0
  data$WillReturn[data$CSN %in% repeat.encounter.numbers] <- 1
  return (data)
}


path <- "../../ED/data/EPIC_DATA/"

# ============ 1. LOAD DATA ================= #
EPIC <- fread(paste0(path, "EPIC.csv"))

# ============ 2. PROCESS DATES ================= #
EPIC <- formatTimes(EPIC)

# ============ 3. CREATE RETURN INDICATOR ================= #
EPIC <- createReturnInd(EPIC)

# ============ 4. CLEAN FACTORS ================= #

# I. FEATURE ENGINEERING

# a. indicator: whether or not patient had the same first and last ED provider (capturing change of provider shifts)
EPIC$SameFirstAndLast <- ifelse(EPIC$First.ED.Provider == EPIC$Last.ED.Provider , 1, 0)

# b. numeric: size of treatment team (NOTE: could be duplicate names, not accounted for in code)
EPIC$SizeOfTreatmentTeam <- unlist( lapply( EPIC$Treatment.Team, function(x) length( unlist( strsplit(x, ";") ) ) ) )

# c. numeric: number of prescriptions
EPIC$NumberOfPrescriptions <- unlist( lapply( EPIC$Current.Medication, function(x) length( unlist( strsplit(x, ";") ) ) ) )

# d. numeric: length of stay in minutes
EPIC$LengthOfStayInMinutes <- (as.numeric(EPIC$ED.Completed.Length.of.Stay..Hours.)*60) + as.numeric(EPIC$ED.Completed.Length.of.Stay..Minutes.) 


# II. RESTRICT SIZE OF FACTOR FOR 1-hot ENCODING:

# a. languages: top 50
top.langs <- names(sort(table(EPIC$Pref.Language), decreasing = TRUE))[1:50]
EPIC$Pref.Language[!EPIC$Pref.Language %in% top.langs ] <- "Other"

# b. presenting complaint: top 50
top.cc <- names(sort(table(EPIC$CC), decreasing = TRUE))[1:50]
EPIC$CC[!EPIC$CC %in% top.cc ] <- "Other"

# c. primary diagnoses: top 49 + Sepsis or Sepsis-related
top.dx <- names(sort(table(EPIC$Primary.Dx), decreasing = TRUE))[1:50]
top.dx[length(top.dx)] <- 'Sepsis or related'
# Find all Sepsis or related cases
if.Sepsis <- str_detect(EPIC$Primary.Dx, 'Sepsis') | str_detect(EPIC$Primary.Dx, 'sepsis')
# Rename these cases
EPIC$Primary.Dx[if.Sepsis] <- 'Sepsis or related'
# Rename other cases
EPIC$Primary.Dx[!EPIC$Primary.Dx %in% top.dx ] <- "Other"

# d. first ED provider: top 50
top.first.ed <- names(sort(table(EPIC$First.ED.Provider), decreasing = TRUE))[1:50]
EPIC$First.ED.Provider[!EPIC$First.ED.Provider %in% top.first.ed ] <- "Other"

# d. last ED provider: top 50
top.last.ed <- names(sort(table(EPIC$Last.ED.Provider), decreasing = TRUE))[1:50]
EPIC$Last.ED.Provider[!EPIC$Last.ED.Provider %in% top.last.ed ] <- "Other"

# e. longest ED provider: top 50
top.longest <- names(sort(table(EPIC$ED.Longest.Attending.ED.Provider), decreasing = TRUE))[1:50]
EPIC$ED.Longest.Attending.ED.Provider[!EPIC$ED.Longest.Attending.ED.Provider %in% top.longest ] <- "Other"

# f. FSA (first 3 letters of Postal code): top 50
EPIC$FSA <- str_extract(EPIC$Address, "[A-Z][0-9][A-Z]")
top.FSAs <- names(sort(table(EPIC$FSA), decreasing = TRUE))[1:50]
EPIC$FSA[!EPIC$FSA %in% top.FSAs ] <- "Other"

# g. Admitting provider: top 50
Admitting.provider <- names(sort(table(EPIC$Admitting.Provider), decreasing = TRUE))[1:50]
EPIC$Admitting.Provider[!EPIC$Admitting.Provider %in% Admitting.provider ] <- "Other"

# III. CLEANING UP:

# a. synthesize arrival methods
EPIC$Arrival.Method[grep("Other", EPIC$Arrival.Method)] <- "Other"
EPIC$Arrival.Method[grep("Ambulatory", EPIC$Arrival.Method)] <- "Ambulatory"
EPIC$Arrival.Method[grep("Air & Ground Ambulance", EPIC$Arrival.Method)] <- "Air & Ground Ambulance"
EPIC$Arrival.Method[grep("Land Ambulance", EPIC$Arrival.Method)] <- "Land Ambulance"
EPIC$Arrival.Method[grep("Stretcher", EPIC$Arrival.Method)] <- "Stretcher"
EPIC$Arrival.Method[grep("Unknown", EPIC$Arrival.Method)] <- "Unknown"
EPIC$Arrival.Method[grep("Car", EPIC$Arrival.Method)] <- "Car"

# b. remove text >> numeric
EPIC$Last.Weight <- gsub("[^0-9\\.]", "", EPIC$Last.Weight)
EPIC$Last.Weight <- as.numeric(EPIC$Last.Weight)

# c. currently binarizes >> what does this column mean??
EPIC$Discharge.Admit.Time[grepl("[^No previous discharge]", EPIC$Discharge.Admit.Time)] <- "Previous Visit"

# d. >> numeric
EPIC$Pulse <- gsub("[^0-9]", "", EPIC$Pulse)
EPIC$Resp <- as.numeric(gsub("[^0-9]", "", EPIC$Resp))
EPIC$Temp <- as.numeric(gsub("[^0-9\\.]", "", EPIC$Temp))

# e. convert blood pressure to Systolic/Diastolic measures
EPIC$Systolic <- str_extract(EPIC$BP, "[0-9]{2,3}/")
EPIC$Systolic <- as.numeric(gsub("/", "", EPIC$Systolic))
EPIC$Diastolic <- str_extract(EPIC$BP, "/[0-9]{2,3}")
EPIC$Diastolic <- as.numeric(gsub("/", "", EPIC$Diastolic))

# f. remove text >> numeric
convertAge <- function(data) {
  data$Age.at.Visit <- as.character(data$Age.at.Visit)
  month.indicies <- grep("m.o", data$Age.at.Visit)
  year.indicies <- grep("y.o", data$Age.at.Visit)
  day.indicies <- grep("days", data$Age.at.Visit)
  week.indicies <- grep("wk.o", data$Age.at.Visit)
  
  data$Age.at.Visit[month.indicies] <- as.numeric(gsub("m\\.o\\.", "", data$Age.at.Visit[month.indicies]))/12
  data$Age.at.Visit[year.indicies] <- as.numeric(gsub("y\\.o\\.", "", data$Age.at.Visit[year.indicies]))
  data$Age.at.Visit[day.indicies] <- as.numeric(gsub("days", "", data$Age.at.Visit[day.indicies])) / 365
  data$Age.at.Visit[week.indicies] <- as.numeric(gsub("wk.o", "", data$Age.at.Visit[week.indicies])) / 52
  
  data$Age.at.Visit <- as.numeric(data$Age.at.Visit)
  return (data)
}

EPIC <- convertAge(EPIC)


# ============ 5. ENSURE CORRECT FACTOR TYPES ================= #

numerics <- c("Age.at.Visit",
              "Last.Weight",
              "ArrivalNumHoursSinceMidnight",
              "SizeOfTreatmentTeam",
              "NumberOfPrescriptions",
              "Pulse",
              "Systolic",
              "Diastolic",
              "Resp",
              "Temp",
              "Disch.Date.Time",
              "LengthOfStayInMinutes",
              "Arrival.to.Room",
              "Roomed",
              "Distance_To_SickKids",
              "Distance_To_Walkin",
              "Distance_To_Hospital")

# excluded for now, too missing! 
#"Door.to.PIA",
#"Door.to.Pain.Med",
#"Door.to.Doc",

factor.columns <- c("Gender",
                    "Pref.Language",
                    "Acuity",
                    "Arrival.Method",
                    "Care.Area",
                    "CC",
                    "Primary.Dx",
                    "First.ED.Provider",
                    "Last.ED.Provider",
                    "ED.Longest.Attending.ED.Provider",
                    "ED.PIA.Threshold",
                    "Day.of.Arrival",
                    "Lab.Status",
                    "Rad.Status",
                    "WillReturn",
                    "ArrivalMonth",
                    "SameFirstAndLast",
                    "FSA",
                    "Discharge.Admit.Time",
                    "Name_Of_Walkin",
                    "Name_Of_Hospital",
                    "Admitting.Provider",
                    "Dispo")

# a. Convert to Factors, Numerics
EPIC[,(factor.columns):=lapply(.SD, as.factor),.SDcols=factor.columns]
EPIC[,(numerics):=lapply(.SD, as.numeric),.SDcols=numerics]

EPIC <- EPIC[,c(factor.columns, numerics), with=F]
# b. One hot encode
#dmy <- dummyVars(" ~ .", data = EPIC)
#EPIC <- data.table(predict(dmy, newdata = EPIC))

fwrite(EPIC, paste0(path, "preprocessed_EPIC.csv"))
