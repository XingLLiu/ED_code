# NEED TO CHECK

# May 27th, 2019

# ================== PATIENT FLOW THROUGH SYSTEM ==================


# ================== 1. Load and preprocess data ==================

source("dischargeCategories.R")

path <- "/home/lebo/data/lebo/data/"

wellSoft <- fread(paste0(path, "cleaned_wellSoft.csv"))
reg_codes <- fread(paste0(path, "RegistrationCodes.csv"))

willReturn <- fread(paste0(path, "willReturn.csv"))

dim(wellSoft); colnames(wellSoft)
dim(reg_codes); colnames(reg_codes)
dim(willReturn); head(willReturn)

wellSoft$Arrival_Time_9 <- as.POSIXct(strptime(wellSoft$Arrival_Time_9, format="%Y-%m-%d %H:%M:%S"), tz="EST")
head(wellSoft$Arrival_Time_9)
wellSoft$Discharge_Time_276 <- as.POSIXct(strptime(wellSoft$Discharge_Time_276, format="%Y-%m-%d %H:%M:%S"), tz="EST")
wellSoft$Year <- format(wellSoft$Arrival_Time_9, "%Y")

wellSoft$Age_Dob_40 <- as.POSIXct(strptime(wellSoft$Age_Dob_40, format="%Y-%m-%d %H:%M:%S"), tz="EST")
wellSoft$DaysOld <- as.numeric(round(difftime(wellSoft$Arrival_Time_9, wellSoft$Age_Dob_40, units='days'), 1))
wellSoft$DaysOld[wellSoft$DaysOld < 0] <- NA
head(wellSoft$DaysOld)

reg_codes$VisitStartDate <- convertToDate(reg_codes$VisitStartDate)
attr(reg_codes$VisitStartDate, "tzone") <- "EST"
head(reg_codes$VisitStartDate)

reg_codes$StartOfVisit <- convertToDateTime(reg_codes$StartOfVisit)
attr(reg_codes$StartOfVisit, "tzone") <- "EST"
reg_codes$EndOfVisit <- convertToDateTime(reg_codes$EndOfVisit)
attr(reg_codes$EndOfVisit, "tzone") <- "EST"

all.wellSoft <- merge(x=wellSoft, 
                  y=reg_codes, 
                  by.x=c("Pt_Accnt_5"),
                  by.y=c("RegistrationNumber"))

nrow(wellSoft); nrow(all.wellSoft)


# ================== 2. Overview of All Numbers ==================


# left before seen by doctor
left.before.seen.num <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% left.labels)); left.before.seen.num # number
left.before.seen.per <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% left.labels)) / nrow(all.wellSoft); round(left.before.seen.per*100,2) # percentage


# seen by doctor
seen.by.doc.num <- nrow(all.wellSoft %>% filter(!DischargeDisposition %in% left.labels)); seen.by.doc.num  # number
seen.by.doc.per <- nrow(all.wellSoft %>% filter(!DischargeDisposition %in% left.labels)) / nrow(all.wellSoft); round(seen.by.doc.per*100, 2) # percentage


# admitted to ED
admit.to.ed.num <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c("ADMIT TO HOLD - SENT HOME",
                                                     "SENT HOME-EMERG. ADMIT"))); admit.to.ed.num


admit.to.ed.per <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c("ADMIT TO HOLD - SENT HOME",
                      "SENT HOME-EMERG. ADMIT"))) / seen.by.doc.num; round(admit.to.ed.per*100, 2)


# admitted to sick kids
admit.hsk.num <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c("ADMITTED TO HSC", # Admitted to Sick Kids
                                                      "ADMIT VIA O.R.", 
                                                      "ADMITTED TO CCU", 
                                                      "ADMITTED VIA O.R.",
                                                      "ADMITTED TO CCU OR THE O.R.",
                                                    "TRANSFERRED-EMERG. ADMIT"))); admit.hsk.num

admit.hsk.per <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c("ADMITTED TO HSC", # Admitted to Sick Kids
                         "ADMIT VIA O.R.", 
                         "ADMITTED TO CCU", 
                         "ADMITTED VIA O.R.",
                         "ADMITTED TO CCU OR THE O.R.",
                         "TRANSFERRED-EMERG. ADMIT"))) / seen.by.doc.num; round(admit.hsk.per*100, 2)



# admitted to another institution
other.inst.num <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c("TRANSFER TO ANOTHER INSTITUTION", # sent to another institution
                                                    "OTHER INSTITUTION",
                                                    "Transfer to another facility",
                                                    "ADMIT TO HOLD - TRANSFERRED"))); other.inst.num

other.inst.per <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c("TRANSFER TO ANOTHER INSTITUTION", # sent to another institution
                          "OTHER INSTITUTION",
                          "Transfer to another facility",
                          "ADMIT TO HOLD - TRANSFERRED"))) / seen.by.doc.num; round(other.inst.per*100,2)

# held for intervention
held.int.num <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c("SENT HOME VIA O.R.", # Held for intervention then sent home 
                                                    "TRNSFR TO D/S",
                                                    "TRANSFER TO DAY SURGERY"))); held.int.num  

held.int.per <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c("SENT HOME VIA O.R.", # Held for intervention then sent home 
                                "TRNSFR TO D/S",
                                "TRANSFER TO DAY SURGERY"))) /seen.by.doc.num; round(held.int.per*100,2)


# died
died.num <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c("DEATH AFTER ARRIVAL", # Death
                                                    "DEAD ON ARRIVAL",
                                                    "EXPIRED",
                                                    "ADMIT TO HOLD - EXP."))); died.num

died.per <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c("DEATH AFTER ARRIVAL", # Death
                                      "DEAD ON ARRIVAL",
                                      "EXPIRED",
                                      "ADMIT TO HOLD - EXP."))) / seen.by.doc.num; round(died.per*100, 2)



# sent home
sent.home.num <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c(home.labels))); sent.home.num

sent.home.per <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c(home.labels))) / seen.by.doc.num; round(sent.home.per*100,2)

# other
other.num <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c(to.remove))); other.num
other.per <- nrow(all.wellSoft %>% filter(DischargeDisposition %in% c(to.remove))) / seen.by.doc.num; round(other.per*100, 2)


# check
admit.to.ed.per + admit.hsk.per + other.inst.per + held.int.per + died.per + sent.home.per + other.per # == 100

seen.by.doc.num
admit.to.ed.num + admit.hsk.num + other.inst.num + held.int.num + died.num + sent.home.num + other.num  == seen.by.doc.num



# ================== 3. BASED ON PATHS ==================

# Date: July 31st, 2018
# 
# Produces all the statistics used in the power point, of number of patients
# and number of unique visits in patient pipeline

# NOTE: Depends on processTrainTestSet having been run and 
# variables stored in environemnt

#source("processTrainTestSet.R")



# all visits
N <- nrow(return.visits) + length(single.visit.ids.all); N

# Of the total, how many: 

# Leave before being seen: no return index leave; return index leave; single visit leave 
index.leave <- nrow(no.return.within.72.index.left) + nrow(return.within.72.index.left) + 
  length(single.visit.ids.left); index.leave
index.leave/ N

# Go on to see a doctor 
index.see.doc <- N - index.leave; index.see.doc
index.see.doc / N

# sanity
nrow(no.return.within.72.index.necessary) + nrow(no.return.within.72.index.sent.home) + 
  nrow(no.return.within.72.index.left) + nrow(return.within.72.index.left) + 
  nrow(return.within.72.index.necessary) + nrow(return.within.72.index.sent.home) + 
  length(single.visit.ids.necessary.visit) + length(single.visit.ids.sent.home) +
  length(single.visit.ids.left) == N


# santiy
index.leave + index.see.doc == N

# Of those that see doctors at index visit, how many: 

# are necessary visits : visits with no returns which are necessary; returns with index necessary; single necessary visits

index.necc <- nrow(no.return.within.72.index.necessary)  + 
  nrow(return.within.72.index.necessary) + 
  length(single.visit.ids.necessary.visit); index.necc
index.necc / index.see.doc

# are sent home : no returns index sent home; return visits index sent home; single visits sent home; 
index.sent.home <-  nrow(no.return.within.72.index.sent.home) + 
  nrow(return.within.72.index.sent.home) +
  length(single.visit.ids.sent.home); index.sent.home
index.sent.home / index.see.doc


# sanity
index.necc + index.sent.home == index.see.doc


calculateDataStatistics <- function(index.df, index.num) {
  
  # index.df : return.within.72.index.necessary
  # index.num: index.necc
  
  # Return within 72
  
  return.all <- nrow(index.df)
  
  # Return and leave before being seen: return wihtin 72 index necessary and leave
  return.left <- nrow(index.df %>% filter(DischargeDisposition %in% left.labels))
  
  # Return and see doctor
  see.doc <- return.all - return.left
  
  # Return and are necessary returns: return wihtin 72 and index necessary and necessary
  necc <- nrow(index.df %>% filter(DischargeDisposition %in% admitted.labels))
  
  # Return and are sent home
  sent.home <-nrow(index.df %>% filter(DischargeDisposition %in% home.labels))
  
  # Never return
  never.return <- index.num - return.all
  
  cat(paste0("\nNever return: ", never.return, "; ", round((never.return / index.num)*100, 2),
             "\nReturn: ", return.all, "; ", round((return.all / index.num)*100, 2),
             "\nLeave before being seen: ", return.left, "; ", round((return.left / return.all)*100, 2),
             "\nSee Doctor: ", see.doc, "; ", round((see.doc / return.all)*100, 2),
             "\nAdmitted: ", necc, "; ", round((necc / see.doc)*100, 2),
             "\nNon-Admitted:", sent.home, "; ", round((sent.home / see.doc)*100, 2)))
}


# Of the index visits that are necessary:
calculateDataStatistics(return.within.72.index.necessary, index.necc)


# Of the index visits sent home:
calculateDataStatistics(return.within.72.index.sent.home, index.sent.home)


# Of the index visits that left before being seen: 
calculateDataStatistics(return.within.72.index.left, index.leave)





# FOR UNIQUE PATIENTS
# same as above, except each patient is restricted to a single visit (for now)


N.unique <- length(single.visit.ids.all) + nrow(returned.within.72.unique) + nrow(no.return.within.72.unique); N.unique

# Leave before being seen: no return index leave; return index leave; single visit leave 
index.leave.unique <- nrow(no.return.within.72.unique.index.left) + nrow(returned.within.72.unique.index.left) + 
  length(single.visit.ids.left); index.leave.unique
index.leave.unique / N.unique

# Go on to see a doctor 
index.see.doc.unique <- N.unique - index.leave.unique; index.see.doc.unique
index.see.doc.unique / N.unique

# santiy
index.leave.unique + index.see.doc.unique == N.unique


# Of those that see doctors at index visit, how many: 

# are necessary visits : visits with no returns which are necessary; returns with index necessary; single necessary visits

index.necc.unique <- nrow(no.return.within.72.unique.index.admit)  + 
  nrow(returned.within.72.unique.index.admit) + 
  length(single.visit.ids.necessary.visit); index.necc.unique
index.necc.unique / index.see.doc.unique

# are sent home : no returns index sent home; return visits index sent home; single visits sent home; 
index.sent.home.unique <-  nrow(no.return.within.72.unique.index.sent.home) + 
  nrow(returned.within.72.unique.index.sent.home) +
  length(single.visit.ids.sent.home); index.sent.home.unique
index.sent.home.unique / index.see.doc.unique


# sanity
index.necc.unique + index.sent.home.unique == index.see.doc.unique


# Of the index visits that are necessary:
calculateDataStatistics(retreiveReturnVisitData(returned.within.72.unique, admitted.labels), index.necc.unique)


# Of the index visits sent home:
calculateDataStatistics(retreiveReturnVisitData(returned.within.72.unique, home.labels), index.sent.home.unique)


# Of the index visits that left before being seen: 
calculateDataStatistics(retreiveReturnVisitData(returned.within.72.unique, left.labels), index.leave.unique)



