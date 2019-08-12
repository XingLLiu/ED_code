# Date: July 31st, 2019

# Calculate number of visits to the ED per person, followed by the time differences between visits. 
# Used to calculate returns within 72h

# Input: wellSoft
#           - needs to have PrimaryMedicalRecordNumber (i.e. joined with reg codes)

# output: timeLapse.rds with
#               - Primary Medical Record Number
#               - Registration Number for Visit 1
#               - Registration Number for Visit 2
#               - Difference in days between Visit 1 and Visit 2


# ================== CALCULATE NUMBER OF VISITS PER PERSON ================== # 
calculateTimeLapse <- function(wellSoft, reg_codes, data.path) {
  print("Merging data")
  all_data <- merge(x=wellSoft, 
                    y=reg_codes[,c("PrimaryMedicalRecordNumber", "DischargeDisposition", "RegistrationNumber")],
                    by.x=c("Pt_Accnt_5"),
                    by.y=c("RegistrationNumber"))
  fwrite(all_data, paste0(data.path, "all_data.csv"))
  print(paste("Lost", nrow(wellSoft) - nrow(all_data), "when merging wellSoft and registration codes"))
  
  if (is.character(all_data$Arrival_Time_9)) { 
    all_data$Arrival_Time_9 <- as.POSIXct(all_data$Arrival_Time_9, tz = "EST") 
  }
  
  all_data$Discharge_Time_With_Updates <- ifelse(all_data$Updated_Discharge_Time_596 != "", 
                                                 all_data$Updated_Discharge_Time_596, all_data$Discharge_Time_276)
  all_data$Discharge_Time_With_Updates[all_data$Discharge_Time_With_Updates==""] <- NA
  if (is.character(all_data$Discharge_Time_With_Updates)) { 
    all_data$Discharge_Time_With_Updates <- as.POSIXct(all_data$Discharge_Time_With_Updates, tz = "EST") 
  }
  
  
  
  print("Caclulating Number of Returns")
  num.returns <- data.frame(all_data %>% dplyr::filter(!DischargeDisposition %in% to.remove) %>% 
                                   dplyr::group_by(PrimaryMedicalRecordNumber) %>%
                                   dplyr::mutate(num.returns = n()))
  
  num.returns <- num.returns %>% dplyr::arrange(desc(num.returns), 
                                                          PrimaryMedicalRecordNumber, Arrival_Time_9)
  sum(num.returns[!duplicated(num.returns$PrimaryMedicalRecordNumber),]$num.returns)
  num.returns$num.returns <- num.returns$num.returns - 1
  single.visit.ids <- (num.returns %>% dplyr::filter(num.returns == 0))
  single.visit.ids <- single.visit.ids$PrimaryMedicalRecordNumber
  
  
  order.returns <- num.returns %>% select("Pt_Accnt_5", "PrimaryMedicalRecordNumber",
                                        "Arrival_Time_9", "Discharge_Time_With_Updates")
  num.returns <- num.returns[!duplicated(num.returns$PrimaryMedicalRecordNumber),
                                       c("PrimaryMedicalRecordNumber", "num.returns")]
  
  N.visits <- sum(num.returns$num.returns+1)
  mean.visits <- mean(num.returns$num.returns+1); median.visits <- median(num.returns$num.returns+1)
  unique.PMRN <- num.returns$PrimaryMedicalRecordNumber
  
  print(paste("There were", N.visits, "unique patients between", min(all_data$Arrival_Time_9, na.rm=T), "and", max(all_data$Arrival_Time_9, na.rm=T)))
  print(paste("The average number of visits was", mean.visits))
  print(paste("The median number of visits was", median.visits))
  
  
  
  # Number of visits per person 
  
  rel.ids <- unique.PMRN[!unique.PMRN %in% single.visit.ids]
  
  stopifnot((length(rel.ids) + length(single.visit.ids)) == length(unique.PMRN))
  

  num.rows <- sum(((num.returns %>% filter(PrimaryMedicalRecordNumber %in% rel.ids))$num.returns)+1) - length(rel.ids)
  
  time.lapse <- data.frame(matrix(ncol = 4, nrow = (num.rows)))

  #time.lapse <- data.frame(matrix(ncol = 4, nrow = 0))
  colnames(time.lapse) <- c("PrimaryMedicalRecordNumber", "RegistrationNumberVisit1", 
                            "RegistrationNumberVisit2", "DifferenceInDays")
  print("Process return visits")


  j <- 1
  for (i in 1:length(rel.ids)) {
    patient.id <- rel.ids[i]; 
    if (i %% 100 == 0) {print(paste("Patient", i, "out of", length(rel.ids)))}
    visits <- order.returns %>% dplyr::filter(PrimaryMedicalRecordNumber %in% c(patient.id))
    visits <- visits %>% dplyr::arrange(Arrival_Time_9)
    
    differences <- difftime(visits[2:nrow(visits), c("Arrival_Time_9")], 
                            visits[1:(nrow(visits)-1), c("Discharge_Time_With_Updates")], units="days")
    visits.reg <- visits$Pt_Accnt_5
    visit.1.reg.num <- visits.reg[1:(length(visits.reg)-1)]; 
    visit.2.reg.num <- visits.reg[2:length(visits.reg)];
    

    time.lapse[j:(j+length(differences)-1),] <- c(rep_len(patient.id, length(differences)),
						 as.character(visit.1.reg.num),
					 	 as.character(visit.2.reg.num),
						 differences)

    j <- j + length(differences)


    
    
  }

  print("Saving timeLapse.csv")
  fwrite(time.lapse, paste0(data.path, "timeLapse.csv"))
  #saveRDS(time.lapse, "timeLapse.rds")
  
  return(time.lapse)
  
  
}


# data.path <- "./data/wellSoft_DATA/"
# source(paste0(data.path, 'dischargeCategories.R'))
# library(data.table)
# library(dplyr)
# 
# reg_codes <- fread(paste0(data.path, "processedRegistrationCodes.csv"))
# wellSoft <- fread(paste0(data.path, "cleaned_wellSoft.csv"))
# timeLapse <- calculateTimeLapse(wellSoft, reg_codes, data.path)
