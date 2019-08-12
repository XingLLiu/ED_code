#### April 16 2019

retrieveIndexVisitData <- function(time.lapse, all_data, labels) {
  
  # returns discharge disposition for subset of visits based on
  # `labels` discharge at index visit
  
  # get Discharge Dispositions for Index visits

  index.discharges <- merge(x = all_data[,c("Pt_Accnt_5", "DischargeDisposition", "PrimaryMedicalRecordNumber")],
                            y = data.frame(RegistrationNumberVisit1=time.lapse[,c("RegistrationNumberVisit1")]),
                            by.x=c("Pt_Accnt_5"),
                            by.y=c("RegistrationNumberVisit1"))
  
  # filter by those who fall into 'labels' at first visit
  altered.data <- index.discharges %>% dplyr::filter(DischargeDisposition %in% labels)
  return (altered.data)
}


retreiveReturnVisitData <- function(time.lapse, all_data, initial.lables, return.visit.labels ) {
  
  # returns discharge disposition for return visits based on
  # `initial.labels` outcomes of index visit
  # `return.labels` outcome of return visit
  #
  # Variables:
  #   time.lapse - dataframe in the form of time.lapse for a particular segment of people
  #         (.e.g retun visits within 72 h with index visit sent home)
  #   labels - one of home, admitted, remove, left (type of disposition, defined in dischargeCategories.R)
  
  altered.data <- retrieveIndexVisitData(time.lapse, all_data, initial.lables) # select people who had 'labels' Discharge Disposition at index visit
  
  # get return info for return visits that happened after 'sent home'labels' at index visit
  altered.data <- merge(x = time.lapse, # index registration number,
                        y = data.frame(Pt_Accnt_5=altered.data[,c("Pt_Accnt_5")]),
                        by.x=c("RegistrationNumberVisit1"),
                        by.y=c("Pt_Accnt_5"))
  # get data for return visit
  altered.data <- merge(x = all_data[,c("Pt_Accnt_5", "DischargeDisposition")],
                        y = altered.data,
                        by.x="Pt_Accnt_5",
                        by.y="RegistrationNumberVisit2")
  
  # filter returns by return labels
  altered.data <- altered.data %>% dplyr::filter(DischargeDisposition %in% return.visit.labels)
  
  altered.data <- data.frame(Pt_Accnt_5=altered.data$RegistrationNumberVisit1,
                             PrimaryMedicalRecordNumber=altered.data$PrimaryMedicalRecordNumber, 
                             DischargeDisposition=altered.data$DischargeDisposition)
  
  return(altered.data)
  
  
}



  
calculateReturnVisits <- function(all_data, time.lapse, diff.days, data.path) {

  all.data <- all_data[,c("PrimaryMedicalRecordNumber", "Arrival_Time_9", "DischargeDisposition", "Pt_Accnt_5"), with=FALSE]
  
  time.lapse <- time.lapse[!duplicated(time.lapse$RegistrationNumberVisit1) & !duplicated(time.lapse$RegistrationNumberVisit2),]
  #time.lapse <- timeLapse[!duplicated(timeLapse$RegistrationNumberVisit1) & !duplicated(timeLapse$RegistrationNumberVisit2),]
  all.data <- all.data[!duplicated(all.data$Pt_Accnt_5),]
  
  print(paste("There were", nrow(all.data), "total patient visits"))
  print(paste("There were", length(unique(all.data$PrimaryMedicalRecordNumber)), "unique patients who visited"))
  
  
  # group data by Primary Medical Record Number and count the number of returns for each person
  num.returns.all <- data.frame(all.data  %>% dplyr::group_by(PrimaryMedicalRecordNumber) %>%
                                       dplyr::mutate(num.returns = n()));
  
  num.returns.all <- num.returns.all %>% dplyr::arrange(desc(num.returns),
                                                                  PrimaryMedicalRecordNumber, Arrival_Time_9)
  
  single.visit.ids.all <- (num.returns.all %>% dplyr::filter(num.returns == 1))
  single.visit.ids.sent.home <- (single.visit.ids.all %>% dplyr::filter(DischargeDisposition %in% home.labels))$Pt_Accnt_5
  single.visit.ids.necessary.visit <- (single.visit.ids.all %>% dplyr::filter(DischargeDisposition %in% admitted.labels))$Pt_Accnt_5
  single.visit.ids.left <- (single.visit.ids.all %>% dplyr::filter(DischargeDisposition %in% left.labels))$Pt_Accnt_5
  single.visit.ids.all <- single.visit.ids.all$Pt_Accnt_5
  
  # sanity check
  stopifnot(length(single.visit.ids.sent.home) + length(single.visit.ids.necessary.visit) + length(single.visit.ids.left) == length(single.visit.ids.all))
  
  print(paste(length(single.visit.ids.all), "patients visited the ED exactly once"))
  print(paste("Of these,", length(single.visit.ids.sent.home), "were sent home"))
  print(paste("Of these,", length(single.visit.ids.necessary.visit), "received some intervention"))
  print(paste("Of these,", length(single.visit.ids.left), "left without being seen"))
  
  # time.lapse
  #   - contains the time differences between visits for each patient
  #   - calculated using all the registration codes
  #   - therefore, need to remove visits that do not have corresponding wellSoft Data (all.data)
  
  # remove visits that do not have corresponding wellSoft data
  return.visits <- time.lapse
  
  num.missing.days <- sum(is.na(as.integer(return.visits$DifferenceInDays)))
  
  print(paste("Of all return visits,", num.missing.days, "had missing DifferenceInDays (i.e. missing discharge time)"))

  # remove vistis with no known difference between days
  return.visits <- return.visits[!is.na(as.integer(return.visits$DifferenceInDays)),] # removes 22 with missing difference in days
  
  # sanity check
  stopifnot(length(intersect(single.visit.ids.all, return.visits$PrimaryMedicalRecordNumber)) == 0) # should be 0
  
  
  return.visits$IsReturn <- ifelse(return.visits$DifferenceInDays < diff.days, 1, 0)
  
  print(paste("Therefore, there were", nrow(return.visits), "total return visits"))
  
  # count number of previous returns per person
  return.visits$Ones <- 1
  # create number of previous visits
  return.visits <- as.data.frame(return.visits %>% dplyr::group_by(PrimaryMedicalRecordNumber) %>% dplyr::mutate(NumPreviousVisits=cumsum(Ones)))
  return.visits$NumPreviousVisits <- return.visits$NumPreviousVisits - 1
  
  
  # create number of previous return visits
  num.return.visits <- as.data.frame(return.visits %>% dplyr::filter(IsReturn == 1) %>% dplyr::group_by(PrimaryMedicalRecordNumber) %>% dplyr::mutate(NumPreviousReturnVisits=cumsum(Ones)))
  num.return.visits$NumPreviousReturnVisits <- num.return.visits$NumPreviousReturnVisits - 1
  num.return.visits$NumPreviousVisits <- NULL
  return.visits <- dplyr::left_join(x = return.visits,
                                    y = num.return.visits)
  
  
  # for each new MRN, set the number of previous visits to -1
  first.rows <- return.visits[!duplicated(return.visits$PrimaryMedicalRecordNumber),]
  first.rows$NumPreviousReturnVisits[is.na(first.rows$NumPreviousReturnVisits)] <- -1
  return.visits$NumPreviousReturnVisits[return.visits$RegistrationNumberVisit1 %in% first.rows$RegistrationNumberVisit1] <- first.rows$NumPreviousReturnVisits
  
  # for each visit, set the NumPreviousReturnVisits to be the value that preceeds it (the row above it) + 1
  return.visits <- as.data.frame(return.visits %>%
                                   dplyr::mutate(NumPreviousReturnVisits = na.locf(NumPreviousReturnVisits) + 1)); 
  
  # makes correction --> at time of return, don't count that visit as a previous return visit
  return.visits$NumPreviousReturnVisits[return.visits$IsReturn==1] <- return.visits$NumPreviousReturnVisits[return.visits$IsReturn==1] - 1
  return.visits$Ones <- NULL
  
  returned.within.72 <- return.visits %>% dplyr::filter(IsReturn == 1)
  no.return.within.72 <- return.visits %>% dplyr::filter(IsReturn == 0)
  
  stopifnot((nrow(returned.within.72) + nrow(no.return.within.72)) ==nrow(return.visits))
  
  index.ids.return.within.72.index.sent.home <- retrieveIndexVisitData(returned.within.72, all.data, home.labels)
  index.ids.return.within.72.index.admit <- retrieveIndexVisitData(returned.within.72, all.data, admitted.labels)
  index.ids.return.within.72.index.left <- retrieveIndexVisitData(returned.within.72, all.data, left.labels)
  
  stopifnot((nrow(index.ids.return.within.72.index.sent.home) + nrow(index.ids.return.within.72.index.admit) + 
               nrow(index.ids.return.within.72.index.left)) == nrow(returned.within.72))
  
  index.ids.necessary.return.within.72.index.sent.home <- retreiveReturnVisitData(returned.within.72, all.data, home.labels, admitted.labels)
  index.ids.un.necessary.return.within.72.index.sent.home <- retreiveReturnVisitData(returned.within.72, all.data, home.labels, home.labels)
  index.ids.left.before.seen.return.within.72.index.sent.home <- retreiveReturnVisitData(returned.within.72, all.data, home.labels, left.labels)
  
  index.ids.necessary.return.within.72.index.admit <- retreiveReturnVisitData(returned.within.72, all.data, admitted.labels, admitted.labels)
  index.ids.un.necessary.return.within.72.index.admit <- retreiveReturnVisitData(returned.within.72, all.data, admitted.labels, home.labels)
  index.ids.left.before.seen.return.within.72.index.admit <- retreiveReturnVisitData(returned.within.72, all.data, admitted.labels, left.labels)
  
  index.ids.necessary.return.within.72.index.left <- retreiveReturnVisitData(returned.within.72, all.data, left.labels, admitted.labels)
  index.ids.un.necessary.return.within.72.index.left <- retreiveReturnVisitData(returned.within.72, all.data, left.labels, home.labels)
  index.ids.left.before.seen.return.within.72.index.left <- retreiveReturnVisitData(returned.within.72, all.data, left.labels, left.labels)
  
  index.ids.no.return.within.72.index.sent.home <- retrieveIndexVisitData(no.return.within.72, all.data, home.labels)
  index.ids.no.return.within.72.index.admit <- retrieveIndexVisitData(no.return.within.72, all.data, admitted.labels)
  index.ids.no.return.within.72.index.left <- retrieveIndexVisitData(no.return.within.72, all.data, left.labels)
  
  ## ======================     TOTAL PATIENT VISITS   ==========================
  
  print("Stats by Total Patient Visits")
  
  print(paste("In total,", nrow(returned.within.72), "patients visits resulted in a 72h return"))
  
  print(paste("Of those who returned,", nrow(index.ids.return.within.72.index.sent.home), "were sent home at their index visit and subsequently returned"))
  print(paste("Of those who returned after being sent home at index,", nrow(index.ids.necessary.return.within.72.index.sent.home), "were necessary"))
  print(paste("Of those who returned after being sent home at index,", nrow(index.ids.un.necessary.return.within.72.index.sent.home), "were unnecessary"))
  print(paste("Of those who returned after being sent home at index,", nrow(index.ids.left.before.seen.return.within.72.index.sent.home), "left before being seen"))
  stopifnot((nrow(index.ids.necessary.return.within.72.index.sent.home) + nrow(index.ids.un.necessary.return.within.72.index.sent.home) + 
               nrow(index.ids.left.before.seen.return.within.72.index.sent.home)) == nrow(index.ids.return.within.72.index.sent.home))
  
  print(paste("Of those who returned,", nrow(index.ids.return.within.72.index.admit), "were admitted at their index visit and subsequently returned"))
  print(paste("Of those who returned after being admitted at index,", nrow(index.ids.necessary.return.within.72.index.admit), "were necessary"))
  print(paste("Of those who returned after being admitted at index,", nrow(index.ids.un.necessary.return.within.72.index.admit), "were unnecessary"))
  print(paste("Of those who returned after being admitted at index,", nrow(index.ids.left.before.seen.return.within.72.index.admit), "left before being seen"))
  stopifnot((nrow(index.ids.necessary.return.within.72.index.admit) + nrow(index.ids.un.necessary.return.within.72.index.admit) + 
               nrow(index.ids.left.before.seen.return.within.72.index.admit)) == nrow(index.ids.return.within.72.index.admit))
  
  
  print(paste("Of those who returned,", nrow(index.ids.return.within.72.index.left), "left before being seen at their index visit and subsequently returned"))
  print(paste("Of those who returned after left before being seen at index,", nrow(index.ids.necessary.return.within.72.index.left), "were necessary"))
  print(paste("Of those who returned after left before being seen at index,", nrow(index.ids.un.necessary.return.within.72.index.left), "were unnecessary"))
  print(paste("Of those who returned after left before being seen at index,", nrow(index.ids.left.before.seen.return.within.72.index.left), "left before being seen"))
  stopifnot((nrow(index.ids.necessary.return.within.72.index.left) + nrow(index.ids.un.necessary.return.within.72.index.left) + 
               nrow(index.ids.left.before.seen.return.within.72.index.left)) == nrow(index.ids.return.within.72.index.left))
  
  
  print(paste(nrow(no.return.within.72), "patients visits did not result in a 72h return"))
  print(paste("Of those who did not return within 72h,", nrow(index.ids.no.return.within.72.index.sent.home), "were sent home at their index visit"))
  print(paste("Of those who did not return within 72h,", nrow(index.ids.no.return.within.72.index.admit), "were admitted at their index visit"))
  print(paste("Of those who did not return within 72h,", nrow(index.ids.no.return.within.72.index.left), "left before being seen at their index visit"))
  

  ## ======================     UNIQUE PATIENTS   ==========================
  
  print("Stats by Unique Patients")
  
  print(paste("In total,", length(unique(returned.within.72$PrimaryMedicalRecordNumber)), "unique patients had a 72h return"))
  
  print(paste("Of those unique patients who returned,", length(unique(index.ids.return.within.72.index.sent.home$PrimaryMedicalRecordNumber)), "were sent home at their index visit and subsequently returned"))
  print(paste("Of those unique patients who returned after being sent home at index,", length(unique(index.ids.necessary.return.within.72.index.sent.home$PrimaryMedicalRecordNumber)), "were necessary"))
  print(paste("Of those unique patients who returned after being sent home at index,", length(unique(index.ids.un.necessary.return.within.72.index.sent.home$PrimaryMedicalRecordNumber)), "were unnecessary"))
  print(paste("Of those unique patients who returned after being sent home at index,", length(unique(index.ids.left.before.seen.return.within.72.index.sent.home$PrimaryMedicalRecordNumber)), "left before being seen"))
  stopifnot((length(unique(index.ids.necessary.return.within.72.index.sent.home$PrimaryMedicalRecordNumber)) + 
               length(unique(index.ids.un.necessary.return.within.72.index.sent.home$PrimaryMedicalRecordNumber)) + 
               length(unique(index.ids.left.before.seen.return.within.72.index.sent.home$PrimaryMedicalRecordNumber))) == 
              length(unique(index.ids.return.within.72.index.sent.home$PrimaryMedicalRecordNumber)))
  
  print(paste("Of those unique patients who returned,", length(unique(index.ids.return.within.72.index.admit$PrimaryMedicalRecordNumber)), "were admitted at their index visit and subsequently returned"))
  print(paste("Of those unique patients who returned after being admitted at index,", length(unique(index.ids.necessary.return.within.72.index.admit$PrimaryMedicalRecordNumber)), "were necessary"))
  print(paste("Of those unique patients who returned after being admitted at index,", length(unique(index.ids.un.necessary.return.within.72.index.admit$PrimaryMedicalRecordNumber)), "were unnecessary"))
  print(paste("Of those unique patients who returned after being admitted at index,", length(unique(index.ids.left.before.seen.return.within.72.index.admit$PrimaryMedicalRecordNumber)), "left before being seen"))
  stopifnot((length(unique(index.ids.necessary.return.within.72.index.admit$PrimaryMedicalRecordNumber)) + 
               length(unique(index.ids.un.necessary.return.within.72.index.admit$PrimaryMedicalRecordNumber)) + 
               length(unique(index.ids.left.before.seen.return.within.72.index.admit$PrimaryMedicalRecordNumber))) == 
              length(unique(index.ids.return.within.72.index.admit$PrimaryMedicalRecordNumber)))
  
  
  print(paste("Of those unique patients who returned,", length(unique(index.ids.return.within.72.index.left$PrimaryMedicalRecordNumber)), "left before being seen at their index visit and subsequently returned"))
  print(paste("Of those unique patients who returned after left before being seen at index,", length(unique(index.ids.necessary.return.within.72.index.left$PrimaryMedicalRecordNumber)), "were necessary"))
  print(paste("Of those unique patients who returned after left before being seen at index,", length(unique(index.ids.un.necessary.return.within.72.index.left$PrimaryMedicalRecordNumber)), "were unnecessary"))
  print(paste("Of those unique patients who returned after left before being seen at index,", length(unique(index.ids.left.before.seen.return.within.72.index.left$PrimaryMedicalRecordNumber)), "left before being seen"))
  stopifnot((length(unique(index.ids.necessary.return.within.72.index.left$PrimaryMedicalRecordNumber)) + 
               length(unique(index.ids.un.necessary.return.within.72.index.left$PrimaryMedicalRecordNumber)) + 
               length(unique(index.ids.left.before.seen.return.within.72.index.left$PrimaryMedicalRecordNumber))) == 
              length(unique(index.ids.return.within.72.index.left$PrimaryMedicalRecordNumber)))
  
  
  print(paste(length(unique(no.return.within.72$PrimaryMedicalRecordNumber)), "patients visits did not result in a 72h return"))
  print(paste("Of those unique patients who did not return within 72h,", length(unique(index.ids.no.return.within.72.index.sent.home$PrimaryMedicalRecordNumber)),"were sent home at their index visit"))
  print(paste("Of those unique patients who did not return within 72h,", length(unique(index.ids.no.return.within.72.index.admit$PrimaryMedicalRecordNumber)), "were admitted at their index visit"))
  print(paste("Of those unique patients who did not return within 72h,", length(unique(index.ids.no.return.within.72.index.left$PrimaryMedicalRecordNumber)), "left before being seen at their index visit"))
  
  


  all.data <- dplyr::left_join(x=all.data,
                               y=return.visits[,c("RegistrationNumberVisit1", "NumPreviousVisits", "NumPreviousReturnVisits")],
                               by=c("Pt_Accnt_5"="RegistrationNumberVisit1"))
  
  all.data$NumPreviousReturnVisits <- ifelse(is.na(all.data$NumPreviousReturnVisits), 0, 
                                             all.data$NumPreviousReturnVisits)
  all.data$NumPreviousVisits <- ifelse(is.na(all.data$NumPreviousVisits), 0, 
                                       all.data$NumPreviousVisits)
  
  # filter data to only be those who were sent home at index and retured, or who only visited once and were sent home
  all.data <- all.data %>% dplyr::filter(Pt_Accnt_5 %in% c(as.character(index.ids.return.within.72.index.sent.home$Pt_Accnt_5),
                                                                   as.character(index.ids.no.return.within.72.index.sent.home$Pt_Accnt_5),
                                                                   single.visit.ids.sent.home))
  
  #remove patients who left before being seen at index
  all.data <- all.data %>% dplyr::filter(!Pt_Accnt_5 %in% c(as.character(index.ids.left.before.seen.return.within.72.index.sent.home$Pt_Accnt_5)))

  
  all.data$WillReturn <- ifelse(all.data$Pt_Accnt_5 %in%
                                  c(as.character(index.ids.return.within.72.index.sent.home$Pt_Accnt_5)),
                                1, 0)
  all.data <- all.data[,c("Pt_Accnt_5", "NumPreviousVisits", "NumPreviousReturnVisits", "WillReturn")]
  dim(all.data)
  print(head(all.data))
  fwrite(all.data, paste0(data.path, "willReturn.csv"))
  
  
  return(all.data)
  
  
}


