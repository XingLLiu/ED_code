# May 27th, 2019
# Preprocesses WellSoft Data

# Input: clean_wellSoft.csv
# Output: 
#     - saves preprocessed.csv
#     - saves factors.csv and numerics.csv, names of factor and numeric columns
#     - returns preprocessed wellSoft data

select_columns <- function(data) {
  print("select_columns")
  columns <- c('Pt_Accnt_5',
               'T2_Priority_12',
               #'Ems_Id_32',
               'Method_Of_Arrival_Indexed_S_33',
               'Primary_Name_Sort_Indexed_37',
               'Sex_41',
               'Address_43',
               'Address_Other_44',
               'City_45',
               'Prov_46',
               'Postal_Code_47',
               'Language_56',
               'Hc_Ver_69',
               'Hc_Issuing_Prov_70',
               'Pt_Clsfctn_73',
               'Reg_Status_75',
               'Guardian_Address_85',
               'Relationship_93',
               'Guardian_2_Address_113',
               'Relationship_120',
               'Reason_For_Referral_146',
               'Referring_Location_148',
               'Sickkids_Consultant_157',
               'Sickkids_Service_159',
               'Pmd_Id_172',
               'Admit_Service_186',
               'Override_Et_Status_Cascadi_261',
               'R_A_Override_Minutes_295',
               'Method_Of_Arrival_321',
               'Cedis_Cc_323',
               'Triage_Cc_324',
               'Cedis_Group_325',
               'Ctas_326',
               'Architectural_Dsgntn_328',
               'Area_Of_Care_330',
               'Rm_Seq_331',
               'Qa_Decision_332',
               'Potential_Study_343',
               'Research_Study_344',
               'Pt_Weight_350',
               'Wt_Assessment_351',
               'Other_Template_355',
               'Rn_T1_358', # could be removed in necessary
               'Rn_T2_360',
               'Isolation_363',
               'Tx_364',
               'Rn_Eval_Initial_366',
               'Rn_Eval_Last_367',
               'Rn_Eval_History_372',
               'Rn_Template_375',
               'Staff_Md_Initial_378',
               'Discharge_Md_Np_379',
               'Staff_Md_History_389',
               'Trainee_Initial_391',
               'Discharge_Trainee_392',
               'Trainee_History_400',
               'H_P_Template_402',
               'Rad_Orders_424',
               'Billing_Status_431',
               'Tests_Done_443', # more can be done here
               'Results_Pending_445',
               'Medications_Received_446',
               'Referral_1_448',
               'Referral_2_461',
               'Instruction_1_476', # more can be done --> binarized
               'Instruction_2_477', # more can be done --> binarized
               'Instruction_3_478',
               'Instruction_4_479',
               'Condition_At_Disposition_481',
               'Md_Disposition_Status_482',
               'Restrictions_514',
               'Calls_515', # more can be done --> binarized
               'Form_Bar_Code_523',
               'Comments_1_547',
               'Comments_2_548',
               'Comments_3_550',
               'Nurse_Practitioner_555',
               'New_Rash_557',
               'Follow_Up_559',
               'Archived_Exposure_Alert_581',
               'Consent_To_Approach_582',
               'Rst_Result_584',
               'Dx_1_Group_586',
               'Dx_Code_1_587',
               'Dx_Code_2_588',
               'Dx_Code_3_589',
               'Dx_2_Group_590',
               'Alerts_641',
               'Arrival_Time_9',
               'Age_Dob_40',
               'Hc_Expiration_Date_71',
               'Last_Visit_213',
               'Emerg_Bed_Assgnmnt_225',
               'Status_Ptgone_227',
               'Status_Rmchk_258',
               'Staff_Md_Name_Entrd_267',
               'Discharge_Time_276',
               'Rmvd_From_Pt_Track_288',
               'Rslts_296',
               'Lab_308',
               'Updated_Discharge_Time_596',
               'Triage_Complete_597',
               'Full_Reg_Time_604',
               'Pia_Npia_606',
               'Sepsis_Screening_Triage_616',
               'Sepsis_Screening_Positive_617')
  
  data <- data[,c(columns), with=FALSE]
  print(dim(data))
  return(data)
}




preprocess <- function(data, date.columns=dateCols, Holidays=holidays, 
                       Factors=factors, Numerics=numerics) {
  start.time <- Sys.time()
  print(paste("Start Time", start.time))
  print("Processing Year")
  data$Year <- format(as.POSIXct(data$Arrival_Time_9), "%Y")
  
  print("Processing Address Info")
  data$TopCities <- data$City_45
  top.cities <- names(sort(table(data$City_45), decreasing=T))[1:100]
  data$TopCities[!data$TopCities %in% top.cities] <- "Other"
  data$PostalCodeFSA <- lapply(data$Postal_Code_47, function(x) substr(x, 1, 3))

  print("Processing Guardian Addresses")
  data$AddressSameAsGuardian <- ifelse(data$Address_43 == data$Guardian_Address_85, 1, 0)
  data$GuardiansAddressesSame <- ifelse(data$Guardian_Address_85 == data$Guardian_2_Address_113, 1, 0)
  

  print("Processing Referrals")
  data$Referral <- ifelse(!is.na(data$Reason_For_Referral_146), 1, 0)
  data$Referral[data$Year %in% c("2008", '2009')] <- NA # no data collected this year
  
  data$Referring_Location_148 <- stri_replace_all_regex(data$Referring_Location_148, 
                                                        "(K|k)in?d(\\-| )(E|e)(\\-| ) ?(C|c)are|(K|k)in?d(E|e)r?n? ?(C|c)are|(K|k)idi?(E|e)?(C|c)are|After Hours|(K|k)iddie ?(C|c)are|(K|k)id ?(C|c)rew", 
                                                        "Clinic")

  data$Referring_Location_148[grepl("(W|w)illiam (O|o)sler|(R|r)ouge (V|v)alley|(H|h)ealth|Brampton Civic|Hospital|St Michaels|mt sinai|Sunnybrook|Toronto Western",data$Referring_Location_148)] <- "Hospital"
  data$Referring_Location_148[grepl("Clinic",data$Referring_Location_148)] <- "Clinic"
  data$Referring_Location_148[!data$Referring_Location_148 %in% c( "Clinic", "Doctor", "Home", "Hospital", "Rehab")] <- NA


  print("Processing Consulting Departments")
  data$SK_Consultant <- ifelse(!is.na(data$Sickkids_Consultant_157), 1, 0)
  
  departments <- names(sort(table(data$Sickkids_Service_159))[1100:1140])
  departments <- c(departments, NA)
  
  data$Sickkids_Service_159[!data$Sickkids_Service_159 %in% departments] <- "Other"
  

  print("Processing Admit Services")
  data$Admit_Service_186[grepl("Transfer",data$Admit_Service_186)] <- "Transfer"
  data$Admit_Service_186[grepl("Other",data$Admit_Service_186)] <- "Other"
  data$Admit_Service_186[grepl("(P|p)a?eds",data$Admit_Service_186)] <- "Pediatric"
  rel.admit.service <- names(table(data$Admit_Service_186)[table(data$Admit_Service_186) > 100])
  rel.admit.service <- c(rel.admit.service, NA)
  data$Admit_Service_186[!data$Admit_Service_186 %in% rel.admit.service] <- "Other"
  
  print(Sys.time())
  print("Processing Overrides, number of triage complaints")
  data$Override_Et_Status_Cascadi_261 <- as.numeric(data$Override_Et_Status_Cascadi_261)
  data$R_A_Override_Minutes_295 <- as.numeric(data$R_A_Override_Minutes_295)
  data$NumTriageComplaints <- str_count(data$Triage_Cc_324, ";") + 1
  

  print("Processing Presenting Complaints")
  data$Cedis_Cc_323 <- gsub("; (.)*", "", data$Cedis_Cc_323)
  data$Cedis_Cc_323 <- gsub(", ?(.)*", "", data$Cedis_Cc_323)
  data$Cedis_Cc_323 <- gsub("\\- (.)*", "", data$Cedis_Cc_323)
  data$Cedis_Cc_323[grepl("Fever",data$Cedis_Cc_323)] <- "Fever"
  data$Cedis_Cc_323[grepl("Allergic Reaction",data$Cedis_Cc_323)] <- "Allergic Reaction"
  data$Cedis_Cc_323[grepl("Asthma",data$Cedis_Cc_323)] <- "Asthma"
  data$Cedis_Cc_323[grepl("Abcess|Abscess",data$Cedis_Cc_323)] <- "Abscess"
  data$Cedis_Cc_323[grepl("Behaviou?r(al)?(.)*Changes?",data$Cedis_Cc_323)] <- "Behaviour Change"
  data$Cedis_Cc_323[grepl("Behaviou?r(al)?(.)*(Problems?|Issues?|Concerns)",data$Cedis_Cc_323)] <- "Behaviour Problem"
  data$Cedis_Cc_323[grepl("Behaviou?r(al)?(.)*Atypical",data$Cedis_Cc_323)] <- "Behaviour Atypical"
  data$Cedis_Cc_323[grepl("Bark",data$Cedis_Cc_323)] <- "Barking Cough"
  data$Cedis_Cc_323[grepl("Bee Sting",data$Cedis_Cc_323)] <- "Bee Sting"
  data$Cedis_Cc_323[grepl("Back Pain",data$Cedis_Cc_323)] <- "Back Pain"
  data$Cedis_Cc_323[grepl("Vomi?t?t?(ing)?|Vx",data$Cedis_Cc_323)] <- "Vomiting"
  data$Cedis_Cc_323[grepl("Urology",data$Cedis_Cc_323)] <- "Urology"
  data$Cedis_Cc_323[grepl("U\\/S",data$Cedis_Cc_323)] <- "Ultrasound"
  data$Cedis_Cc_323[grepl("Testicular Pain",data$Cedis_Cc_323)] <- "Testicular Pain"
  data$Cedis_Cc_323[grepl("Transplant",data$Cedis_Cc_323)] <- "Transplant"
  data$Cedis_Cc_323[grepl("Abdominal(.)*Pain|^Abdominal$|Abdo(.)*Pain(.)*",data$Cedis_Cc_323)] <- "Abdominal Pain"
  data$Cedis_Cc_323 <- stri_replace_all_regex(data$Cedis_Cc_323, '^, ?|^;|^- ?|\\? ?|(\\")*$|^(\\")*', "")
  max.complaints <- names(sort(table(data$Cedis_Cc_323), decreasing = TRUE)[1:50])
  rare.complaints <- names(table(data$Cedis_Cc_323)[table(data$Cedis_Cc_323) < 100])
  
  data$RareComplaint <- ifelse(data$Cedis_Cc_323 %in% rare.complaints, 1, 0)
  data$Cedis_Cc_323[!data$Cedis_Cc_323 %in% max.complaints] <- "Other" 
  
  print("Processing Complaint Groups")
  
  data$Cedis_Group_325 <- gsub("^(; ?){1,2}", "", data$Cedis_Group_325)
  data$Cedis_Group_325 <- gsub(";(.)*", "", data$Cedis_Group_325)
  data$Cedis_Group_325 <- gsub("-(.)*", "", data$Cedis_Group_325)
  data$Cedis_Group_325 <- gsub(",(.)*", "", data$Cedis_Group_325)
  
  print("Processing Room Sequences")
  data$Rm_Seq_331 <- gsub("^; ?", "", data$Rm_Seq_331)
  data$Rm_Seq_331 <- gsub("^, ?", "", data$Rm_Seq_331)
  data$NumRooms <- str_count(data$Rm_Seq_331, ",") + 1
  
  print("Processing QA Studies")
  data$Qa_Decision_332[grepl("(F|f)ollow ?- ?(U|u)p", data$Qa_Decision_332)] <- "Follow-up"
  data$Qa_Decision_332[grepl("Resolved", data$Qa_Decision_332)] <- "Resolved"
  data$Qa_Decision_332[!data$Qa_Decision_332 %in% c("Resolved", "Follow-up", NA)] <- "Other"
  
  data$EligibleForReturnVisitStudy <- ifelse(grepl("Return Visit Survey", data$Potential_Study_343), 1, 0)
  data$EligibleForReturnVisitStudy[data$Year %in% c('2015')] <- NA # no data collected that year
  data$NumPotentialStudies <- str_count(data$Potential_Study_343, ",") + 1
  data$NumPotentialStudies[data$NumPotentialStudies==14] <- 2 # manually correct error
  
  data$Research_Study_344[grepl("Not Enrolled", data$Research_Study_344)] <- "Not Enrolled"
  data$Research_Study_344[grepl("Not Applicable", data$Research_Study_344)] <- "Not Applicable"
  data$EnrolledInReturnVisitStudy <- ifelse(grepl("Return Visit Survey", data$Research_Study_344), 1, 0)
  data$EnrolledInReturnVisitStudy[!data$Year %in% c("2016", "2017", "2018")] <- NA
  data$NumStudiesEnrolledIn <- str_count(data$Research_Study_344, ",") + 1
  
  print(Sys.time())
  print("Processing Patient Weight")
  data$Pt_Weight_350 <- as.numeric(data$Pt_Weight_350)
  
  data$Other_Template_355 <- gsub("^(I|in?)$", "Infectious Disease Control", data$Other_Template_355) 
  
  data$SameT1T2Nurse <- ifelse(data$Rn_T1_358==data$Rn_T2_360, 1, 0)

  data$Rn_T1_358 <- as.numeric(as.factor(data$Rn_T1_358))
  
  print("Processing Isolation")
  data$Isolation_363[grepl("None/Routine Practices", data$Isolation_363)] <- "None/Routine Practices"
  data$Isolation_363 <- gsub(";(.)*", "", data$Isolation_363)
  data$Isolation_363 <- gsub(",(.)*", "", data$Isolation_363)
  data$Isolation_363 <- gsub("Droplet/Contact", "Droplet Contact", data$Isolation_363)
  data$Isolation_363 <- gsub("ARO-Contact|ARO V-Contact", "ARO Contact", data$Isolation_363)
  data$Isolation_363[grepl("Droplet/Contact", data$Isolation_363)] <- "None/Routine Practices"

  
  print("Processing History of MDs/RNs")
  data$NumProcedures <- str_count(data$Tx_364, "proc:")
  data$NumMedications <- str_count(data$Tx_364, "med:")
  data$InitialAndLastRNSame <- ifelse(data$Rn_Eval_Initial_366==data$Rn_Eval_Last_367, 
                                   1, 0)
  
  data$NumNurses <- str_count(data$Rn_Eval_History_372, ";")
  
  data$SepsisScreening <- str_count(data$Rn_Template_375, "(S|s)epsis screening")
  
  
  data$SameInitialAndDischargeMD <- ifelse(data$Staff_Md_Initial_378 == data$Discharge_Md_Np_379,
                                           1, 0)
  
  data$Staff_Md_History_389 <- gsub("^; |; ;", "", data$Staff_Md_History_389)
  data$NumDoctors <- str_count(data$Staff_Md_History_389, ";")
  
  data$SameInitialAndDischargeTrainee <- ifelse(data$Trainee_Initial_391 == data$Discharge_Trainee_392, 1, 0)
  
  data$Trainee_Initial_391 <- ifelse(!is.na(data$Trainee_Initial_391), 1, 0)
  
  data$Trainee_History_400 <- gsub("^; |^; ; |^;", "", data$Trainee_History_400)
  
  data$NumTrainees <- str_count(data$Trainee_History_400, ";")
  
  print("Processing Tests/Rad Orders")
  data$Rad_Orders_424[grepl("Ultrasound", data$Rad_Orders_424)] <- "Ultrasound"
  data$Rad_Orders_424[grepl("(X|x)ray", data$Rad_Orders_424)] <- "Xray"
  data$Rad_Orders_424[grepl("MRI", data$Rad_Orders_424)] <- "MRI"
  data$Rad_Orders_424[grepl("EEG", data$Rad_Orders_424)] <- "EEG"
  data$Rad_Orders_424[grepl("CT", data$Rad_Orders_424)] <- "CT"
  data$Rad_Orders_424[!data$Rad_Orders_424 %in% c("Ultrasound", "Xray", "MRI", "EEG", 
                                                  "CT", NA)] <- "Other"
  
  data$Tests_Done_443 <- gsub("^, ?|^- ", "", data$Tests_Done_443)
  
  data$NumTestsDone <- str_count(data$Tests_Done_443, ",") + 1
  data$NumTestsDone[is.na(data$NumTestsDone)] <- 0
  
  data$Results_Pending_445[data$Results_Pending_445 %in% c("None", "none")] <- "None"
  data$Results_Pending_445[is.na(data$Results_Pending_445)] <- "No Tests Done"
  data$Results_Pending_445[!data$Results_Pending_445 %in% c("None", "No Tests Done")] <- "Results Pending"
  
  print("Processing Medications and Referrals")
  data$Medications_Received_446 <- ifelse(!is.na(data$Medications_Received_446), 1, 0)
  data$Medications_Received_446[data$Year %in% c("2016", "2017", "2018")] <- NA
  data$Referral_1_448 <- ifelse(!is.na(data$Referral_1_448), 1, 0)
  data$Referral_2_461 <- ifelse(!is.na(data$Referral_2_461), 1, 0)
  
  print(Sys.time())
  print("Processing Instructions to Patient")
  data$Instruction_1_476 <- ifelse(!is.na(data$Instruction_1_476), 1, 0)
  data$Instruction_2_477 <- ifelse(!is.na(data$Instruction_2_477), 1, 0)
  data$Instruction_3_478 <- ifelse(!is.na(data$Instruction_3_478), 1, 0)
  data$Instruction_4_479 <- ifelse(!is.na(data$Instruction_4_479), 1, 0)
  
  print("Processing Patient's Disposition Condition and Restrictions")
  data$Condition_At_Disposition_481[!data$Condition_At_Disposition_481 %in% c("Good", "Fair", "Poor", "Expired", 
                                                                              "Stable", "Critical", "Well")] <- NA
  
  data$Restrictions_514[grepl("(N|n)one", data$Restrictions_514)] <- "None"
  data$Restrictions_514[!data$Restrictions_514 %in% c('None') & !is.na(data$Restrictions_514)] <- "Other Restriction"

  data$Calls_515 <- ifelse(!is.na(data$Calls_515), 1, 0)
  
  print("Processing Comments")
  comment.years <- c("2015", "2016", "2017", "2018")
  data$Comments_1_547 <- ifelse(!is.na(data$Comments_1_547), 1, 0)
  data$Comments_1_547[!data$Year %in% comment.years] <- NA
  data$Comments_2_548 <- ifelse(!is.na(data$Comments_2_548), 1, 0)
  data$Comments_2_548[!data$Year %in% comment.years] <- NA
  data$Comments_3_550 <- ifelse(!is.na(data$Comments_3_550), 1, 0)
  data$Comments_3_550[!data$Year %in% comment.years] <- NA
  
  data$Nurse_Practitioner_555 <- ifelse(!is.na(data$Nurse_Practitioner_555), 1, 0)
  data$Nurse_Practitioner_555[data$Year %in% c("2008", "2009")] <- NA
  
  data$New_Rash_557[is.na(data$New_Rash_557) & data$Year %in% c("2015")] <- "No"
  
  
  print("Processing Follow Ups, Archieved Exposure, RST")
  data$Follow_Up_559[grepl("Pediatrician|(F|f)amily (MD|(P|p)hysician)|GP|(P|p)a?ediatrician|peds|Primary Care Doctor", data$Follow_Up_559)] <- "Family Doctor"
  data$Follow_Up_559[grepl("((R|r)egular|(F|f)amily|(O|o)wn|(C|c)hild\\'s) (D|d)octor", data$Follow_Up_559)] <- "Family Doctor"
  
  data$Follow_Up_559[!is.na(data$Follow_Up_559) & !data$Follow_Up_559 %in% c("Family Doctor")] <- "Other"
  data$Follow_Up_559[data$Year %in% c("2015", "2016", "2017", "2018") & is.na(data$Follow_Up_559)] <- "No Follow Up"
  
  data$Archived_Exposure_Alert_581[!is.na(data$Archived_Exposure_Alert_581)] <- "Measles Exposure (March 2014)"
  data$Archived_Exposure_Alert_581[is.na(data$Archived_Exposure_Alert_581) & data$Year %in% c("2014")] <- "No Exposure"
  
  data$Consent_To_Approach_582[data$Year %in% c("2016", "2017", "2018") & is.na(data$Consent_To_Approach_582)] <- "Missing"

  data$Rst_Result_584[grepl("positive", data$Rst_Result_584)] <- "Positive"
  data$Rst_Result_584[grepl("negative", data$Rst_Result_584)] <- "Negative"
  data$Rst_Result_584[data$Year %in% c("2015", "2016", "2017", "2018") & is.na(data$Rst_Result_584)] <- "RST Not Done"
  
  
  print("Processing Discharge Groups")
  data$Dx_1_Group_586[grepl("^Infectious", data$Dx_1_Group_586)] <- "Infectious Diseases"
  data$Dx_1_Group_586[grepl("^Endocrine, Nutritional", data$Dx_1_Group_586)] <- "Endocrine, Nutritional And Metabolic Disorders"
  data$Dx_1_Group_586[grepl("^Symptoms, (S|s)igns", data$Dx_1_Group_586)] <- "Symptoms, Signs And Abnormal Findings"
  rare.DX <- names(table(data$Dx_1_Group_586)[table(data$Dx_1_Group_586) < 50])
  data$Dx_1_Group_586[data$Dx_1_Group_586 %in% rare.DX] <- "Rare"
  data$Dx_1_Group_586[data$Year %in% c("2015", "2016", "2017", "2018") & is.na(data$Dx_1_Group_586)] <- "Missing"
  
  data$Dx_2_Group_590[grepl("^Symptoms, (S|s)igns", data$Dx_2_Group_590)] <- "Symptoms, Signs And Abnormal Findings"
  data$Dx_2_Group_590[grepl("^Infectious", data$Dx_2_Group_590)] <- "Infectious Diseases"
  data$Dx_2_Group_590[grepl("^Endocrine, Nutritional", data$Dx_2_Group_590)] <- "Endocrine, Nutritional And Metabolic Disorders"
  rare.DX.2 <- names(table(data$Dx_2_Group_590)[table(data$Dx_2_Group_590) < 50])
  data$Dx_2_Group_590[data$Dx_2_Group_590 %in% rare.DX.2] <- "Rare"
  data$Dx_2_Group_590[data$Year %in% c("2015", "2016", "2017", "2018") & is.na(data$Dx_2_Group_590)] <- "Missing"
  
  print("Processing Number of Discharge Codes, Alerts")
  dx.cols <- data[,c("Dx_Code_1_587", "Dx_Code_2_588", "Dx_Code_3_589")]
  dx.cols[is.na(dx.cols)] <- 0
  dx.cols$Dx_Code_1_587 <- as.numeric(dx.cols$Dx_Code_1_587)
  dx.cols$Dx_Code_2_588 <- as.numeric(dx.cols$Dx_Code_2_588)
  dx.cols$Dx_Code_3_589 <- as.numeric(dx.cols$Dx_Code_3_589)
  data$NumDX <- rowSums(dx.cols)
  data$NumDx[!data$Year %in% c("2015", "2016", "2017", "2018")] <- NA
  
  
  data$Alerts_641 <- gsub(", ?$", "", data$Alerts_641)
  data$NumAlerts <- str_count(data$Alerts_641, ",") + 1
  data$NumAlerts[is.na(data$NumAlerts)] <- 0
  
  # Dates : format(as.POSIXct(rawData$Arrival_Time_9), "%Y") 
  #difftime : later time , earlier time

  print("=============== Processing Dates ===============")
  print(Sys.time())
  print("Processing Arrival Date")
  arrivalDate <- as.Date(data$Arrival_Time_9)
  holidayDates <- merge(x=data.frame(arrivalDate), 
                    y=Holidays[,c("Date", "Public.Holiday")],
                    by.x="arrivalDate", by.y="Date",
                    all.x=TRUE)
  
  data$Arrival_Time_9 <- as.POSIXct(data$Arrival_Time_9, tz = "EST")
  month.num <- month(data$Arrival_Time_9)
  data$ArrivalMonth <- month.name[month.num]
  data$ArrivalNumHoursSinceMidnight <- hour(as.ITime(data$Arrival_Time_9))
  data$ArrivalNumHoursSinceMidnight[data$ArrivalNumHoursSinceMidnight < 0] <- NA
  data$ArrivalDayOfWeek <- weekdays(data$Arrival_Time_9)
  data$ArrivalIsBusinessDay <- is.bizday(data$Arrival_Time_9)
  data$ArrivalHoliday <- holidayDates$Public.Holiday
  
  print("Processing DoB, HC Expiration, Last Visit")
  #data$Age_Dob_40 <- as.POSIXct(data$Age_Dob_40, tz = "EST")
  data$DaysOld <- as.numeric(round(difftime(data$Arrival_Time_9, data$Age_Dob_40, units='days'), 1))
  data$DaysOld[data$DaysOld < 0] <- NA
  
  #data$Hc_Expiration_Date_71 <- as.Date(data$Hc_Expiration_Date_71)
  data$WeeksToHCExpiration <- as.numeric(round(difftime(data$Hc_Expiration_Date_71, data$Arrival_Time_9, units='weeks'), 1))
  data$WeeksToHCExpiration[data$WeeksToHCExpiration < 0] <- NA
  
  #data$Last_Visit_213 <- as.Date(data$Last_Visit_213)
  data$HoursSinceLastVisit <- as.numeric(round(difftime(data$Arrival_Time_9, data$Last_Visit_213, units="hours"), 1))
  data$HoursSinceLastVisit[data$HoursSinceLastVisit < 0] <- NA
  
  
  print("Processing Time Spent in ED")
  data$MinsToEDBedAssignment <- as.numeric(round(difftime(data$Emerg_Bed_Assgnmnt_225, data$Arrival_Time_9, units="mins"), 1))
  data$MinsToEDBedAssignment[data$MinsToEDBedAssignment < 0] <- NA
  
  
  data$PtGone <- ifelse(!is.na(data$Status_Ptgone_227), 1, 0)
  data$PtGone[!data$Year %in% c("2015", "2016", "2017", "2018")] <- NA 
  
  data$HoursToRmCheck <- as.numeric(round(difftime(data$Status_Rmchk_258, data$Arrival_Time_9, units="hours"), 1))
  data$HoursToRmCheck[data$HoursToRmCheck < 0] <- NA  
  
  data$MinsToMDNameEntrd <- as.numeric(round(difftime(data$Staff_Md_Name_Entrd_267, data$Arrival_Time_9, units="mins"), 1))
  data$MinsToMDNameEntrd[data$MinsToMDNameEntrd < 0] <- NA  
  
  print("Processing Discharge Information")
  data$Discharge_Time_With_Updates <- ifelse(data$Updated_Discharge_Time_596 != "", data$Updated_Discharge_Time_596, data$Discharge_Time_276)
  
  data$LengthOfStayInMinutes <- as.numeric(round(difftime(data$Discharge_Time_With_Updates, data$Arrival_Time_9, units="mins"), 1))
  data$LengthOfStayInMinutes[data$LengthOfStayInMinutes < 0] <- NA  
  
  data$Discharge_Time_276 <- as.POSIXct(data$Discharge_Time_276, tz = "EST")
  data$DepartureNumHoursSinceMidnight <- hour(as.ITime(data$Discharge_Time_276))
  data$DepartureNumHoursSinceMidnight[data$DepartureNumHoursSinceMidnight < 0] <- NA
  
  print("Processing Other Times")
  data$HoursToRmPtFromPtTrack <- as.numeric(round(difftime(data$Rmvd_From_Pt_Track_288, data$Arrival_Time_9, units="hours"), 1))
  data$HoursToRmPtFromPtTrack[data$HoursToRmPtFromPtTrack < 0] <- NA  
  
  
  data$HoursToRslts <- as.numeric(round(difftime(data$Rslts_296, data$Arrival_Time_9, units="hours"), 1))
  data$HoursToRslts[data$HoursToRslts < 0] <- NA 
    
  
  data$HoursToLab <- as.numeric(round(difftime(data$Lab_308, data$Arrival_Time_9, units="hours"), 1))
  data$HoursToLab[data$HoursToLab < 0] <- NA 
    
  data$MinsToCompleteTraige <- as.numeric(round(difftime(data$Triage_Complete_597, data$Arrival_Time_9, units="mins"), 1))
  data$MinsToCompleteTraige[data$MinsToCompleteTraige < 0] <- NA
  
  data$MinsToFullRegistration <- as.numeric(round(difftime(data$Full_Reg_Time_604, data$Arrival_Time_9, units="mins"), 1))
  data$MinsToFullRegistration[data$MinsToFullRegistration < 0] <- NA
    
  data$PIAInMins <- as.numeric(round(difftime(data$Pia_Npia_606, data$Arrival_Time_9, units="mins"), 1))
  data$PIAInMins[data$PIAInMins < 0] <- NA
  
  data$SepsisScreeningAtTriage <- ifelse(!is.na(data$Sepsis_Screening_Triage_616), 1, 0)
  data$SepsisScreeningAtTriage[!data$Year %in% c("2016", "2017", "2018")] <- NA
  
  data$SepsisScreeningPositive <- ifelse(!is.na(data$Sepsis_Screening_Positive_617), 1, 0)
  data$SepsisScreeningPositive[!data$Year %in% c("2016", "2017", "2018")] <- NA
  
  print(paste("Returning Preprocessed Data in", round((Sys.time() - start.time), 3), "minutes"))
  
  # rename columns
  
  data <- data %>% 
    rename(
      T2_Priority_12 <- T2Priority,
      Method_Of_Arrival_Indexed_S_33 <- MethodOfArrival,
      Sex_41 <- Sex, 
      Prov_46 <- Province,
      Language_56 <- Language,
      Hc_Ver_69 <- HealthCardVersionCode,
      Hc_Issuing_Prov_70 <- HealthCardIssuingProvince,
      Pt_Clsfctn_73 <- PtClsfctn,
      Reg_Status_75 <- RegStatus,
      Relationship_93 <- RelationshipToChild1,
      Relationship_120 <- RelationshipToChild2,
      Referring_Location_148 <- ReferringLocation,
      Pmd_Id_172 <- PmdId,
      Admit_Service_186 <- AdmitService,
      Method_Of_Arrival_321 <- MethodOfArrival2,
      Cedis_Cc_323 <- PresentingComplaint,
      Cedis_Group_325 <- PresentingComplaintGroup,
      Ctas_326 <- CTAS,
      Architectural_Dsgntn_328 <- ArchitecturalDsgntn,
      Area_Of_Care_330 <- AreaOfCare, 
      Qa_Decision_332 <- QaDecision,
      Wt_Assessment_351 <- WeightMeasuredOrEstimated,
      Other_Template_355 <- NameOfTemplateUsed,
      Isolation_363 <- IsolationDueTo,
      Rn_Template_375 <- RnTemplate,
      Trainee_Initial_391 <- TraineePresent,
      H_P_Template_402 <- HPTemplate,
      Rad_Orders_424 <- TypeOfRadOrdered,
      Billing_Status_431 <- BillingStatus,
      Results_Pending_445 <- ResultsPending,
      Medications_Received_446 <- PatientReceivedMedication, 
      Referral_1_448 <- ReferralMade1,
      Referral_2_461 <- ReferralMade2,
      Instruction_1_476 <- OneInstructionExists,
      Instruction_2_477 <- SecondInstructionExists,
      Instruction_3_478 <- ThirdInstructionExists,
      Instruction_4_479 <- FourthInstructionExists,
      Condition_At_Disposition_481 <- ConditionAtDisposition,
      Md_Disposition_Status_482 <- MDDispositionStatus,
      Restrictions_514 <- Restrictions,
      Calls_515 <- CallsExists,
      Form_Bar_Code_523 <- FormBarCode,
      Comments_1_547 <- Comment1Exists,
      Comments_2_548 <- Comment2Exists,
      Comments_3_550 <- Comment3Exists,
      Nurse_Practitioner_555 <- NursePractitionerInvolved,
      New_Rash_557 <- NewRashPresent,
      Follow_Up_559 <- FollowUpScheduled,
      Archived_Exposure_Alert_581 <- MeaslesExposureMarch2014,
      Consent_To_Approach_582 <- ConsentToApproachForStudy,
      Rst_Result_584 <- RSTResult,
      Dx_1_Group_586 <- DiseaseGroup1,
    )
  
  return(data[,c(Factors, Numerics, "Arrival_Time_9", "Year")])

}






factors <- c("TopCities",   # QP
             "PostalCodeFSA",  # QP
            "T2Priority", # ?
             "MethodOfArrival", # QP
             'Sex', # QP
             "Province", #QP  
             'Language', #QP
             'HealthCardVersionCode', #QP
             'HealthCardIssuingProvince', #QP
             'PtClsfctn', # ?
             'RegStatus', #?
             'AddressSameAsGuardian', #DP
             'RelationshipToChild1', # DP
             'GuardiansAddressesSame', # DP
             'RelationshipToChild2', #DP
             'Referral',
             'ReferringLocation',  
             'SK_Consultant',
             #'Sickkids_Service_159', #?
             'PmdId', #?
             'AdmitService', #?
             'MethodOfArrival2', # QP
             'PresentingComplaint',
             #'Triage_Cc_324',
             'RareComplaint',
             'PresentingComplaintGroup',
             'CTAS',
             'ArchitecturalDsgntn', # ?
             'AreaOfCare', ## needs to be mapped!
             'QaDecision',
             'EligibleForReturnVisitStudy',
             'EnrolledInReturnVisitStudy',
             'WeightMeasuredOrEstimated',
             'NameOfTemplateUsed',
             #'Rn_T1_358',
             'SameT1T2Nurse',
             'IsolationDueTo',
             'RnTemplate',
             'SepsisScreening',
             'SameInitialAndDischargeMD',
             'TraineePresent',
             'SameInitialAndDischargeTrainee',
             'HPTemplate', #?
             'TypeOfRadOrdered',
             'BillingStatus',
             'ResultsPending',
             'PatientReceivedMedication',
             'ReferralMade1',
             'ReferralMade2',
             'OneInstructionExists',
             "SecondInstructionExists",
             "ThirdInstructionExists",
             "FourthInstructionExists",
             'ConditionAtDisposition',
             'MDDispositionStatus',
             'Restrictions',
             'CallsExists',
             'FormBarCode',
             'Comment1Exists',
             'Comment2Exists',
             'Comment3Exists',
             'NursePractitionerInvolved',
            "InitialAndLastRNSame",
             'NewRashPresent',
             'FollowUpScheduled',
             'MeaslesExposureMarch2014',
             'ConsentToApproachForStudy',
             'RSTResult',
             'DiseaseGroup1',
             'ArrivalMonth',
             'ArrivalDayOfWeek',
             'ArrivalIsBusinessDay',
             'ArrivalHoliday',
             'PtGone',
             'SepsisScreeningAtTriage',
             'SepsisScreeningPositive',
             'WillReturn')

numerics <- c("NumTriageComplaints",
              "NumRooms",
              "NumPotentialStudies",
              "NumStudiesEnrolledIn",
              "NumProcedures",
              "NumMedications",
              "NumNurses",
              "NumDoctors",
              "NumTrainees",
              "NumTestsDone",
              'NumDX',
              'NumAlerts',
              'ArrivalNumHoursSinceMidnight',
              'DaysOld',
              'WeeksToHCExpiration',
              'HoursSinceLastVisit',
              'MinsToEDBedAssignment',
              'HoursToRmCheck',
              'MinsToMDNameEntrd',
              'LengthOfStayInMinutes',
              'DepartureNumHoursSinceMidnight',
              'HoursToRmPtFromPtTrack',
              'HoursToRslts',
              'HoursToLab',
              'MinsToCompleteTraige',
              'MinsToFullRegistration',
              'PIAInMins',
              'mean_Median_total_income',
              'Distance_To_Hospital',
              'Distance_To_SickKids',
              'Distance_To_Walkin',
              'NumberOfPatientsAtStart',
              'TotalNumberOfPatients',
              'NumberCTAS1',
              'NumberCTAS2',
              'NumberCTAS3',
              'NumberCTAS4',
              'NumberCTAS5',
              'NumberCTAS9')

newVars <- c("Year",
             "AddressSameAsGuardian",
             "GuardiansAddressesSame",
             "Referral",
             "SK_Consultant",
             "NumTriageComplaints",
             "RareComplaint",
             "NumRooms",
             "EligibleForReturnVisitStudy",
             "NumPotentialStudies",
             "EnrolledInReturnVisitStudy",
             "NumStudiesEnrolledIn",
             "SameT1T2Nurse",
             "NumProcedures",
             "NumMedications",
             "InitialAndLastRNSame",
             "NumNurses",
             "SepsisScreening",
             "SameInitialAndDischargeMD",
             "NumDoctors",
             "SameInitialAndDischargeTrainee",
             "NumTrainees",
             "NumTestsDone",
             'NumDX',
             'NumAlerts',
             'ArrivalMonth',
             'ArrivalNumHoursSinceMidnight',
             'ArrivalDayOfWeek',
             'ArrivalIsBusinessDay',
             'ArrivalHoliday',
             'DaysOld',
             'WeeksToHCExpiration',
             'HoursSinceLastVisit',
             'MinsToEDBedAssignment',
             'PtGone',
             'HoursToRmCheck',
             'MinsToMDNameEntrd',
             'LengthOfStayInMinutes',
             'DepartureNumHoursSinceMidnight',
             'HoursToRmPtFromPtTrack',
             'HoursToRslts',
             'HoursToLab',
             'MinsToCompleteTraige',
             'MinsToFullRegistration',
             'PIAInMins',
             'SepsisScreeningAtTriage',
             'SepsisScreeningPositive')

preprocessWellSoft <- function(data, data.path) {
  holidays <- read.csv(paste0(data.path, "holidays.csv"))
  holidays$Date <- as.Date(holidays$Date)
  preprocessed <- select_columns(data)
  
  preprocessed <- addExtraVariables(data, data.path)
  
  preprocessed <- preprocess(data)
  
  fwrite(preprocessed, paste0(data.path, "preprocessed_wellSoft.csv"))
  write.csv(x = factors, paste0(data.path, "factors.csv"))
  write.csv(x=numerics, paste0(data.path, "numerics.csv"))
  
  return(preprocessed)
  
}

