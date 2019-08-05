## NEEDS TO BE TESTED PROPERLY!!!
#
# Creates private, semi-private and de-identified data sets for training/testing
# Input: preprocessed_wellSoft.csv
# Output: 4 versions of data for training 
#         v1_Data.csv: derivedFromPrivate and quasiprivate variables removed
#         v2_Data.csv: derivedFromPrivate removed and some quasiprivate generalized
#         v3_Data.csv: derivedFromPrivate removed and some quasiprivate perturbed
#         v4_Data.csv: derivedFromPrivate removed and quasiprivate included
#
# De-identification info from: https://www.ipc.on.ca/wp-content/uploads/2016/08/Deidentification-Guidelines-for-Structured-Data.pdf

# private == directly identifiable (in Canada, anything more identifiable than province == PHI)

derivedFromPrivate <- c("AddressSameAsGuardian",
                        'RelationshipToChild1',
                        'GuardiansAddressesSame',
                        'RelationshipToChild2',
                        "mean_Median_total_income",
                        "Distance_To_Hospital",
                        "Distance_To_SickKids",
                        "Distance_To_Walkin")


quasiprivate <- c("TopCities", 
                  "PostalCodeFSA", 
                  "MethodOfArrival", 
                  "Sex", 
                  "Province", 
                  "Language", 
                  "HealthCardVersionCode",
                  "HealthCardIssuingProvince", 
                  "MethodOfArrival2",
                  "RareComplaint",
                  "ArrivalMonth",
                  "ArrivalDayOfWeek",
                  "ArrivalIsBusinessDay",
                  "ArrivalHoliday",
                  "Year",
                  "ArrivalNumHoursSinceMidnight",
                  "DepartureNumHoursSinceMidnight",
                  "LengthOfStayInMinutes",
                  "DaysOld"); length(quasiprivate)


# METHOD 1: Remove all derived from private and quasi-private


# METHOD 2: - Remove all derived from private; 
#           - Generalize quasi-identifiers 
#             * Remove:  TopCities, Sex, HealthCardVersionCode, HealthCardIssuingProvince,
#                       MethodOfArrival2, RareComplaint, ArrivalIsBusinessDay,
#                       LengthOfStayInMinutes
#             * Generalize: PostalCodeFSA, MethodOfArrival, Province, Language,
#                           ArrivalMonth, ArrivalDayOfWeek, ArrivalHoliday,
#                           Year, ArrivalNumHoursSinceMidnight, 
#                           DepartureNumHoursSinceMidnight, DaysOld


# Generalize_Times
start.hours <- wellSoft$ArrivalNumHoursSinceMidnight
sh <- start.hours
sh[8 <= start.hours & start.hours < 17] <- 'Morning' 
sh[17 <= start.hours & start.hours < 23] <- 'Evening'
sh[(0 <= start.hours & start.hours < 8) | start.hours == "23"] <- 'Night'

end.hours <- wellSoft$DepartureNumHoursSinceMidnight
eh <- end.hours
eh[8 <= end.hours & end.hours < 17] <- 'Morning' 
eh[17 <= end.hours & end.hours < 23] <- 'Evening'
eh[(0 <= end.hours & end.hours < 8) | end.hours == "23"] <- 'Night'


wellSoft$v2_ArrivalTimeOfDay <- sh # --> Insead of ArrivalNumHoursSinceMidnight
wellSoft$v2_DepartureTimeOfDay <- eh # --> Instead of DepartureNumHoursSinceMidnight

wellSoft$v2_ArrivalMonth <- wellSoft$ArrivalMonth
wellSoft$v2_ArrivalMonth <- gsub(pattern = "September|October|November", 
                          replacement = "Fall",
                          x = wellSoft$v2_ArrivalMonth)
wellSoft$v2_ArrivalMonth <- gsub(pattern = "December|January|February", 
                          replacement = "Winter",
                          x = wellSoft$v2_ArrivalMonth)
wellSoft$v2_ArrivalMonth <- gsub(pattern = "March|April|May", 
                          replacement = "Spring",
                          x = wellSoft$v2_ArrivalMonth)
wellSoft$v2_ArrivalMonth <- gsub(pattern = "June|July|August", 
                          replacement = "Summer",
                          x = wellSoft$v2_ArrivalMonth)
wellSoft$v2_ArrivalDayOfWeek <- ifelse(wellSoft$ArrivalDayOfWeek =="Saturday" | wellSoft$ArrivalDayOfWeek == 'Sunday',
                                "Weekend",
                                "Weekday")
wellSoft$v2_ArrivalHoliday <- ifelse(!is.na(wellSoft$ArrivalHoliday), 
                              "HOLIDAY",
                              "NOT HOLIDAY")

wellSoft$v2_MethodOfArrival <- ifelse(wellSoft$MethodOfArrival == "Ambulatory",
                                                     "Ambulatory",
                                                     "Other")

wellSoft$v2_Language <- ifelse(wellSoft$Language=="English",
                                  "English",
                                  "Other")
wellSoft$v2_PostalCodeFSA <- PostalCodeFSA
wellSoft$v2_Province <- wellSoft$Province
wellSoft$v2_Province[!wellSoft$v2_Province=="ON"] <- "Other"

wellSoft$v2_DaysOld <- wellSoft$DaysOld
wellSoft$v2_DaysOld[wellSoft$DaysOld < (365*2)] <- "<2"
wellSoft$v2_DaysOld[wellSoft$DaysOld >= (365*2) & wellSoft$DaysOld < (365*5)] <- "2-5"
wellSoft$v2_DaysOld[wellSoft$DaysOld >= (365*5) & wellSoft$DaysOld < (365*12)] <- "5-12"
wellSoft$v2_DaysOld[wellSoft$DaysOld >= (365*12)] <- ">=12"



# METHOD 3:   - Remove direct; 
#             - Perturb quai-identifiers 
#               (for each quasi, randomly sample from the distribution of variable)
#             * Remove:  TopCities, PostalCodeFSA, MethodOfArrival, Sex, Language,
#                       HealthCardVersionCode, HealthCardIssuingProvince,
#                       MethodOfArrival2, RareComplaint, ArrivalIsBusinessDay,
#                       ArrivalHoliday, 
#                       LengthOfStayInMinutes
#             * Perturb: ArrivalMonth, ArrivalDayOfWeek, Year,
#                       ArrivalNumHoursSinceMidnight, 
#                       DepartureNumHoursSinceMidnight,
#                       LengthOfStayInMinutes, DaysOld



wellSoft$v3_Province <- wellSoft$Province


set.seed(0)
createOffsets <- function(small, large, n=nrow(wellSoft)) sample(small:large, n, replace=T)

wellSoft$v3_DaysOld <- createOffsets(-(12*30), (12*30)) + wellSoft$DaysOld # +/- 1 year
wellSoft$v3_DaysOld[wellSoft$v3_DaysOld<0] <- createOffsets(0, 15, sum(wellSoft$v3_DaysOld<0))

wellSoft$v3_ArrivalNumHoursSinceMidnight <- createOffsets(-90, 90) + wellSoft$ArrivalNumHoursSinceMidnight # +/- 1.5h
wellSoft$v3_ArrivalNumHoursSinceMidnight[wellSoft$v3_ArrivalNumHoursSinceMidnight<0] <- createOffsets(0, 180, sum(wellSoft$v3_ArrivalNumHoursSinceMidnight<0))

wellSoft$v3_DepartureNumHoursSinceMidnight <- createOffsets(-90, 90) + wellSoft$DepartureNumHoursSinceMidnight # +/- 1.5h
wellSoft$v3_DepartureNumHoursSinceMidnight[wellSoft$v3_DepartureNumHoursSinceMidnight<0] <- createOffsets(0, 180, sum(wellSoft$v3_DepartureNumHoursSinceMidnight<0))


wellSoft$v3_LengthOfStayInMinutes <- wellSoft$v3_DepartureNumHoursSinceMidnight - wellSoft$v3_ArrivalNumHoursSinceMidnight
wellSoft$v3_LengthOfStayInMinutes[wellSoft$v3_LengthOfStayInMinutes<0] <- 24-abs(wellSoft$v3_LengthOfStayInMinutes[wellSoft$v3_LengthOfStayInMinutes<0])


wellSoft$v3_Year <- createOffsets(-2, 2) + wellSoft$Year # +/- 2 years
wellSoft$v3_ArrivalDayOfWeek <- sample(wellSoft$ArrivalDayOfWeek)
wellSoft$v3_ArrivalMonth <- sample(wellSoft$ArrivalMonth)


# METHOD 4: include everything as is

fwrite(paste0(path, "multipleVersions_wellSoft.csv"))

