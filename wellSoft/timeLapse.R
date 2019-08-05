# NEED TO CHECK

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

library(data.table)


# ================== CALCULATE NUMBER OF VISITS PER PERSON ================== # 


num.readmissions <- data.frame(wellSoft %>% dplyr::filter(!DischargeDisposition %in% to.remove) %>% 
                                 dplyr::group_by(PrimaryMedicalRecordNumber) %>%
                                 dplyr::mutate(num.returns = n()))

num.readmissions <- num.readmissions %>% dplyr::arrange(desc(num.returns), 
                                                        PrimaryMedicalRecordNumber, StartOfVisit)

single.visit.ids <- (num.readmissions %>% dplyr::filter(num.returns == 1))
single.visit.ids.sent.home <- (single.visit.ids %>% filter(DischargeDisposition %in% home.labels))$PrimaryMedicalRecordNumber
single.visit.ids <- single.visit.ids$PrimaryMedicalRecordNumber


order.returns <- num.readmissions[, c("RegistrationNumber", "PrimaryMedicalRecordNumber",
                                      "StartOfVisit", "EndOfVisit")]

num.readmissions <- num.readmissions[!duplicated(num.readmissions$PrimaryMedicalRecordNumber),
                                     c("PrimaryMedicalRecordNumber", "num.returns")]

N.visits <- nrow(num.readmissions)
mean.visits <- mean(num.readmissions$num.returns); median.visits <- median(num.readmissions$num.returns)
unique.PMRN <- num.readmissions[order(-num.readmissions$num.returns),]$PrimaryMedicalRecordNumber




# Number of visits per person 

order.returns
rel.ids <- unique.PMRN[!unique.PMRN %in% single.visit.ids]

time.lapse <- data.frame(matrix(ncol = 4, nrow = 0))
colnames(time.lapse) <- c("PrimaryMedicalRecordNumber", "RegistrationNumberVisit1", 
                          "RegistrationNumberVisit2", "DifferenceInDays")

length(rel.ids)
for (i in 1:length(rel.ids)) {
  patient.id <- rel.ids[i]; print(paste("Patient", i, "out of", length(rel.ids)))
  visits <- order.returns %>% dplyr::filter(PrimaryMedicalRecordNumber %in% c(patient.id))
  visits <- visits %>% dplyr::arrange(StartOfVisit)

  differences <- difftime(visits[2:nrow(visits), c("StartOfVisit")], 
                                         visits[1:(nrow(visits)-1), c("EndOfVisit")], units="days")
  visits.reg <- visits$RegistrationNumber

  visit.1.reg.num <- visits.reg[1:(length(visits.reg)-1)]
  visit.2.reg.num <- visits.reg[2:length(visits.reg)]

  time.lapse. <- rbind(time.lapse., data.frame(PrimaryMedicalRecordNumber=rep_len(patient.id, length(differences)),
                                             RegistrationNumberVisit1=visit.1.reg.num,
                                             RegistrationNumberVisit2=visit.2.reg.num,
                                             DifferenceInDays=differences))


}



saveRDS(time.lapse, "timeLapse.rds")

all(rel.ids %in% time.lapse$PrimaryMedicalRecordNumber) # verify all patients were processed 
all(time.lapse$PrimaryMedicalRecordNumber %in% rel.ids) # verify all patients were processed  -- won't be true since now removed certain labels


