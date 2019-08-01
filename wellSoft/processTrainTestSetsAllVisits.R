## NEED TO CHECK


# Creates training and testing sets using all the data sources
# Creates sets for two problems:
#       1. CLASSIFICATION 1: likely return vs likely non-return for patients who are sent home at their index visit
#       2. CLASSIFICATION 2: likely admit vs likely non-admit for patients who do return within 72h of their index visit
#
# NOTE: run after compileAllData.R

library(zoo)
# ------------------- HELPER FUNCTIONS ------------------- #

retrieveIndexVisitData <- function(regs, labels) {

  # returns discharge disposition for subset of visits based on
  # `labels` discharge at index visit

  # get Discharge Dispositions for Index visits
  index.discharges <- merge(x = data.frame(RegistrationNumber = regs[,c("RegistrationNumberVisit1")]),
                            y = all.data[,c("RegistrationNumber", "DischargeDisposition")])

  # filter by those who fall into 'labels' at first visit
  altered.data <- index.discharges %>% dplyr::filter(DischargeDisposition %in% labels)
  return (altered.data)
}


retreiveReturnVisitData <- function(regs, initial.lables, return.visit.labels ) {

  # returns discharge disposition for return visits based on
  # `initial.labels` outcomes of index visit
  # `return.labels` outcome of return visit
  #
  # Variables:
  #   regs - dataframe in the form of time.lapse for a particular segment of people
  #         (.e.g retun visits within 72 h with index visit sent home)
  #   labels - one of home, admitted, remove, left (type of disposition, defined in dischargeCategories.R)

  altered.data <- retrieveIndexVisitData(regs, initial.lables) # select people who had 'labels' Discharge Disposition at index visit

  # get return info for return visits that happened after 'sent home'labels' at index visit
  altered.data <- merge(x = regs,
                        y = data.frame(RegistrationNumber=altered.data[,c("RegistrationNumber")]), # index registration number
                        by.x="RegistrationNumberVisit1",
                        by.y="RegistrationNumber")
  # get data for return visit
  altered.data <- merge(x = all.data[,c("RegistrationNumber", "DischargeDisposition")],
                        y = altered.data,
                        by.x="RegistrationNumber",
                        by.y="RegistrationNumberVisit2")

  # filter returns by return labels
  altered.data <- altered.data %>% dplyr::filter(DischargeDisposition %in% return.visit.labels)

  altered.data <- data.frame(RegistrationNumber=altered.data$RegistrationNumberVisit1)

  return(altered.data)


}


# ------------------- Code ------------------- #

diff.days <- 3.00 # sets the difference between days (3.00 = 72 h)


# ------------------------------- CALCULATE RETURN VISITS ------------------------------- #

all.data <- fread("./../../../mybigdata/data/cleaned_wellSoft.csv", integer64 = "numeric", na.strings = c('""', "", "NA", "NULL"))
time.lapse <- readRDS("./timeLapse.rds")


# group data by Primary Medical Record Number and count the number of returns for each person
num.readmissions.all <- data.frame(all.data  %>% dplyr::group_by(PrimaryMedicalRecordNumber) %>%
                                     dplyr::mutate(num.returns = n()));

num.readmissions.all <- num.readmissions.all %>% dplyr::arrange(desc(num.returns),
                                                                PrimaryMedicalRecordNumber, StartOfVisit)

single.visit.ids.all <- (num.readmissions.all %>% dplyr::filter(num.returns == 1))
single.visit.ids.sent.home <- (single.visit.ids.all %>% dplyr::filter(DischargeDisposition %in% home.labels))$RegistrationNumber
single.visit.ids.necessary.visit <- (single.visit.ids.all %>% dplyr::filter(DischargeDisposition %in% admitted.labels))$RegistrationNumber
single.visit.ids.left <- (single.visit.ids.all %>% dplyr::filter(DischargeDisposition %in% left.labels))$RegistrationNumber
single.visit.ids.all <- single.visit.ids.all$RegistrationNumber

# sanity check
length(single.visit.ids.sent.home) + length(single.visit.ids.necessary.visit) + length(single.visit.ids.left) == length(single.visit.ids.all)

# time.lapse
#   - contains the time differences between visits for each patient
#   - calculated using all the registration codes
#   - therefore, need to remove visits that do not have corresponding wellSoft Data (all.data)

# remove visits that do not have corresponding wellSoft data
return.visits <- time.lapse %>% dplyr::filter(RegistrationNumberVisit1 %in% c(unique(all.data$RegistrationNumber)) & RegistrationNumberVisit2 %in% c(unique(all.data$RegistrationNumber)))

# remove vistis with no known difference between days
return.visits <- return.visits[!is.na(as.integer(return.visits$DifferenceInDays)),] # removes 886 with missing difference in days

# sanity check
length(intersect(single.visit.ids.all, return.visits$PrimaryMedicalRecordNumber)) == 0 # should be 0


return.visits$IsReturn <- ifelse(return.visits$DifferenceInDays < diff.days, 1, 0)

# count number of previous returns per person
return.visits$Ones <- 1
# create number of previous visits
return.visits <- as.data.frame(return.visits %>% dplyr::group_by(PrimaryMedicalRecordNumber) %>% dplyr::mutate(NumPreviousVisits=cumsum(Ones)))
# create number of previous return visits
num.return.visits <- as.data.frame(return.visits %>% dplyr::filter(DifferenceInDays < diff.days) %>% dplyr::group_by(PrimaryMedicalRecordNumber) %>% dplyr::mutate(NumPreviousReturnVisits=cumsum(Ones)))
return.visits$NumPreviousVisits <- return.visits$NumPreviousVisits - 1
num.return.visits$NumPreviousReturnVisits <- num.return.visits$NumPreviousReturnVisits - 1
num.return.visits$NumPreviousVisits <- NULL
return.visits <- dplyr::left_join(x = return.visits,
                                  y = num.return.visits)

return.visits[1, c("NumPreviousReturnVisits")] <- 0
# fill in number of previous return visits for non-72h returns
return.visits <- as.data.frame(return.visits %>%
  dplyr::mutate(NumPreviousReturnVisits = na.locf(NumPreviousReturnVisits)))
return.visits$Ones <- NULL



returned.within.72 <- return.visits %>% dplyr::filter(IsReturn == 1)
no.return.within.72 <- return.visits %>% dplyr::filter(IsReturn == 0)

index.ids.return.within.72.index.sent.home <- retrieveIndexVisitData(returned.within.72, home.labels)
index.ids.no.return.within.72.index.sent.home <- retrieveIndexVisitData(no.return.within.72, home.labels)

index.ids.necessary.return.within.72.index.sent.home <- retreiveReturnVisitData(returned.within.72, home.labels, admitted.labels)
index.ids.un.necessary.return.within.72.index.sent.home <- retreiveReturnVisitData(returned.within.72, home.labels, home.labels)
index.ids.left.before.seen.return.within.72.index.sent.home <- retreiveReturnVisitData(returned.within.72, home.labels, left.labels)


# sanity
nrow(index.ids.necessary.return.within.72.index.sent.home) + nrow(index.ids.un.necessary.return.within.72.index.sent.home) +
  nrow(index.ids.left.before.seen.return.within.72.index.sent.home) == length(index.ids.return.within.72.index.sent.home$RegistrationNumber)


all.data <- dplyr::left_join(x=all.data,
                             y=return.visits[,c("RegistrationNumberVisit1", "NumPreviousVisits", "NumPreviousReturnVisits")],
                             by=c("RegistrationNumber"="RegistrationNumberVisit1"))

all.data$NumPreviousReturnVisits <- ifelse(is.na(all.data$NumPreviousReturnVisits), 0, 
                                           all.data$NumPreviousReturnVisits)
all.data$NumPreviousVisits <- ifelse(is.na(all.data$NumPreviousVisits), 0, 
                                           all.data$NumPreviousVisits)

all.data <- all.data %>% dplyr::filter(RegistrationNumber %in% c(as.character(index.ids.return.within.72.index.sent.home$RegistrationNumber),
                                                                 as.character(index.ids.no.return.within.72.index.sent.home$RegistrationNumber),
                                                                 single.visit.ids.sent.home))

all.data <- all.data %>% dplyr::filter(!RegistrationNumber %in% c(as.character(index.ids.left.before.seen.return.within.72.index.sent.home$RegistrationNumber)))


all.data$WillReturn <- ifelse(all.data$RegistrationNumber %in%
                                         c(as.character(index.ids.return.within.72.index.sent.home$RegistrationNumber)),
                                       1, 0)
# sanity
sum(all.data$WillReturn==1) == nrow(index.ids.return.within.72.index.sent.home) - nrow(index.ids.left.before.seen.return.within.72.index.sent.home)
sum(all.data$WillReturn==0) == nrow(index.ids.no.return.within.72.index.sent.home) + length(single.visit.ids.sent.home)

# Class 0 --> no return 
all.data$Multiclass_Label <- 0

# Class 1 --> return with no admit
all.data$Multiclass_Label <- ifelse(all.data$RegistrationNumber %in% 
                                               as.character(index.ids.un.necessary.return.within.72.index.sent.home$RegistrationNumber),
                                             1, all.data$Multiclass_Label)

# Class 2 --> return with admit

all.data$Multiclass_Label <- ifelse(all.data$RegistrationNumber %in% 
                                               as.character(index.ids.necessary.return.within.72.index.sent.home$RegistrationNumber),
                                             2, all.data$Multiclass_Label)

# sanity
sum(all.data$Multiclass_Label ==1) == nrow(index.ids.un.necessary.return.within.72.index.sent.home)
sum(all.data$Multiclass_Label ==2) == nrow(index.ids.necessary.return.within.72.index.sent.home)



# 
# return.within.72.index.sent.home <- retreiveReturnVisitData(returned.within.72, home.labels)
# 
# return.within.72.index.necessary <- retreiveReturnVisitData(returned.within.72, admitted.labels)
# 
# return.within.72.index.left <- retreiveReturnVisitData(returned.within.72, left.labels)
# 
# # sanity
# nrow(return.within.72.index.sent.home) + nrow(return.within.72.index.necessary) +
#   nrow(return.within.72.index.left) == nrow(returned.within.72)
# 
# 
# no.return.within.72.index.sent.home <- retreiveReturnVisitData(no.return.within.72, home.labels)
# 
# no.return.within.72.index.necessary <- retreiveReturnVisitData(no.return.within.72, admitted.labels)
# 
# no.return.within.72.index.left <- retreiveReturnVisitData(no.return.within.72, left.labels)
# 
# # sanity
# nrow(no.return.within.72.index.sent.home) + nrow(no.return.within.72.index.necessary) +
#   nrow(no.return.within.72.index.left) == nrow(no.return.within.72)
# 


# ------------------------------- BUILD TRAIN/TEST SETS ------------------------------- #

# Two classification problems: first, predicting admitted/non-admitted for index visit
# and second,  predicting admitted/non-admitted for return visit for patients sent home at index visit.

# For each problem, since classes are so unbalanced, stratify train/test selection by labels of
# admitted/non-admitted


# NOTE: From talking with Olivia in ED, for classification 1 we are only focused on patients who were seen by an
#       MD at their index visit and were subsequently admitted or non-admitted. For classification 2, we are only
#       focused on patients who were sent home at their index visit and subsequently returned within 72h.
#       Thus, patients who left before seeing an MD at their index visit are excluded from classification 1 and 2, and patients
#       admitted at their index after seeing an MD visit are excluded from classification 2.



# sanity check -- no visits represented in both!
intersect(no.return.within.72$RegistrationNumberVisit1, returned.within.72$RegistrationNumberVisit1) # should be 0


train.ratio <- 0.80



# # CLASSIFICATION 1: predicting return/non-return for patients sent home at index visit
# 
# # a. return: return within 72 index sent home
# 
# class.1.return.reg.nums <- as.character(returned.within.72.index.sent.home$RegistrationNumber); length(class.1.return.reg.nums) # 30,925 visits
# 
# 
# # create train/test split --> random split
# class.1.return.train.num <- floor(length(class.1.return.reg.nums) * train.ratio); class.1.return.train.num; length(class.1.return.reg.nums) - class.1.return.train.num
# 
# #class.1.return.reg.nums.train <- sample(class.1.return.reg.nums, floor(train.ratio*length(class.1.return.reg.nums))); length(class.1.return.reg.nums.train)#class.1.return.reg.nums[1:class.1.return.train.num]; 
# class.1.return.reg.nums.test <- class.1.return.reg.nums[which(!class.1.return.reg.nums %in% (class.1.return.reg.nums.train))]; length(class.1.return.reg.nums.test)
# 
# # sanity
# all(c(class.1.return.reg.nums.train, class.1.return.reg.nums.test) %in% class.1.return.reg.nums)
# length(c(class.1.return.reg.nums.train, class.1.return.reg.nums.test)) == length(class.1.return.reg.nums)
# !any(class.1.return.reg.nums.train %in% class.1.return.reg.nums.test) # train/test separated!
# !any(class.1.return.reg.nums.test %in% class.1.return.reg.nums.train)
# 
# 
# 
# # b. non-return: single visits non-admitted; return within 72 index non-admitted; no return within 72 index admitted
# 
# class.1.non.return.reg.nums <- c(single.visit.ids.sent.home,
#                                  as.character(no.return.within.72.index.sent.home$RegistrationNumber)); length(class.1.non.return.reg.nums) # 415,814 total patient visits 
# 
# # create train/test split
# #class.1.non.return.train.num <- floor(length(class.1.non.return.reg.nums) * train.ratio); class.1.non.return.train.num; length(class.1.non.return.reg.nums) - class.1.non.return.train.num
# 
# class.1.non.return.reg.nums.train <- sample(class.1.non.return.reg.nums, floor(train.ratio*length(class.1.non.return.reg.nums)))#class.1.non.return.reg.nums[1:class.1.non.return.train.num]; length(class.1.non.return.reg.nums.train)
# class.1.non.return.reg.nums.test <- class.1.non.return.reg.nums[which(!class.1.non.return.reg.nums %in% class.1.non.return.reg.nums.train)]#class.1.non.return.reg.nums[(class.1.non.return.train.num+1):length(class.1.non.return.reg.nums)]; length(class.1.non.return.reg.nums.test)
# 
# # sanity
# intersect(class.1.return.reg.nums, class.1.non.return.reg.nums) # should be 0
# 
# 
# # now add together to create train and test
# 
# class.1.return.train.data <- all.data %>% dplyr::filter(RegistrationNumber %in% class.1.return.reg.nums.train)
# class.1.return.train.data$WillReturn <- 1
# class.1.return.train.data <- dplyr::left_join(x=class.1.return.train.data,
#                                               y=return.visits[,c("RegistrationNumberVisit1", "NumPreviousVisits", "NumPreviousReturnVisits")],
#                                               by=c("RegistrationNumber" = "RegistrationNumberVisit1"))
# 
# class.1.non.return.train.data <- all.data %>% dplyr::filter(RegistrationNumber %in% class.1.non.return.reg.nums.train)
# class.1.non.return.train.data$WillReturn <- 0
# class.1.non.return.train.data <- dplyr::left_join(x=class.1.non.return.train.data,
#                                               y=return.visits[,c("RegistrationNumberVisit1", "NumPreviousVisits", "NumPreviousReturnVisits")],
#                                               by=c("RegistrationNumber" = "RegistrationNumberVisit1"))
# class.1.non.return.train.data$NumPreviousVisits[is.na(class.1.non.return.train.data$NumPreviousVisits)] <- 0 
# class.1.non.return.train.data$NumPreviousReturnVisits[is.na(class.1.non.return.train.data$NumPreviousReturnVisits)] <- 0 
# 
# 
# class.1.return.test.data <- all.data %>% dplyr::filter(RegistrationNumber %in% class.1.return.reg.nums.test)
# class.1.return.test.data$WillReturn <- 1
# class.1.return.test.data <- dplyr::left_join(x=class.1.return.test.data,
#                                               y=return.visits[,c("RegistrationNumberVisit1", "NumPreviousVisits", "NumPreviousReturnVisits")],
#                                               by=c("RegistrationNumber" = "RegistrationNumberVisit1"))
# class.1.non.return.test.data <- all.data %>% dplyr::filter(RegistrationNumber %in% class.1.non.return.reg.nums.test)
# class.1.non.return.test.data$WillReturn <- 0
# class.1.non.return.test.data <- dplyr::left_join(x=class.1.non.return.test.data,
#                                                   y=return.visits[,c("RegistrationNumberVisit1", "NumPreviousVisits", "NumPreviousReturnVisits")],
#                                                   by=c("RegistrationNumber" = "RegistrationNumberVisit1"))
# class.1.non.return.test.data$NumPreviousVisits[is.na(class.1.non.return.test.data$NumPreviousVisits)] <- 0 
# class.1.non.return.test.data$NumPreviousReturnVisits[is.na(class.1.non.return.test.data$NumPreviousReturnVisits)] <- 0 
# 
# # # CLASSIFICATION 2: predicting admitted/non-admitted for patients who return after sent home at index visit
# # 
# # 
# # # admitted: 
# # 
# # # return.within.72.unique.return.admitted contains registration numbers for return visits that resulted in admission 
# # # --> need to get registration numbers of initial visits
# # class.2.admit.reg.nums <- merge(x=return.within.72.return.admitted,
# #                                 y=return.visits,
# #                                 by.x="RegistrationNumber",
# #                                 by.y="RegistrationNumberVisit2"); nrow(class.2.admit.reg.nums)
# # 
# # class.2.admit.reg.nums <- as.character(class.2.admit.reg.nums$RegistrationNumberVisit1)
# # 
# # # create train/test split
# # class.2.admit.train.num <- floor(length(class.2.admit.reg.nums) * train.ratio); class.2.admit.train.num; length(class.2.admit.reg.nums) - class.2.admit.train.num
# # class.2.admit.reg.nums.train <- class.2.admit.reg.nums[1:class.2.admit.train.num]; length(class.2.admit.reg.nums.train)
# # class.2.admit.reg.nums.test <- class.2.admit.reg.nums[(class.2.admit.train.num+1):length(class.2.admit.reg.nums)]; length(class.2.admit.reg.nums.test)
# # 
# # 
# # # non-admitted: 
# # 
# # class.2.non.admit.reg.nums <- merge(x=return.within.72.unique.return.non.admitted,
# #                                     y=return.visits,
# #                                     by.x="RegistrationNumber",
# #                                     by.y="RegistrationNumberVisit2"); nrow(class.2.non.admit.reg.nums)
# # 
# # class.2.non.admit.reg.nums <- as.character(class.2.non.admit.reg.nums$RegistrationNumberVisit1); length(class.2.non.admit.reg.nums )
# # 
# # # create train/test split
# # class.2.non.admit.train.num <- floor(length(class.2.non.admit.reg.nums) * train.ratio); class.2.non.admit.train.num; length(class.2.non.admit.reg.nums) - class.2.non.admit.train.num
# # class.2.non.admit.reg.nums.train <- class.2.non.admit.reg.nums[1:class.2.non.admit.train.num]; length(class.2.non.admit.reg.nums.train)
# # class.2.non.admit.reg.nums.test <- class.2.non.admit.reg.nums[(class.2.non.admit.train.num+1):length(class.2.non.admit.reg.nums)]; length(class.2.non.admit.reg.nums.test)
# # 
# # # sanity
# # intersect(class.2.admit.reg.nums, class.2.non.admit.reg.nums) # should be 0 overlap
# # 
# # 
# # # sanity
# # length(intersect(class.1.return.reg.nums, c(class.2.admit.reg.nums, class.2.non.admit.reg.nums))) == length(c(class.2.admit.reg.nums, class.2.non.admit.reg.nums))
# # all(c(class.2.admit.reg.nums, class.2.non.admit.reg.nums) %in% class.1.return.reg.nums) 
# # 
# # 
# # # now get full data
# # 
# # class.2.admit.train.data <- all.data %>% filter(RegistrationNumber %in% class.2.admit.reg.nums.train)
# # class.2.admit.train.data$AdmitOnReturn <- 1
# # class.2.non.admit.train.data <- all.data %>% filter(RegistrationNumber %in% class.2.non.admit.reg.nums.train)
# # class.2.non.admit.train.data$AdmitOnReturn <- 0
# # 
# # 
# # class.2.admit.test.data <- all.data %>% filter(RegistrationNumber %in% class.2.admit.reg.nums.test)
# # class.2.admit.test.data$AdmitOnReturn <- 1
# # class.2.non.admit.test.data <- all.data %>% filter(RegistrationNumber %in% class.2.non.admit.reg.nums.test)
# # class.2.non.admit.test.data$AdmitOnReturn <- 0
# # 
# # 
# 
# 
# 
# ## ------------------------------------  COMPILE FINAL TRAIN/TEST SETS  ------------------------------------ ##
# 
# set.seed(0)
# class.1.train <- rbind(class.1.return.train.data, class.1.non.return.train.data); class.1.train <- class.1.train[sample(nrow(class.1.train)),]
# class.1.test <- rbind(class.1.return.test.data, class.1.non.return.test.data); class.1.test <- class.1.test[sample(nrow(class.1.test)),]
# 
# # 
# # set.seed(0)
# # class.2.train <- rbind(class.2.admit.train.data, class.2.non.admit.train.data); class.2.train <- class.2.train[sample(nrow(class.2.train)),]
# # class.2.test <- all.data %>% filter(RegistrationNumber %in% class.2.reg.nums.test); class.2.test <- class.2.test[sample(nrow(class.2.test)),]
# 
# 
# 
# ## ------------------------------------  COMPILE FINAL TRAIN/TEST SETS  ------------------------------------ ##
# 
# 
# # SANITY 
# nrow(class.1.train); nrow(class.1.test); 
# #nrow(class.2.train); nrow(class.2.test)
# sum(class.1.train$WillReturn == 1); sum(class.1.train$WillReturn==0)
# sum(class.1.test$WillReturn == 1); sum(class.1.test$WillReturn==0)
# #sum(class.2.train$AdmitOnReturn == 1); sum(class.2.train$AdmitOnReturn==0)
# # check that training data is the size it should be
# nrow(class.1.train) == class.1.return.train.num + class.1.non.return.train.num
# #nrow(class.2.train) == class.2.admit.train.num + class.2.non.admit.train.num
# 
# fwrite(class.1.train, "./class_1_train_all_visits_final.csv")
# fwrite(class.1.test, "./class_1_test_all_visits_final.csv")

# save 

# 
# class.1.test$fullAddress <- ifelse(is.na(class.1.test$Address_Other_44),
#                                     paste(class.1.test$Address_43, class.1.test$City_45, class.1.test$Prov_46, class.1.test$Postal_Code_47, sep=", "),
#                                     paste(class.1.test$Address_Other_44, class.1.test$Address_43, class.1.test$City_45, class.1.test$Prov_46, class.1.test$Postal_Code_47, sep=", "))
# 

# class.1.test <- dplyr::left_join(x=class.1.test,
#                                  y=geoSpatial,
#                                  by=c("fullAddress"="Address"))
# 
# print(intersect(class.1.train$RegistrationNumber, class.1.test$RegistrationNumber))
# 
# fwrite(class.1.train, "./class_1_train_all_visits.csv")
# fwrite(class.1.test, "./class_1_test_all_visits.csv")

#saveRDS(class.2.train, "./class_2_train_all_visits.rds")
#saveRDS(class.2.test, "./class_2_test_all_visits.rds")


# filter return visits by discharge disposition at index visit
returned.within.72.index.admit <- retrieveIndexVisitData(returned.within.72, admitted.labels)
returned.within.72.index.sent.home <- retrieveIndexVisitData(returned.within.72, home.labels)
returned.within.72.index.left <- retrieveIndexVisitData(returned.within.72, left.labels)

# sanity 
nrow(returned.within.72.index.admit) + nrow(returned.within.72.index.sent.home) +
  nrow(returned.within.72.index.left) == nrow(returned.within.72)

# filter no return visits by discharge disposition at index visit
no.return.within.72.index.admit <- retrieveIndexVisitData(no.return.within.72, admitted.labels)
no.return.within.72.index.sent.home <- retrieveIndexVisitData(no.return.within.72, home.labels)
no.return.within.72.index.left <- retrieveIndexVisitData(no.return.within.72, left.labels)

# sanity
nrow(no.return.within.72.index.admit) + nrow(no.return.within.72.index.sent.home) +
  nrow(no.return.within.72.index.left) == nrow(no.return.within.72)



# Get reg numbers for return visits given discharge at index visit
return.within.72.all.returns <- retreiveReturnVisitData(returned.within.72, home.labels); nrow(return.within.72.all.returns)
return.within.72.return.admitted <- return.within.72.all.returns %>% dplyr::filter(DischargeDisposition %in% admitted.labels); nrow(return.within.72.return.admitted)
return.within.72.return.non.admitted <- return.within.72.all.returns %>% dplyr::filter(DischargeDisposition %in% home.labels); nrow(return.within.72.return.non.admitted)

set.seed(0)

