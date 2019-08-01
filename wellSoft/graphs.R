
library(data.table)
library(ggplot2)
library(dplyr)
library(openxlsx)
library(gridExtra)
library(lemon)
source('http://www.sthda.com/upload/rquery_wordcloud.r')
library(tm)

# ================== 1. Load and preprocess data ==================

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


# ================== 2. Reg Codes MRN ==================

MRN <- reg_codes[,c("PrimaryMedicalRecordNumber", "VisitStartDate", "DischargeDisposition")]
MRN <- MRN[order(MRN$VisitStartDate),]
MRN$Year <- year(MRN$VisitStartDate)
MRN.unique <- MRN[!duplicated(MRN$PrimaryMedicalRecordNumber),]

MRNGenerationPlot <- ggplot(MRN.unique[MRN.unique$VisitStartDate > as.Date("2008-01-01"),], 
                            aes(x=VisitStartDate, fill=DischargeDisposition)) + geom_histogram(); MRNGenerationPlot

# ================== 3. Missingness ==================

completely_empty <- colSums(x=is.na(wellSoft))
completely_empty[completely_empty == nrow(wellSoft)]
colSums(all(wellSoft==""))
wellSoft$TimeInED <- difftime(wellSoft$Discharge_Time_276, wellSoft$Arrival_Time_9, units="hours")


# ================== 4. BASIC PLOTS ==================
head(times)
visits_hist <- ggplot() + geom_histogram(aes(x=wellSoft$Arrival_Time_9), bins=100) + 
  theme_bw() + 
  ggtitle("Number of Visits from May 2008 - April 2018") + 
  ylab("Number of Visits") + xlab("Date"); visits_hist

discharge_hist <- ggplot() + geom_histogram(aes(x=wellSoft$Discharge_Time_276), bins=100) + 
  theme_bw() + 
  ggtitle("Discharge Times from EDIS") + 
  ylab("Number of Visits") + xlab("Date"); discharge_hist


timeInED <- (wellSoft$TimeInED/24)
timeInED_filtered <- timeInED[timeInED > 0 & timeInED < 5]
 

timeInED_hist <- ggplot(wellSoft, aes(x=TimeInED/24/365)) + 
  geom_histogram(bins=100) + 
  xlab("Time In ED (Years)") +
  ylab("Number of Patients") + 
  ggtitle("Histogram of Number of Years Spent in the ED") + theme_bw(); timeInED_hist


timeInEDFiltered_hist <- ggplot(wellSoft %>% filter(TimeInED/24/364 > 0 & TimeInED/24/364 < 5),
                                aes(x=TimeInED/24/364)) + 
  geom_histogram(bins=100) + 
  xlab("Time In ED (Years)") +
  ylab("Number of Patients") + 
  theme_bw() + 
  ggtitle("Histogram of Number of Years Spent in the ED"); timeInEDFiltered_hist

# ================== 5. Presenting Complaint ==================

length(unique(wellSoft$Cedis_Cc_323))

unique.complaints <- unique(wellSoft$Cedis_Cc_323)
unique.complaints <- trimws(as.character(unlist(sapply(unique.complaints, function(x) ifelse(grepl("; ", x), strsplit(x, ";"), x)))))

unique.complaints <- unique(unique.complaints)
length(unique.complaints)

sort(unique.complaints)


temp <- sapply(wellSoft$Cedis_Cc_323, function(x) ifelse(grepl("; ", x), strsplit(x, ";"), x))
complaint.year.table <- data.frame("Arrival_Time" = rep(wellSoft$Arrival_Time_9, sapply(temp, length)), "Presenting_Complaint" = trimws(unlist(temp)))
complaint.year.table <- complaint.year.table[order(complaint.year.table$Arrival_Time),]

uniques <- unique(complaint.year.table$Presenting_Complaint)
  
results <- complaint.year.table[!duplicated(complaint.year.table$Presenting_Complaint),]

results$num.occurances <- sapply(uniques, function(x) sum(grepl(paste0("^\\Q", x, "\\E$"), complaint.year.table$Presenting_Complaint)))
dim(results); head(results)


# ================== 6. EXAMINE DIFFERENCES BETWEEN CALCULATED LENGTH OF STAY AND REG LENGTH OF STAY ==================

# Script that examines differences between start and end times in the two data sets
# NOTE: Can now add start times from wellSoft Data as well
#
# 6 graphs: 
#   1. WellSoft: Difference between Time left ED and Discharge Time
#   2. Registration Codes (unfiltered): Difference between calculated and encoded length of stay in minutes
#   3. Registration Codes (filtered): Difference between calculated and encoded length of stay in minutes
#   4. WellSoft vs Registration Codes: Difference between end times 
#   5. WellSoft vs Registration Codes (unfiltered): Difference between calculated (WS) and encoded (RC) length of stay in minutes
#   6. WellSoft vs Registration Codes (filtered): Difference between calculated (WS) and encoded (RC) length of stay in minutes


stend <- merge(x=reg_codes[,c("RegistrationNumber","StartOfVisit", "EndOfVisit", "LengthOfStayInMinutes")], 
                          y=wellSoft[,c("Pt_Accnt_5", "Time_Left_Ed_280","Discharge_Time_276")], 
                          by.x="RegistrationNumber", by.y="Pt_Accnt_5")

stend$CalculatedLengthOfStayRegCodes <- as.integer(difftime(stend$EndOfVisit, stend$StartOfVisit, units='mins'))
stend$CalculatedLengthOfStayWellSoft <- as.integer(difftime(stend$Discharge_Time_276, stend$StartOfVisit, units="mins"))

# create bounds for calculations 
stend.regCodes <- stend[0 < stend$CalculatedLengthOfStayRegCodes & stend$CalculatedLengthOfStayRegCodes < 8000,]
stend.wellSoft <- stend[0 < stend$CalculatedLengthOfStayWellSoft & stend$CalculatedLengthOfStayWellSoft < 2*(10^4),] # remove well soft calculated length of stay when > 2,000,000


# 1. WellSoft: Difference between Time left ED and Discharge Time
ggplot(stend, aes(x=Time_Left_Ed_280, y=Discharge_Time_276, colour=DischargeDisposition)) + geom_point() + 
  geom_abline(slope=1, intercept=0, colour="#E41A1C") + theme_bw() + 
  xlab("Well Soft Data: Time Left ED") + ylab("Well Soft Data: Discharge Time") + 
  ggtitle("Comparing Well Soft Variables Time Left ED and Discharge Time")


# 2. Registration Codes (unfiltered): Difference between calculated and encoded length of stay in minutes
ggplot(stend, aes(x=CalculatedLengthOfStayRegCodes, y=LengthOfStayInMinutes)) + geom_point() +
  geom_abline(slope=1, intercept=0, colour="#E41A1C") + theme_bw() + 
  xlab("Calculated Length of Stay Based on Start and End Times in Registration Data") + 
  ylab("Length of Stay in Registration Codes")

# 3. Registration Codes (filtered): Difference between calculated and encoded length of stay in minutes
ggplot(stend.regCodes, aes(x=CalculatedLengthOfStayRegCodes, y=LengthOfStayInMinutes)) + geom_point() +
  geom_abline(slope=1, intercept=0, colour="#E41A1C") + theme_bw() + 
  xlab("Calculated Length of Stay Based on Start and End Times in Registration Data (Filtered)") + 
  ylab("Length of Stay in Registration Codes")

# 4. WellSoft vs Registration Codes: Difference between end times 
ggplot(stend, aes(x=EndOfVisit, y=Discharge_Time_276)) + geom_point() + 
  geom_abline(slope=1, intercept=0, colour="#E41A1C") + theme_bw() + 
  xlab("Registration Data: End of Visit") + ylab("Well Soft Data: End of Visit") + 
  ggtitle("Comparing End of Visit in Registration Data with Discharge Time from WellSoft Data")


# 5. WellSoft vs Registration Codes (unfiltered): Difference between calculated (WS) and encoded (RC) length of stay in minutes
ggplot(stend, aes(x=CalculatedLengthOfStayWellSoft, y=LengthOfStayInMinutes)) + geom_point() +
  geom_abline(slope=1, intercept=0, colour="#E41A1C") + theme_bw() + 
  xlab("Calculated Length of Stay Based on Start and End Times in Well Soft Data") + 
  ylab("Length of Stay in Registration Codes") 

# 6. WellSoft vs Registration Codes (filtered): Difference between calculated (WS) and encoded (RC) length of stay in minutes
ggplot(stend.wellSoft, aes(x=CalculatedLengthOfStayWellSoft, y=LengthOfStayInMinutes)) + geom_point() +
  geom_abline(slope=1, intercept=0, colour="#E41A1C") + theme_bw() + 
  xlab("Calculated Length of Stay Based on Start and End Times in Well Soft Data (Filtered)") + 
  ylab("Length of Stay in Registration Codes")


# ================== 7. SEX PLOTS ==================


sexlang <- wellSoft[,c("Year", "Sex_41", "Language_56", "DaysOld")]
dim(sexlang); head(sexlang)
print("Process sexlang")
sexlang <- sexlang[!is.na(sexlang$DaysOld),];dim(sexlang)
sexlang <- sexlang[sexlang$Sex_41 !="" & sexlang$Sex_41 !="U", ]
sexlang$AgeGroup <- ifelse(sexlang$DaysOld >= (365*2), "Over Two", "Under Two")

sexlang <- sexlang %>% filter(Sex_41 %in% c("M", "F"))
under.2.male <- sum(sexlang$Sex_41=="M" & sexlang$AgeGroup=="Under Two")/sum(sexlang$AgeGroup=="Under Two")
over.2.male <- sum(sexlang$Sex_41=="M" & sexlang$AgeGroup=="Over Two")/sum(sexlang$AgeGroup=="Over Two")

sexlang$AvgPropMale <- ifelse(sexlang$AgeGroup=="Under Two", 
                              under.2.male,
                              over.2.male)

ggplot(data=sexlang, aes(x=Year, fill=Sex_41)) + 
  geom_bar(position = "dodge") + 
  theme_bw() + 
  facet_wrap(~AgeGroup) + 
  ggtitle("Number of Males vs Females April 2008 - May 2018") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 


ggplot(data=sexlang, aes(x=Year)) + 
  geom_bar(aes(fill=Sex_41), position = "fill") + 
  theme_bw() + 
  geom_hline(aes(yintercept=AvgPropMale)) + 
  geom_text(aes(0,AvgPropMale,label = round(AvgPropMale, 4)*100, vjust = -1)) + 
  facet_wrap(~AgeGroup) + 
  ggtitle("Percentage of Males vs Females April 2008 - May 2018") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  scale_y_continuous(labels = scales::percent_format())
  

# ================== 8. TEST SIGNIFICANCE  ==================


tTestData <- data.frame(table(sexlang[sexlang$Year != "2018" & sexlang$Year != "2008",c("Year", "Sex_41")])); head(tTestData)


# difference in means
ggplot(data=tTestData, aes(x=Sex_41, y=Freq)) + 
  geom_boxplot() + 
  theme_bw() + 
  ggtitle("Difference Between Male and Female Visits to SKH ED, 2008 - 2018")

# normality assumption -- both above 0.05 therefore not diff than normal
with(tTestData, shapiro.test(Freq[Sex_41 == "M"]))
with(tTestData, shapiro.test(Freq[Sex_41 == "F"]))


plot(density(tTestData$Freq[tTestData$Sex_41=="M"]))
plot(density(tTestData$Freq[tTestData$Sex_41=="F"]))
# test variances
var.test(Freq ~ Sex_41, data = tTestData)


#t-test
t.test(Freq ~ Sex_41, tTestData,  var.equal=T)

perMoreMen <- data.frame((table(sexlang[,c("Year", "Sex_41")])[,2] - table(sexlang[,c("Year", "Sex_41")])[,1]) / table(sexlang[,c("Year")]) * 100)
colnames(perMoreMen) <- c("Year", "Percentage")
png("wellSoft_percentMoreMen.png", height = 50*nrow(perMoreMen), width = 200*ncol(perMoreMen))
grid.table(perMoreMen)
dev.off()

# ================== 7. LANGUAGE AND SEX PLOTS ==================


order.langs <- sort(table(sexlang$Language_56), decreasing=T)
sexlang$Language_56 <- factor(as.character(sexlang$Language_56), names(order.langs))
lang.data <- data.frame(table(sexlang[,c("Year", "Sex_41", "Language_56")])); head(lang.data)# %>% group_by(Year, Sex_41) %>% summarize(total=sum(.))


ggplot(data=lang.data %>% filter(Language_56 %in% names(order.langs)[1:64] & Year %in% c(as.character(seq(2008, 2015, 1)))), 
              aes(x=Year, y=Freq, fill=Sex_41)) + 
  geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
  theme_bw() + 
  facet_rep_wrap(~Language_56, scales = "free_y") + 
  ggtitle("Number of Males vs Females July 2018 - June 2019 by Langauge") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 



ggplot(data=lang.data %>% filter(Language_56 %in% names(order.langs)[1:9] & Year %in% c(as.character(seq(2008, 2014, 1)))), 
       aes(x=Year, y=Freq, fill=Sex_41)) + 
  geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
  theme_bw() + 
  facet_rep_wrap(~Language_56, scales = "free_y") + 
  ggtitle("Number of Males vs Females July 2018 - June 2019 by Top 9 Langauge") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

top.langs <- names(order.langs[1:9])
ggplot(data=sexlang %>% filter(Language_56 %in% c(top.langs)), 
       aes(x=Year)) + 
  geom_bar(aes(fill=Sex_41), position = "fill") + 
  #geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
  theme_bw() + 
  facet_rep_wrap(~Language_56, scales = "free_y") + 
  ggtitle("Percentage of Males vs Females July 2018 - June 2019 by Top 9 Langauge") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  xlim(as.character(seq(2008, 2015, 1)))


# ================== 8. TOP LANGUAGES AND TOP DIAGNOSES ==================


createOtherCategory <- function(data, top.num) {
  freq.data <- as.data.frame(sort(table(data), decreasing=TRUE))
  
  top.x <- freq.data[1:top.num,]
  bottom.x <- data.frame(x="Other", y=sum(freq.data[top.num:nrow(freq.data), c("Freq")]))
  
  colnames(bottom.x) <- colnames(freq.data)
  
  total <- rbind(top.x, bottom.x)
  
  return(total)
  
}

# plot top n.langs languages
n.langs <- 30
total.non.eng <- createOtherCategory(wellSoft$Language_56[wellSoft$Language_56!="English" & wellSoft$Language_56!=""], n.langs)
total.non.eng <- total.non.eng[order(total.non.eng$Freq),]
non.eng.bar.chart <- ggplot(data=total.non.eng, aes(x=data, y=Freq)) +
  geom_bar(stat="identity") + theme_bw() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) + 
  xlab("Language") + ggtitle(paste("Top", n.langs, "Non-English Languages from April 2008 - April 2018")) + 
  ylab("Frequency") + coord_flip(); non.eng.bar.chart


# plot same as word cloud
word.cloud <- rquery.wordcloud(total.non.eng)

