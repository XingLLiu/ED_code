
library(data.table)
library(ggpubr)
library(ggplot2)
library(dplyr)
library(openxlsx)
library(gridExtra)
library(lemon)
source('http://www.sthda.com/upload/rquery_wordcloud.r')
library(tm)


## Creates plots of interesting patterns in the data
# 
# Steps: 
#
#   1. LOAD AND PREPROCESS DATA
#   2. REG CODES MRN
#   3. MISSINGNESS
#   4. ARRIVAL/DISCHARGE TIMES
#   5. PRESENTING COMPLAINT
#   6. EXAMINE DIFFERENCES BETWEEN CALCULATED LENGTH OF STAY AND REG LENGTH OF STAY
#   7. DIFFERENCES BETWEEN SEXES COMING TO ED
#       7.1 TEST SIGNIFICANCE
#       7.2 LANGUAGE AND SEX PLOTS
#   8. TOP LANGUAGES AND TOP DIAGNOSES
#   9. VARIABILITY OVER TIME


# ================== 1. LOAD AND PREPROCESS DATA ==================

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


# ================== 2. REG CODES MRN ==================

MRN <- reg_codes[,c("PrimaryMedicalRecordNumber", "VisitStartDate", "DischargeDisposition")]
MRN <- MRN[order(MRN$VisitStartDate),]
MRN$Year <- year(MRN$VisitStartDate)
MRN.unique <- MRN[!duplicated(MRN$PrimaryMedicalRecordNumber),]

MRNGenerationPlot <- ggplot(MRN.unique[MRN.unique$VisitStartDate > as.Date("2008-01-01"),], 
                            aes(x=VisitStartDate, fill=DischargeDisposition)) + geom_histogram(); MRNGenerationPlot

# ================== 3. MISSINGNESS ==================

completely_empty <- colSums(x=is.na(wellSoft))
completely_empty[completely_empty == nrow(wellSoft)]
colSums(all(wellSoft==""))
wellSoft$TimeInED <- difftime(wellSoft$Discharge_Time_276, wellSoft$Arrival_Time_9, units="hours")


# ================== 4. ARRIVAL/DISCHARGE TIMES ==================
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

# ================== 5. PRESENTING COMPLAINT ==================

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


# ================== 7. DIFFERENCES BETWEEN SEXES COMING TO ED ==================


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
  

# ================== 7.1 TEST SIGNIFICANCE  ==================


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

# ================== 7.2 LANGUAGE AND SEX PLOTS ==================


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


# ================== 9. VARIABILITY OVER YEARS ==================



plotCols <- function(data, colnames, max.percent, flag, dim.ncol, dim.nrow, other.flag.index=NA) {
  i <- 1
  j <- 1
  plot.list <- list()
  years <- c("2008", "2009", "2010", "2011", 
             "2012", "2013", "2014", "2015",
             "2016", "2017", "2018")
  
  y.tab.data <- melt(table(data$Year)) # count number of total visits per year
  
  for (col in colnames) {
    
    print(paste("Column", i,":", col))
    
    x <- data[,c(col, "Year"), with=FALSE]
    #x[x==""] <- NA
    x <- x[complete.cases(x),]
    tab.data <- melt(table(x$Year)); 
    tab.data$Var1 <- factor(tab.data$Var1, levels=years)
    
    tab.data <- merge(x=tab.data, y=y.tab.data, by="Var1", all.x = TRUE, all.y=TRUE)
    colnames(tab.data) <- c("Var1", "value", "totalVisits")
    tab.data$value[is.na(tab.data$value)] <- 0
    tab.data$percent <- round((tab.data$value/tab.data$totalVisits)*100, 2)
    colours <- rainbow(length(years))
    names(colours) <- years
    plot <- ggplot(data=tab.data, aes(x=Var1, y=value, fill=Var1)) +
      geom_bar(stat="identity") + ggtitle(col) + xlab("") + 
      ylab("") + scale_x_discrete(limits=c(years)) +
      scale_y_continuous(limits=c(0, 90000)) + 
      scale_fill_manual(values=colours) +
      theme(text = element_text(size=4),
            axis.text.x = element_text(angle=90),
            plot.title = element_text(size=6, face = 'bold')) +
      geom_text(data=tab.data, aes(x = Var1, y = value, label = percent),
                position = position_dodge(width = 1), hjust=-0.50, size = 3) + 
      coord_flip() + theme_bw(); plot
    
    plot.list[[j]] <- plot
    j <- j + 1
    if ((i %% 9) == 0| (i==120 & flag=='Date') | (i==253 & flag=="Other") |  
        ((!flag %in% c("Date","Other")) & (i %% other.flag.index==0))) {
      
      master_plot <- ggarrange(plotlist=plot.list, ncol=dim.ncol, nrow=dim.nrow,
                               common.legend = TRUE, legend="bottom"); master_plot
      if (flag=='Date') {
        path <- "./plots/variable_plots/dates/"
      } else if (flag=='Other') {
        path <- "./plots/variable_plots/others/"
      } else {
        path <- paste0("./plots/variable_plots/")
      }
      dir.create(file.path(path), showWarnings = FALSE, recursive = T)
      print(paste("Saving Plot:", i, "to", path))

      ggsave(filename=paste0(path, "plot_", i, ".png"), plot=master_plot)
      j <- 1
    }
    
    i <- i + 1
    
  }
}



calculateSDBands <- function(data, colnames, max.sd, flag, dim.ncol, dim.nrow, other.flag.index=NA) {
  
  if (length(colnames) < 20) {
    indicies <- c(length(colnames))
  } else {
    indicies <- seq(20, length(colnames), by=20); indicies <- c(indicies, length(colnames))
  }
  
  
  old.num <- 1
  i <- 1
  
  y.tab.data <- melt(table(data$Year))
  
  # collect stats
  all.stats <- data.frame(matrix(ncol=6, nrow=length(colnames)))
  colnames(all.stats) <- c("variable", "mean","sd", "max", "min", "colour")
  plot.list <- list()
  
  for (col.num in indicies) {
    print(i)
    q1 <- data[,c(colnames), with=FALSE][,old.num:col.num]
    
    if (!'Arrival_Time_9' %in% colnames(q1)) {
      q1$Arrival_Time_9 <- data$Arrival_Time_9
    }
    print("Melting dataframe")
    all.cols <- melt.data.table(q1, id.vars = 'Arrival_Time_9')
    #all.cols$Year <- format(as.POSIXct(all.cols$Arrival_Time_9), "%Y")
    all.cols <- merge(x=all.cols, y=unique(data[,c("Arrival_Time_9", "Year"), with=FALSE]),
                      by.x="Arrival_Time_9", by.y="Arrival_Time_9")
    
    all.cols$Arrival_Time_9 <- NULL
    all.cols <- all.cols[!is.na(all.cols$Year),]
    all.cols$value <- gsub("^$|^ $", NA, all.cols$value)
    
    print("Calculating missingness")
    missing <- data.frame(all.cols %>% group_by(Year, variable) %>% count(is.na(value)))
    missing <- missing[missing$is.na.value.==FALSE,]
    missing$is.na.value. <- NULL
    missing <- merge(missing, y.tab.data, by.x="Year", by.y="Var1")
    missing$percent <- (missing$n/missing$value) * 100
    
    summ.stats <- group_by(missing, variable)
    summ.stats <- summarise(summ.stats, mean=mean(percent), sd=sd(percent), max=max(percent), min=min(percent))
    summ.stats$colour <- 'red' # red not useful
    summ.stats$colour[summ.stats$mean != 0 & summ.stats$sd <= useful.sd] <- 'darkorange1'# high mean and low sd
    summ.stats$colour[summ.stats$mean >= 50 & summ.stats$sd <= useful.sd] <- 'green4'# high mean and low sd
    missing <- merge(x=missing, y=summ.stats[,c("variable", "colour")])
    
    print("Creating plot")
    
    p <- ggplot(data=missing) + 
      geom_boxplot(aes(x=variable, y=percent), fill='red', alpha=0.45) + 
      geom_boxplot(data=missing[missing$colour=='darkorange1',], aes(x=variable, y=percent), fill='darkorange1') + 
      geom_boxplot(data=missing[missing$colour=='green4',], aes(x=variable, y=percent), fill='green4') + 
      theme_bw() + 
      scale_y_continuous(limits=c(0, 120)) +
      geom_text(data=summ.stats[summ.stats$colour=='red',], aes(x = variable, y = max, label = round(sd, 2)),
                colour='red', position = position_dodge(width = 1), hjust=-0.50, size = 3) +
      geom_text(data=summ.stats[summ.stats$colour=='darkorange1',], aes(x = variable, y = max, label = round(sd, 2)), 
                colour='darkorange1', position = position_dodge(width = 1), hjust=-0.50, size = 3) +
      geom_text(data=summ.stats[summ.stats$colour=='green4',], aes(x = variable, y = max, label = round(sd, 2)), 
                colour='green4', position = position_dodge(width = 1), hjust=-0.50, size = 3) +
      scale_x_discrete(limits = rev(levels(missing$variable))) + 
      coord_flip() + 
      theme(axis.text.y = element_text(colour = rev(summ.stats$colour))); p
    
    plot.list[[i]] <- p
    all.stats[old.num:col.num,] <- summ.stats
    old.num <- col.num
    
    
    if (i %%4 == 0 | (flag=='Other' & i==ceiling(length(colnames)/20)) | (flag=='Date' & i==ceiling(length(colnames)/20)) | 
        ((!flag %in% c("Date","Other")) & (i %% other.flag.index==0))) {
      if (i==6) {
        start.index <- i-1
      } else if (i==13) {
        start.index <- i
      } else if (!flag %in% c("Date", "Other")) {
        start.index <- 1
      } else {
        start.index <- i - 3
      }
      master_plot <- ggarrange(plotlist=plot.list[start.index:(i)], ncol=dim.ncol, nrow=dim.nrow,
                               common.legend = TRUE, legend="bottom"); master_plot
      if (flag=='Date') {
        path <- paste0("./plots/significant_variables/dates/")
      } else if (flag=='Other') {
        path <- paste0("./plots/significant_variables/others/")
      } else {
        path <- paste0("./plots/significant_variables/")
      }
      print(paste("Saving plot", i, "in", path))
      dir.create(file.path(path), showWarnings = FALSE, recursive=T)
      ggsave(filename=paste0(path, "plot_", i, ".png"), plot=master_plot)
    }
    
    i <- i + 1
  }
  
  all.stats$variable <- colnames; head(all.stats); tail(all.stats)
  
  
  return (all.stats)
}

dateCols <- read.csv("./dates_colnames.csv")
dateCols <- as.character(dateCols$x)
head(dateCols)

otherCols <- read.csv("./other_colnames.csv")
otherCols <- as.character(otherCols$x); head(otherCols)

otherCols <- otherCols[!otherCols %in% dateCols]

max.percent.difference <- 2.00




plotCols(wellSoft[,c(dateCols, "Year"), with=FALSE], dateCols, max.percent.difference,
         "Date", 3, 3)
plotCols(wellSoft[,c(otherCols, "Year"), with=FALSE], otherCols, max.percent.difference,
         "Other", 3, 3)

useful.sd <- 3

# melt has bizarre behaviour when integer 64 are melted with other variables --> do them separately
other.types <- sapply(wellSoft[,c(otherCols, "Year", "Arrival_Time_9"), with=FALSE], class)

other.integer64s <-  names(other.types[other.types=='integer64'])
other.non.int64s <- otherCols[!otherCols %in% other.integer64s]

other.nonint64s.sds <- calculateSDBands(wellSoft[,c(other.non.int64s, "Year", "Arrival_Time_9"), with=FALSE], 
                                        other.non.int64s, 
                                        useful.sd, "Other", 2, 2)

other.int64s.sds <- calculateSDBands(wellSoft[,c(other.integer64s, "Year", "Arrival_Time_9"), with=FALSE], 
                                     other.integer64s, 
                                     useful.sd, "OtherInt64", 2, 2, 1)

date.sds <- calculateSDBands(wellSoft[,c(dateCols, "Year"), with=FALSE], dateCols[!dateCols %in% 'Arrival_Time_9'], 
                             useful.sd, "Date", 2, 2)

other.sds <- rbind(other.nonint64s.sds, other.int64s.sds)

nrow(other.sds); table(other.sds$colour)
nrow(date.sds); table(date.sds$colour)

# ======================= EXPLORE PREVIOUS VARIABLES USED ======================= #
green.other <- other.sds[other.sds$colour=='green4',]$variable
green.date <- date.sds[date.sds$colour=='green4',]$variable

green.vars <- c(green.other, green.date)

vars.in.previous.model <- c('Age_Dob_40',
                            'Sex_41', #Y
                            'Age_42',
                            'Discharge_Time_276',
                            'Address_43',
                            'City_45', #Y
                            'Prov_46', #Y
                            'Postal_Code_47',
                            'Language_56', #N
                            'Hc_Ver_69', #Y
                            'Hc_Issuing_Prov_70', #Y
                            'Method_Of_Arrival_Indexed_S_33', #N
                            'Cedis_Cc_323', #Y 
                            'Area_Of_Care_330', #Y
                            'Pt_Weight_350', #Y
                            'Staff_Md_Initial_378', #Y
                            'Discharge_Md_Np_379', #Y
                            'Cpso_385', # N
                            'Trainee_Initial_391', #N
                            'Condition_At_Disposition_481', #N,
                            'T2_Priority_12' #acuity
)

problematic.vars <- setdiff(vars.in.previous.model, green.vars); problematic.vars

plotCols(wellSoft[,c(problematic.vars, "Year"), with=FALSE], problematic.vars, max.percent.difference,
         "Small", 2, 3, other.flag.index=length(problematic.vars))

calculateSDBands(wellSoft[,c(problematic.vars, "Year", "Arrival_Time_9"), with=FALSE], problematic.vars, 
                 useful.sd, "Small", 1, 1, other.flag.index=1)






# ======================= EXPLORE NEW GOOD VARIABLES ======================= #

other.sds %>% filter(colour=='green4')
date.sds %>% filter(colour=='green4')
setdiff(green.vars, vars.in.previous.model) # 58 new variables
new.potential.vars <- setdiff(green.other, vars.in.previous.model); new.potential.vars # 48 new others 


exploreVariables <- function(start.index, end.index, data, new.potential.vars) {
  plots <- list()
  i <- 1
  j <- 1
  for (var in new.potential.vars[start.index:end.index]) {
    print(paste("Variable", j, ":", var))
    var.data <- data[,c(var), with=FALSE]
    #var.data <- gsub("^$|^ $", NA, var.data)
    
    if (length(unique(unlist(var.data))) < 10) {
      print(head(unique(var.data)))
      print(table(var.data))
      print(paste("Plotting", var, "at", i))
      p <- ggplot(data=data, aes_string(x="Year", fill=var)) + geom_bar()
      plots[[i]] <- p
      i <- i + 1
    }
    j <- j + 1
    print(paste("Number Unique:", length(unique(unlist(var.data)))))
    print(paste("% Missing:", round((sum(is.na(var.data))/nrow(data))*100, 3)))
    print("=======================================")
  }
  if (length(plots)==0) {
    return("NO PLOTS")
  }
  return(ggarrange(plotlist=plots, ncol=2, nrow=ceiling(length(plots)/2)))
  
  
}

length(new.potential.vars) # 48 others
green.1.10 <- exploreVariables(1, 10, wellSoft, new.potential.vars); green.1.10
green.11.20 <- exploreVariables(11, 20, wellSoft, new.potential.vars); green.11.20
green.21.30 <- exploreVariables(21, 30, wellSoft, new.potential.vars); green.21.30
green.31.40 <- exploreVariables(31, 40, wellSoft, new.potential.vars); green.31.40
green.41.49 <- exploreVariables(41, 49, wellSoft, new.potential.vars); green.41.49


test <- wellSoft[,c("Rn_T1_358", "Rn_T2_360"), with=FALSE]
test <- test[complete.cases(test),]
sum(test$Rn_T1_358==test$Rn_T2_360)/nrow(test)
length(setdiff(test$Rn_T1_358, test$Rn_T2_360))
length(setdiff(test$Rn_T2_360, test$Rn_T1_358))

useful.greens <- c("Pt_Clsfctn_73","Reg_Status_75", # factors as is
                   # difference between guardian addresses and patient address
                   "Guardian_Address_85", "Guardian_2_Address_113", 
                   #number of semi colons
                   "Staff_Md_History_389", 
                   # difference between RNs and dummy vars
                   "Rn_T2_360", "Rn_T1_358",
                   #factor
                   "Md_Disposition_Status_482",
                   #factor
                   "Disposition_Type_483")






###########################################################################3
# Explore whole data set #


exploreVariables <- function(data, new.potential.vars, type="Other", arrivalTime=wellSoft$Arrival_Time_9) {
  plots <- list()
  i <- 1
  j <- 1
  for (var in new.potential.vars) {
    print(paste("Variable", j, ":", var))
    var.data <- data[,c(var), with=FALSE]
    if (type=="Other") {
      print(head(unique(var.data)))
    } else if (type=="Date") {
      temp <- cbind(arrivalTime, data[,c(var), with=FALSE])
      temp <- temp[complete.cases(temp),]
      temp$HoursDiff <- as.numeric(round(difftime(unlist(temp[,c(var), with=FALSE]), temp$arrivalTime, 
                                                  units='hours'), 1))
      print(head(temp))
      print(tail(temp))
      print(paste("Average Hours Difference:", round(mean(temp$HoursDiff, na.rm = T), 2)))
    }
    
    if (length(unique(unlist(var.data))) < 20) {
      
      print(table(var.data))
      print(paste("Plotting", var, "at", i))
      p <- ggplot(data=data, aes_string(x="Year", fill=var)) + geom_bar()
      plots[[i]] <- p
      i <- i + 1
    } 
    
    
    j <- j + 1
    print(paste("Number Unique:", length(unique(unlist(var.data)))))
    print(paste("% Missing:", round((sum(is.na(var.data))/nrow(data))*100, 3)))
    print("=======================================")
  }
  if (length(plots)==0) {
    return("NO PLOTS")
  }
  return(ggarrange(plotlist=plots, ncol=2, nrow=ceiling(length(plots)/2)))
  
  
}


dateCols <- read.csv("./dates_colnames.csv")
dateCols <- as.character(dateCols$x)
head(dateCols)

otherCols <- read.csv("./other_colnames.csv")
otherCols <- as.character(otherCols$x); head(otherCols)

otherCols <- otherCols[!otherCols %in% dateCols]

# Other Variables
p.9 <- exploreVariables(wellSoft, otherCols[1:9]); p.9
p.18 <- exploreVariables(wellSoft, otherCols[10:18]); p.18
p.27 <- exploreVariables(wellSoft, otherCols[19:27]); p.27
p.36 <- exploreVariables(wellSoft, otherCols[28:36]); p.36
p.45 <- exploreVariables(wellSoft, otherCols[37:45]); p.45
p.54 <- exploreVariables(wellSoft, otherCols[46:54]); p.54
p.63 <- exploreVariables(wellSoft, otherCols[55:63]); p.63; print("63")
p.72 <- exploreVariables(wellSoft, otherCols[64:72]); p.72; print("72")
p.81 <- exploreVariables(wellSoft, otherCols[73:81]); p.81; print("81")
p.90 <- exploreVariables(wellSoft, otherCols[82:90]); p.90; print("90")
p.99 <- exploreVariables(wellSoft, otherCols[91:99]); p.99; print("99")
p.108 <- exploreVariables(wellSoft, otherCols[100:108]); p.108; print("108")
p.117 <- exploreVariables(wellSoft, otherCols[109:117]); p.117; print("117")
p.126 <- exploreVariables(wellSoft, otherCols[118:126]); p.126; print("126")
p.135 <- exploreVariables(wellSoft, otherCols[127:135]); p.135; print("135")
p.144 <- exploreVariables(wellSoft, otherCols[136:144]); p.144; print("144")
p.153 <- exploreVariables(wellSoft, otherCols[145:153]); p.153; print("153")
p.162 <- exploreVariables(wellSoft, otherCols[154:162]); p.162; print("162")
p.171 <- exploreVariables(wellSoft, otherCols[163:171]); p.171; print("171")
p.180 <- exploreVariables(wellSoft, otherCols[172:180]); p.180; print("180")
p.189 <- exploreVariables(wellSoft, otherCols[181:189]); p.189; print("189")
p.198 <- exploreVariables(wellSoft, otherCols[190:198]); p.198; print("198")
p.207 <- exploreVariables(wellSoft, otherCols[199:207]); p.207; print("207")
p.216 <- exploreVariables(wellSoft, otherCols[208:216]); p.216; print("216")
p.225 <- exploreVariables(wellSoft, otherCols[217:225]); p.225; print("225")
p.234 <- exploreVariables(wellSoft, otherCols[226:234]); p.234; print("234")
p.243 <- exploreVariables(wellSoft, otherCols[235:243]); p.243; print("243")
p.252 <- exploreVariables(wellSoft, otherCols[244:252]); p.252; print("252")

# Dates
p.date.9 <- exploreVariables(wellSoft, dateCols[1:9], "Date"); p.date.9
p.date.18 <- exploreVariables(wellSoft, dateCols[10:18], "Date"); p.date.18
p.date.27 <- exploreVariables(wellSoft, dateCols[19:27], "Date"); p.date.27
p.date.36 <- exploreVariables(wellSoft, dateCols[28:36], "Date"); p.date.36
p.date.45 <- exploreVariables(wellSoft, dateCols[37:45], "Date"); p.date.45
p.date.54 <- exploreVariables(wellSoft, dateCols[46:54], "Date"); p.date.54
p.date.63 <- exploreVariables(wellSoft, dateCols[55:63], "Date"); p.date.63
p.date.72 <- exploreVariables(wellSoft, dateCols[64:72], "Date"); p.date.72
p.date.81 <- exploreVariables(wellSoft, dateCols[73:81], "Date"); p.date.81
p.date.90 <- exploreVariables(wellSoft, dateCols[82:90], "Date"); p.date.90
p.date.99 <- exploreVariables(wellSoft, dateCols[91:99], "Date"); p.date.99
p.date.108 <- exploreVariables(wellSoft, dateCols[100:108], "Date"); p.date.108
p.date.117 <- exploreVariables(wellSoft, dateCols[109:117], "Date"); p.date.117






