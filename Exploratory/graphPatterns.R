# Explores patterns in Data

library(data.table)
library(gridExtra)
library(dplyr)
library(ggplot2)
library(lemon)


# ===================== 1. Load Data and preprocess =====================

path <- "./data/EPIC_DATA/"
path <- "./"
EPIC <- fread(paste0(path, "EPIC.csv"))
EPIC$Arrived <- as.POSIXct(EPIC$Arrived, tz="EST", format="%d/%m/%y %H%M")
EPIC$Month <- factor(month.name[month(EPIC$Arrived)], levels=(c(unique(month.name[month(EPIC$Arrived)]))))

EPIC$Age.at.Visit <- as.character(EPIC$Age.at.Visit)
month.indicies <- grep("m.o", EPIC$Age.at.Visit)
year.indicies <- grep("y.o", EPIC$Age.at.Visit)
day.indicies <- grep("days", EPIC$Age.at.Visit)
week.indicies <- grep("wk.o", EPIC$Age.at.Visit)

EPIC$Age.at.Visit[month.indicies] <- as.numeric(gsub("m\\.o\\.", "", EPIC$Age.at.Visit[month.indicies]))/12
EPIC$Age.at.Visit[year.indicies] <- as.numeric(gsub("y\\.o\\.", "", EPIC$Age.at.Visit[year.indicies]))
EPIC$Age.at.Visit[day.indicies] <- as.numeric(gsub("days", "", EPIC$Age.at.Visit[day.indicies])) / 365
EPIC$Age.at.Visit[week.indicies] <- as.numeric(gsub("wk.o", "", EPIC$Age.at.Visit[week.indicies])) / 52

EPIC$Age.at.Visit <- as.numeric(EPIC$Age.at.Visit)


# ===================== 2. Examine differences in numbers =====================


gender.df <- table(EPIC[,c("Month", "Gender")]); gender.df

png("EPIC_GenderTable.png", height = 50*nrow(gender.df), width = 200*ncol(gender.df))
grid.table(gender.df)
dev.off()


# ===================== 3. Plot differences for over/under 2 =====================


sexData <- EPIC[,c("Arrived", "Month", "Gender", "Pref.Language", "Age.at.Visit")]; head(sexData)
sexData$OverTwo <- ifelse(sexData$Age.at.Visit > 2.00, "Over Two", "Under Two")
sexData <- sexData %>% filter(Gender %in% c("M", "F"))
prop.over.2 <- sum(sexData$Gender=="M" & sexData$OverTwo=="Over Two")/sum(sexData$OverTwo=="Over Two")
prop.under.2 <- sum(sexData$Gender=="M" & sexData$OverTwo=="Under Two")/sum(sexData$OverTwo=="Under Two")
sexData$AvgPropMale <- ifelse(sexData$OverTwo=="Over Two", prop.over.2, prop.under.2)
sexData$Gender <- factor(sexData$Gender, levels=c("F", "M"))


ggplot(data=sexData, aes(x=Month, fill=Gender)) + 
  geom_bar(position = "dodge") + 
  theme_bw() + 
  facet_wrap(~OverTwo) + 
  ggtitle("Number of Males vs Females July 2018 - June 2019") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 


ggplot(data=sexData, aes(x=Month)) + 
  geom_bar(aes(fill=Gender), position = "fill") + 
  theme_bw() + 
  geom_hline(aes(yintercept=AvgPropMale)) + 
  geom_text(aes("July",AvgPropMale,label = round(AvgPropMale, 4)*100, vjust = -1)) + 
  facet_wrap(~OverTwo) + 
  ggtitle("Percentage of Males vs Females July 2018 - June 2019") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(labels = scales::percent_format())
  
perMoreMen <- data.frame((table(sexData[,c("Month", "Gender")])[,2] - table(sexData[,c("Month", "Gender")])[,1]) / table(sexData[,c("Month")]) * 100)
colnames(perMoreMen) <- c("Month", "Percentage")
png("EPIC_percentMoreMen.png", height = 50*nrow(gender.df), width = 200*ncol(gender.df))
grid.table(perMoreMen)
dev.off()


# ===================== 3. Test Significances between Means =====================

tTestData <- data.frame(table(sexData[,c("Month", "Gender")])); tTestData

# difference in means
ggplot(data=tTestData, aes(x=Gender, y=Freq)) + geom_boxplot()

# normality assumption -- both above 0.05 therefore not diff than normal
with(tTestData, shapiro.test(Freq[Gender == "M"]))
with(tTestData, shapiro.test(Freq[Gender == "F"]))

# test variances
var.test(Freq ~ Gender, data = tTestData)

#t-test
t.test(Freq ~ Gender, tTestData, var.equal=T)


# ===================== 4. Plot differences for languages =====================
order.langs <- sort(table(sexData$Pref.Language), decreasing=T)
sexData$Pref.Language <- factor(as.character(sexData$Pref.Language), names(order.langs))
lang.data <- data.frame(table(sexData[,c("Month", "Gender", "Pref.Language")])); head(lang.data)# %>% group_by(Month, Gender) %>% summarize(total=sum(.))



ggplot(data=lang.data , aes(x=Month, y=Freq, fill=Gender)) + 
  #geom_bar(position = "dodge") + 
  geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
  theme_bw() + 
  facet_rep_wrap(~Pref.Language, scales = "free_y") + 
  ggtitle("Number of Males vs Females July 2018 - June 2019 by Langauge") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

ggplot(data=lang.data %>% filter(Pref.Language %in% c(names(order.langs[1:9]))), 
       aes(x=Month, y=Freq, fill=Gender)) + 
  #geom_bar(position = "dodge") + 
  geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
  theme_bw() + 
  facet_rep_wrap(~Pref.Language, scales = "free_y") + 
  ggtitle("Number of Males vs Females July 2018 - June 2019 by Top 9 Langauge") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

top.langs <- names(order.langs[1:9])
ggplot(data=sexData %>% filter(Pref.Language %in% c(top.langs)), 
       aes(x=Month)) + 
  geom_bar(aes(fill=Gender), position = "fill") + 
  #geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
  theme_bw() + 
  facet_rep_wrap(~Pref.Language, scales = "free_y") + 
  ggtitle("Percentage of Males vs Females July 2018 - June 2019 by Top 9 Langauge") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 


# ===================== 5. Plot top diagnoses (with Sepsis highlighted) =====================

createOtherCategory <- function(data, top.num) {
  freq.data <- as.data.frame(sort(table(data), decreasing=TRUE))
  
  top.x <- freq.data[1:top.num,]
  bottom.x <- data.frame(x="Other", y=sum(freq.data[top.num:nrow(freq.data), c("Freq")]))
  
  colnames(bottom.x) <- colnames(freq.data)
  
  total <- rbind(top.x, bottom.x)
  
  return(total)
  
}

n.diag <- 101
## ASSUMES Primary.Dx HAS BEEN CHANGED TO CREATE SEPSIS LABLE THAT ENCOMPASES ALL SEPSIS CASES! 
most.freq.diagnoses <- createOtherCategory(EPIC$Primary.Dx[!EPIC$Primary.Dx == 'None'], n.diag); most.freq.diagnoses
most.freq.diagnoses$Highlight <- ifelse(most.freq.diagnoses$data=="Sepsis", "Yes", "No")

bold.italic.16.text <- element_text(color = "black", size=9)

top.diagnoses <- ggplot(data=most.freq.diagnoses[1:100,], aes(x=data, y=Freq, fill=Highlight)) +
  geom_bar(stat="identity") + theme_bw() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) + 
  scale_fill_manual( values = c( "Yes"="tomato", "No"="gray" ), guide = FALSE ) + 
  labs(x="Primary Diagnosis", title="Top 100 Primary Diagnoses from July 2018 - Febrary 2019", y="Frequency") + 
  theme(axis.text.x = bold.italic.16.text); top.diagnoses 


