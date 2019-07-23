library(plotROC)
library(dplyr)
require(gridExtra)
library(pROC)
library(lubridate)
library(reshape2)
library(scales)
library(RColorBrewer)
library(data.table)


# Processes Current Sepsis RN and MD Trigger # 
# input: folder containing Sepsis reports
# output: 
#         - RNreports.csv: all RN triggers
#         - MDreports.csv: all MD triggers
#
# NOTE: RN and MD currently only have Diagnosis column (no Primary.Dx or Diagnoses)
#       Therefore analysis is only done with Diagnosis 
#       Can change once have access to all columns


generateStats <- function(predictions, labels) {
  
  
  TP <- sum(predictions==1 & labels==1)
  TN <- sum(predictions==0 & labels==0)
  FP <- sum(predictions==1 & labels==0)
  FN <- sum(predictions==0 & labels==1)
  
  FPR <- FP / (FP + TN )
  FNR <- FN / (FN + TP)
  Sensitivity <- TP / (TP + FN)
  Specificity <- TN / (TN + FP)
  roc_obj <- roc(labels, predictions)
  auroc <- auc(roc_obj)
  
  return(data.frame(TP=TP, TN=TN, FP=FP, FN=FN, FPR=FPR, FNR=FNR, 
                    Sensitivity=Sensitivity, Specificity=Specificity,
                    AUROC=auroc))
}



path <- "./data/EPIC_DATA/"

# ============ 1. Load Files ================= #

# a. Get Nurses files
RN_files = Sys.glob(paste0(path, "Sepsis_Reports/*RN.csv")); RN_files
# First apply read.csv, then rbind
RN_reports = do.call(rbind, lapply(RN_files, function(x) read.csv(x, stringsAsFactors = FALSE)))
RN_reports$Arrived <- as.POSIXct(RN_reports$Arrived, tz="EST", format="%d/%m/%y %H%M")
RN_reports <- RN_reports[order(RN_reports$Arrived),]


# b. Get doctors files
MD_files = Sys.glob(paste0(path, "/Sepsis_Reports/*MD.csv")); MD_files
MD_reports = do.call(rbind, lapply(MD_files, function(x) read.csv(x, stringsAsFactors = FALSE)))
MD_reports$Arrived <- as.POSIXct(MD_reports$Arrived, tz="EST", format="%d/%m/%y %H%M")
MD_reports <- MD_reports[order(MD_reports$Arrived),]


# c. Get EPIC file
EPIC <- fread(paste0(path, "EPIC.csv"))


# ============ 2. Preprocess EPIC ================= #

EPIC$Arrived <- as.POSIXct(EPIC$Arrived, tz="EST", format="%d/%m/%y %H%M")
EPIC <- EPIC[order(EPIC$Arrived),]
EPIC <- EPIC[!is.na(EPIC$Arrived),]

EPIC$Month <- factor(month.abb[month(EPIC$Arrived)])

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


# ============ 3. Explore Definition of Sepsis ================= #
# currently,  RN and MD reports only have Diagnosis Column 
# --> Primary.Dx, Diagnosis, Diangoses all potentially indicate Sepsis 
#     --> change once get access to other columns/determine best way to define Sepsis

diagnosis <- grepl('*(S|s)epsis*', EPIC$Diagnosis); length(diagnosis)
diagnoses <- grepl('*(S|s)epsis*', EPIC$Diagnoses); length(diagnoses)
primary.dx <- grepl('*(S|s)epsis*', EPIC$Primary.Dx); length(primary.dx)

potential.Sepsis <- EPIC[(diagnosis | diagnoses | primary.dx),c("MRN", "CSN", "Diagnosis", "Diagnoses", "Primary.Dx")]
dim(potential.Sepsis)
nrow(potential.Sepsis[grepl("epsis", potential.Sepsis$Primary.Dx),])


table(potential.Sepsis[grepl("(S|s)epsis", potential.Sepsis$Diagnosis) & 
                         !grepl("(S|s)epsis", potential.Sepsis$Primary.Dx),c("Diagnoses",  "Primary.Dx")])

table(potential.Sepsis[grepl("(S|s)epsis", potential.Sepsis$Diagnoses) & 
                         !grepl("(S|s)epsis", potential.Sepsis$Primary.Dx),c("Diagnoses",  "Primary.Dx")])



# ============ 4. Create correct Sepsis Labels ================= #
# Find Sepsis labels --> need to change to correct label! 
RN_diagnosed_sepsis <- RN_reports[grep('*(S|s)epsis*', RN_reports$Diagnosis),]; dim(RN_diagnosed_sepsis)
MD_diagnosed_sepsis <- MD_reports[grep('*(S|s)epsis*', MD_reports$Diagnosis),]; dim(MD_diagnosed_sepsis)
EPIC_diagnosed_sepsis <- EPIC[grep('*(S|s)epsis*', EPIC$Diagnosis, useBytes = TRUE),]; dim(EPIC_diagnosed_sepsis)


EPIC$RN_prediction <- ifelse(EPIC$CSN %in% RN_reports$CSN, 1, 0)
EPIC$RN_True_Sepsis <- ifelse(EPIC$CSN %in% RN_diagnosed_sepsis$CSN, 1, 0)
EPIC$MD_prediction <- ifelse(EPIC$CSN %in% MD_reports$CSN, 1, 0)
EPIC$MD_True_Sepsis <- ifelse(EPIC$CSN %in% MD_diagnosed_sepsis$CSN, 1, 0)
EPIC$True_Sepsis <- ifelse(EPIC$CSN %in% EPIC_diagnosed_sepsis$CSN, 1, 0)




# ============ 5. Limit time frames of data ================= #
# Limit time frames on all three data sets so comparison of false negs/false positives
# are accurate

# filter RN_reports and MD_reports to end where EPIC data ends
max.date <- min(c(EPIC$Arrived[which.max(EPIC$Arrived)], 
                  RN_reports$Arrived[which.max(RN_reports$Arrived)],
                  MD_reports$Arrived[which.max(MD_reports$Arrived)])); max.date
EPIC <- EPIC %>% dplyr::filter(Arrived <= max.date)
RN_reports <- RN_reports %>% dplyr::filter(Arrived <= max.date)
MD_reports <- MD_reports %>% dplyr::filter(Arrived <= max.date)


# filter all data to start in october ---> when tools actually started working!
min.date <- max(c(EPIC$Arrived[which.min(EPIC$Arrived)], 
                  RN_reports$Arrived[which.min(RN_reports$Arrived)],
                  MD_reports$Arrived[which.min(MD_reports$Arrived)])); min.date
EPIC <- EPIC %>% dplyr::filter(Arrived >= min.date)
RN_reports <- RN_reports %>% dplyr::filter(Arrived >= min.date)
MD_reports <- MD_reports %>% dplyr::filter(Arrived >= min.date)


# ============ 5. Remove Young Patients ================= #
# Remove super young false negatives as their Sepsis labels could be incorrect
tool.false.negs <- (EPIC %>% dplyr::filter(RN_prediction==0 & 
                                             MD_prediction==0 & 
                                             True_Sepsis==1 & 
                                             Age.at.Visit <= 0.33))$CSN; tool.false.negs

EPIC <- EPIC %>% dplyr::filter(!CSN %in% tool.false.negs)
RN_diagnosed_sepsis <- RN_diagnosed_sepsis %>% dplyr::filter(!CSN %in% tool.false.negs)
MD_diagnosed_sepsis <- MD_diagnosed_sepsis %>% dplyr::filter(!CSN %in% tool.false.negs)
EPIC_diagnosed_sepsis <- EPIC_diagnosed_sepsis %>% dplyr::filter(!CSN %in% tool.false.negs)


# ============ 6. Manually check differences between tools vs Reality ================= #
#check for differences between RN and MD tool firing
RN_MD_Discrepency_CSN <- setdiff(RN_diagnosed_sepsis$CSN, MD_diagnosed_sepsis$CSN); RN_MD_Discrepency_CSN # 3 alerts fired by nurses tool and missed by MD tool
MD_RN_Discrepency_CSN <- setdiff(MD_diagnosed_sepsis$CSN, RN_diagnosed_sepsis$CSN); MD_RN_Discrepency_CSN # no patients fired by MD tool and missed by RN tool
EPIC_RN_Discrepency_CSN <- setdiff(EPIC$CSN, RN_diagnosed_sepsis$CSN); MD_RN_Discrepency_CSN # no patients fired by MD tool and missed by RN tool
EPIC_MD_Discrepency_CSN <- setdiff(EPIC$CSN, RN_diagnosed_sepsis$CSN); MD_RN_Discrepency_CSN # no patients fired by MD tool and missed by RN tool

EPIC %>% filter(CSN %in% EPIC_RN_Discrepency_CSN) %>% select(MRN, CSN, Diagnosis)
# No RN alert fired for 37 patients who had Sepsis
EPIC %>% filter(CSN %in% EPIC_MD_Discrepency_CSN) %>% select(MRN, CSN, Diagnosis)
# No RN alert fired for 40 patients who had Sepsis

# Stats
nrow(EPIC); nrow(RN_diagnosed_sepsis); nrow(MD_diagnosed_sepsis); nrow(EPIC_diagnosed_sepsis)



# MD predictions
MDSepsis_MD_Predictions <- generateStats(EPIC$MD_prediction, EPIC$MD_True_Sepsis); MDSepsis_MD_Predictions
TrueSepsis_MD_Predictions <- generateStats(EPIC$MD_prediction, EPIC$True_Sepsis); TrueSepsis_MD_Predictions


# RN predictions
RNSepsis_RN_Predictions <- generateStats(EPIC$RN_prediction, EPIC$RN_True_Sepsis); RNSepsis_RN_Predictions
TrueSepsis_RN_Predictions <- generateStats(EPIC$RN_prediction, EPIC$True_Sepsis); TrueSepsis_RN_Predictions


# ============ 7. (Optional) Jitter Age ================= #
# If you want to create some disturbance around age so not plotted as discrete (looks strange)
# set.seed(0)
# EPIC$Age.at.Visit.Disturbed <- EPIC$Age.at.Visit + rnorm(nrow(EPIC), mean = 0, sd = 0.2)
# EPIC$Age.at.Visit.Disturbed[EPIC$Age.at.Visit.Disturbed<0] <- 0.1


# ============ 8. Create Plots ================= #

# Whether or not to include RN tool as well as MD tool
plot.with.RN <- TRUE##FALSE#


if (plot.with.RN) {
  
  EPIC.FN <- EPIC$RN_prediction==0 & EPIC$MD_prediction==0 & EPIC$True_Sepsis==1
  EPIC.TP <- (EPIC$RN_prediction==1 | EPIC$MD_prediction==1) & EPIC$True_Sepsis==1
  EPIC.FP <- (EPIC$RN_prediction==1 | EPIC$MD_prediction==1) & EPIC$True_Sepsis==0

} else {

  EPIC.FN <- EPIC$MD_prediction==0 & EPIC$True_Sepsis==1
  EPIC.TP <- EPIC$MD_prediction==1 & EPIC$True_Sepsis==1
  EPIC.FP <- EPIC$RN_prediction==1 & EPIC$True_Sepsis==0

} 

# Create Colour column

EPIC$Colour <- "No Sepsis"
EPIC$Colour[EPIC.FN] <- "Sepsis Not Detected (False Negative)"
EPIC$Colour[EPIC.TP] <- "Sepsis Correctly Detected"  
EPIC$Colour[EPIC.FP] <- "Incorrectly Flagged As Sepsis (False Positive)"



plot.data <- EPIC %>% filter(True_Sepsis == 1 | (True_Sepsis == 0 & (RN_prediction == 1 | MD_prediction == 1)))
Tool <- factor(plot.data$Colour, levels=c("Incorrectly Flagged As Sepsis (False Positive)",
                                          "Sepsis Correctly Detected",
                                          "Sepsis Not Detected (False Negative)"))
plot.data.long <- table(plot.data[,c("Month", "Colour")])
plot.data.long <- melt(plot.data.long, id="Month")

# plot another version with no False Positives
plot.data.no.FPs <- EPIC %>% filter(True_Sepsis == 1)
Tool.no.FPs <- factor(plot.data.no.FPs$Colour, levels=c("Sepsis Correctly Detected",
                                          "Sepsis Not Detected (False Negative)"))
plot.data.long.no.FPs <- table(plot.data.no.FPs[,c("Month", "Colour")])
plot.data.long.no.FPs <- melt(plot.data.long.no.FPs, id="Month")


# order correctly
plot.data.long$Month <- factor(plot.data.long$Month, levels = month.abb)
plot.data.long.no.FPs$Month <- factor(plot.data.long.no.FPs$Month, levels = month.abb)


# plots
all.points <- qplot(plot.data$Arrived, plot.data$Age.at.Visit, colour=Tool, size=I(4)) + 
  ylab("Age at Visit (years)") + xlab("Date Arrived") + theme_bw() + 
  scale_color_manual(values=c("red2", "green3", "purple2"), name = "") +
  theme(legend.position="bottom") +
  ylim(-0.5, 18.5) + 
  scale_fill_manual(values = alpha(c("gray30", "gray50", "gray70"), .8)) + 
  ggtitle("Outcome of Sepsis Alerts: Patients Wihtout Sepsis Incorrectly Identified, Patients With Sepsis Correctly \nIdentified and Patients With Sepsis Missed By Current RN and MD Tools"); all.points

no.FPs <- qplot(plot.data.no.FPs$Arrived, plot.data.no.FPs$Age.at.Visit, colour=Tool.no.FPs, size=I(4)) + 
  ylab("Age at Visit (years)") + xlab("Date Arrived") + theme_bw() + 
  scale_color_manual(values=c("green3", "purple2"), name = "") +
  theme(legend.position="bottom") +
  ylim(-0.5, 18.5) + 
  scale_fill_manual(values = alpha(c("gray30", "gray50", "gray70"), .8)) + 
  ggtitle("Outcome of Sepsis Alerts: Patients With Sepsis Correctly Identified and Patients With Sepsis\nMissed By Current RN and MD Tools"); no.FPs

# two below were going to be used in a paper 
# --> looks strange because currently Sepsis Report data only runs Oct --> Feb
bar.percent <-  ggplot(plot.data.long, aes(x = Month, y = value, fill = Colour)) + 
  geom_bar(position = "fill",stat = "identity") +  
  scale_y_continuous(labels = percent_format()) + theme_bw() + 
  ylab("Percentage") + ggtitle("Outcome of Sepsis Alerts") + 
  scale_fill_manual(values=c("red2", "green3", "purple2"), name = "") +
  theme(legend.position="bottom", 
        legend.text=element_text(size=22),
        plot.title = element_text(size=26),
        axis.text=element_text(size=22),
        axis.title=element_text(size=24,face="bold"),
        axis.title.y = element_text(margin = margin(t = 0, r = 30, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 30, r = 0, b = 0, l = 0))) +
      guides(fill=guide_legend(ncol=1,
                           keywidth=0.4,
                           keyheight=0.4,
                           default.unit="inch")); bar.percent


bar.count <-  ggplot(plot.data.long, aes(x = Month, y = value, fill = Colour)) + 
  geom_bar(stat = "identity")  + theme_bw() + 
  ylab("Number of Patients") + ggtitle("Outcome of Sepsis Alerts") + 
  scale_fill_manual(values=c("red2", "green3", "purple2"), name = "") +
  theme(legend.position="bottom", 
        legend.text=element_text(size=24),
        plot.title = element_text(size=26),
        axis.text=element_text(size=24),
        axis.title=element_text(size=24,face="bold"),
        axis.title.y = element_text(margin = margin(t = 0, r = 30, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 30, r = 0, b = 0, l = 0))) +
  guides(fill=guide_legend(ncol=1,
        keywidth=0.4,
        keyheight=0.4,
        default.unit="inch")); bar.count

# ============ 9. Save RN and MD reports ================= #
fwrite(x = RN_reports, file = paste0(path, "RNreports.csv"))
fwrite(x = MD_reports, file = paste0(path, "MDreports.csv"))

