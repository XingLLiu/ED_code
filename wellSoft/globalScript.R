# ================== RUNS ALL CODE ==================

source('dischargeCategories.R')
source('processRegistrationCodes.R')
source('cleanWellSoft.R')
source('preprocessWellSoft.R')
source('timeLapse.R')
source('calculateReturnVisits.R')
source('rollingDates.R')
source("helper_functions.R")

library(xlsx)
library(stringi)
library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(scales)
library(bizdays)
library(openxlsx)
library(stringr)
library(zoo)
library(CombMSC)
library(randomForest)
library(AUC)
library(data.table)
library(dplyr)
library(Amelia)
require(MLmetrics)
library(DMwR)
library(yardstick)
library(e1071)
library(caret)
library(glmnet)
library(rpart)
library(nnet)



data.path <- "./data/wellSoft_DATA/"

# ========================================================================
# All Steps:

# 1. PREPROCESS DATA
#   A. Clean data from scratch
#     - I. Read all raw data
#     - II. Preprocess Registration Codes (processRegistrationCodes.R)
#     - III. Clean wellSoft Data
#
# 2. CALCULATE OTHER STATS FROM CLEANED WELLSOFT
#   A. Time lapse 
#   B. Return visits
#
# ** LOAD CLEANED DATA
# -- can load all prior data sets here: cleaned_wellsoft, reg_codes, all_data, timeLapse, willReturn
#
# 3. PREPROCESS WELLSOFT DATA (not run all the way through)
#
# 4. RUN MODELS (not tested with new preprocess script)


# ==================  1. PREPROCESS DATA  ==================

# A. Clean data from scratch
#    - I. Read all raw data
reg_codes <- fread(paste0(data.path, "RegistrationCodes.csv"))
wellSoft <- fread(paste0(data.path, "raw_wellSoft.csv"))

#   - II. Preprocess Registration Codes (processRegistrationCodes.R)
reg_codes <- processRegistrationCodes(reg_codes, data.path)

#   - III. Clean wellSoft Data 
wellSoft <- cleanWellSoft(wellSoft, data.path)



# ================== 2. CALCULATE OTHER STATS FROM CLEANED WELLSOFT  ================== # 

# A. timeLapse.csv --> input cleaned_wellSoft.csv and processedRegistrationCodes.csv
timeLapse <- calculateTimeLapse(wellSoft, reg_codes, data.path)


# B. willReturn.csv --> input all_data.csv and timeLapse.csv
diff.days <- 3.00 # sets the difference between days (3.00 = 72 h)
all_data <- fread(paste0(data.path, "all_data.csv"))

willReturn <- calculateReturnVisits(all_data, timeLapse, diff.days, data.path)


##  ==========    (OR) LOAD CLEANED DATA   ==========

wellSoft <- fread(paste0(data.path, "cleaned_wellSoft.csv"))
reg_codes <- fread(paste0(data.path, "processedRegistrationCodes.csv"))
all_data <- fread(paste0(data.path, "all_data.csv"))
timeLapse <- fread(paste0(data.path, "timeLapse.csv"))
willReturn <- fread(paste0(data.path, "willReturn.csv"))


# ================== 3. PREPROCESS WELLSOFT DATA  ================== # 

wellSoft <- preprocessWellSoft(wellSoft, file.path)


# ==================  4. RUN MODELS  ==================

factors <- read.csv(paste0(file.path, "factors.csv")); factors <- as.character(factors$x)
numerics <- read.csv(paste0(file.path, "numerics.csv")); numerics <- as.character(numerics$x)

#preprocessed <- fread(inputFile, integer64 = "numeric", na.strings = c('""', "", "NA", "NULL", " "))


# ALL MODELS ::"LR", "glmnet", "tree", "CVRF", "SVM", "NB", "CVNN"


models.to.run <- c("LR")#, "glmnet", "tree", "CVRF", "SVM", "NB", "CVNN")



for (model in models.to.run) {
  
  print(paste("Running",model, "on 2 years of rolling data"))
  p.2 <- runModels(preprocessed, 2, 60, factors, models.to.run)
  
  print(paste("Running", model, "on 3 years of rolling data"))
  p.3 <- runModels(preprocessed, 3, 60, factors, models.to.run)
  
}


p.2.aucs <- plotStatsGraph(as.numeric(unlist(p.2[2])), as.numeric(unlist(p.2[3])),
                           2008, 2018, 1); p.2.aucs

# ==================  (EXTRA) CREATE PLOTS   ==================
source("graphs.R")



