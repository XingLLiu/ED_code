# ================== RUNS ALL CODE ==================

source('dischargeCategories.R')
source('processRegistrationCodes.R')
source('cleanWellSoft.R')
source('preprocessWellSoft.R')
source('timeLapse.R')
source('calculateReturnVisits.R')

data.path <- "/home/lebo/mybigdata/data/"

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
# 3. LOAD CLEANED DATA
#
# 4. PREPROCESS WELLSOFT DATA


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


##  ==========    3. (OR) LOAD CLEANED DATA   ==========

wellSoft <- fread(paste0(data.path, "cleaned_wellSoft.csv"))
reg_codes <- fread(paste0(data.path, "processedRegistrationCodes.csv"))
all_data <- fread(paste0(data.path, "all_data.csv"))
timeLapse <- fread(paste0(data.path, "timeLapse.csv"))
willReturn <- fread(paste0(data.path, "willReturn.csv"))


# ================== 3. PREPROCESS WELLSOFT DATA  ================== # 

wellSoft <- preprocessWellSoft(wellSoft, file.path)


# ==================  2. CREATE TRAIN/TEST SPLITS  ==================




# ==================  (EXTRA) CREATE PLOTS   ==================
source("graphs.R")



