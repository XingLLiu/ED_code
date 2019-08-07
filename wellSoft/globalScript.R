# ================== RUNS ALL CODE ==================

source('dischargeCategories.R')
source('processRegistrationCodes.R')
source('cleanWellSoft.R')
source('preprocessWellSoft.R')
source('timeLapse.R')

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




# ========================================================================
# All Steps:

# 1. PREPROCESS DATA
#   A. Clean data from scratch
#     - I. Read all raw data
#     - II. Preprocess Registration Codes (processRegistrationCodes.R)
#     - III. Clean wellSoft Data
#   B. Load cleaned Data 
#
# 2. CALCULATE OTHER STATS FROM CLEANED WELLSOFT
#   A. Time lapse 
#   B. Return visits
#
# 3. PREPROCESS WELLSOFT DATA


# ==================  1. PREPROCESS DATA  ==================

# A. Clean data from scratch
#    - I. Read all raw data
reg_codes <- fread(paste0(data.path, "RegistrationCodes.csv"))
wellSoft <- fread(paste0(data.path, "raw_wellSoft.csv"))

#   - II. Preprocess Registration Codes (processRegistrationCodes.R)
reg_codes <- processRegistrationCodes(reg_codes, data.path)

#   - III. Clean wellSoft Data 
wellSoft <- cleanWellSoft(wellSoft, data.path)

##  ==========    OR   ==========

# B. Load Preprocessed Data 
wellSoft <- fread(paste0(data.path, "cleaned_wellSoft.csv"))
reg_codes <- fread(paste0(data.path, "processedRegistrationCodes.csv"))



# ================== 2. CALCULATE OTHER STATS FROM CLEANED WELLSOFT  ================== # 

# A. timeLapse.csv --> input cleaned_wellSoft.csv and processedRegistrationCodes.csv
timeLapse <- calculateTimeLapse(wellSoft, reg_codes, data.path)



# ================== 3. PREPROCESS WELLSOFT DATA  ================== # 

wellSoft <- preprocessWellSoft(wellSoft, file.path)


# ==================  2. CREATE TRAIN/TEST SPLITS  ==================




# ==================  (EXTRA) CREATE PLOTS   ==================
source("graphs.R")



