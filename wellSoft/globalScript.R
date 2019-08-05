# ================== RUNS ALL CODE ==================

source('dischargeCategories.R')
source('processRegistrationCodes.R')
source('cleanWellSoft.R')
source('preprocessWellSoft.R')

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
# A. Preprocess Data From Scratch
#   - I. Read all raw data
#   - II. Preprocess Registration Codes (processRegistrationCodes.R)
#   - III. Clean wellSoft Data
#   - IV. Preprocess Wellsoft Data
# B. Load Preprocessed Data 


# ==================  1. PREPROCESS DATA  ==================

# A. Preprocess Data From Scratch
#    - I. Read all raw data
reg_codes <- read.xlsx(paste0(data.path, "RegistrationCodes.xlsx", sheet=1, startRow=1, colNames=TRUE))
wellSoft <- fread(paste0(data.path, "raw_wellSoft.csv"))

#   - II. Preprocess Registration Codes (processRegistrationCodes.R)
reg_codes <- processRegistrationCodes(reg_codes, data.path)

#   - III. Clean wellSoft Data 
wellSoft <- cleanWellSoft(wellSoft, file.path)

#   - IV. Preprocess Wellsoft Data
wellSoft <- preprocessWellSoft(wellSoft, file.path)


# B. Load Preprocessed Data 
wellSoft <- fread(paste0(data.path, "preprocessed_wellSoft.csv"))
reg_codes <- fread(paste0(data.path, "processedRegistrationCodes.xlsx"))





# ==================  2. CREATE TRAIN/TEST SPLITS  ==================




# ==================  (EXTRA) CREATE PLOTS   ==================
source("graphs.R")



