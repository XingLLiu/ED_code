library(data.table)
library(dplyr)
library(stringr)
library(caret)
library(foreign)

print('All required packages loaded')

print("Combining Data Sources to Produce EPIC.csv")
source("combineEPICData.R")
# Add here the python script for alerts
print("Preprocessing EPIC.csv to Produce preprocessed_EPIC.csv")
source("preprocessEPIC.R")