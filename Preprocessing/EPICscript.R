library(data.table)
library(dplyr)
library(stringr)
library(caret)
library(foreign)

print('All required packages loaded')

print("Combining Data Sources to Produce EPIC.csv")
source("combineEPICData.R")
print("Preprocessing EPIC.csv to Produce preprocessed_EPIC.csv")
source("preprocessEPIC.R")