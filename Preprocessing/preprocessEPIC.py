"""
Python implementation of ~/Preprocessing/preprocessEPIC.R
"""

## --- FOR BETA TESTING --- #
#import os
#os.chdir("/home/erik/Documents/projects/ED/ED_code/Preprocessing")

# ---- MODULE SUPPORT ---- #
import os
import pandas as pd
import numpy as np

# ---- FUNCTION SUPPORT ---- #

path = "../../data/EPIC_DATA/"
print(os.listdir(path))

# ============ 1. LOAD DATA ================= #

# Loop through each column and parse
EPIC = pd.read_csv(path + "EPIC.csv",encoding='ISO-8859-1')

# ============ 2. PROCESS DATES ================= #

# Arrival times
EPIC['Arrived'] = pd.to_datetime(EPIC['Arrived'].str.strip().str.replace('\\s','-'),
                                format='%d/%m/%y-%H%M')
# Arrival month
EPIC['ArrivalMonth'] = EPIC['Arrived'].dt.strftime('%m')
# Arrival hour
EPIC['ArrivalNumHoursSinceMidnight'] = \
    pd.to_numeric(EPIC['Arrived'].dt.strftime('%H').str.replace('^0','',regex=True),errors='coerce')
# Day of the week
EPIC['ArrivalDayOfWeek'] = EPIC['Arrived'].dt.strftime('%A')
# Discharge date time
EPIC['Disch.Date.Time'] = pd.to_datetime(EPIC['Disch.Date.Time'].str.strip().str.replace('\\s','-'),
                                        format='%d/%m/%Y-%H%M',errors='coerce')
# Arrival room hour
EPIC['Arrival.to.Room'] = (pd.to_datetime(EPIC['Arrival.to.Room'],format='%H:%M') \
                                - pd.to_datetime('1900-01-01')).astype('timedelta64[m]')
## Time at which roomed --- NOT USING VARIABLE CURRENTLY
#EPIC['Roomed']

# ============ 3. CREATE RETURN INDICATOR ================= #

pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 101)

cnt = ['CSN','MRN','Arrived','Dispo','Disch.Date.Time','Discharge.Admit.Time']

# Find the duplicated MRN
repeat_MRN = EPIC['MRN'][EPIC['MRN'].duplicated()].unique()
repeat_df = EPIC[cnt][EPIC['MRN'].isin(repeat_MRN)]
repeat_df.reset_index(drop=True,inplace=True)
repeat_df.sort_values(['MRN','Disch.Date.Time'],inplace=True)
repeat_df['lag_discharge'] = repeat_df.groupby('MRN')['Disch.Date.Time'].shift(1)
repeat_df['hour_diff'] = (repeat_df['Disch.Date.Time'] - repeat_df.lag_discharge).astype('timedelta64[m]') / 60
repeat_df['WillReturn'] = (repeat_df.hour_diff <= 72).astype(int)

# Q1: If Dispo=='Discharge' or Dispo=='Admit' what proportion of Disch.Date.Time values are missing?
EPIC['Disch.Date.Time'][EPIC.Dispo=='Discharge'].isnull().value_counts()
EPIC['Disch.Date.Time'][EPIC.Dispo=='Admit'].isnull().value_counts() # Around 185
# A1: None for Dispo==Discharge and <200 for Dispo==Admit

# Q2: Is Discharge.Admit.Time is a value, is Dispo always admit?
EPIC[EPIC['Discharge.Admit.Time'].str.contains('No')].Dispo.value_counts()
EPIC[~EPIC['Discharge.Admit.Time'].str.contains('No')].Dispo.value_counts()

# Q2: Can Length of Stay be captured by difference between Arrived and Disch.Date.Time?
#ED.Completed.Length.of.Stay..Hours.                                                 18.8
#ED.Completed.Length.of.Stay..Minutes.                                               1126
# A2: 


# ============ 4. CLEAN FACTORS ================= #




