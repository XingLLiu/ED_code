"""
THIS SCRIPT READS IN THE DATA FROM THE ~/ED_EPIC_DATA/ED_EPIC_DATA_JUNE2018_JUNE24_2019 
    FOLDER AND CONVERTS TO A READABLE CSV
@author: erik
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir_data",help="Folder where data can be read in",type=str)
parser.add_argument("--dir_output",help="Folder where data will be saved",type=str)
parser.add_argument("--dir_code",help="Folder where scripts live",type=str)
args = parser.parse_args()
# Assign arguments
dir_data = args.dir_data
dir_output = args.dir_output

# For beta testing
dir_code = '/home/xingliu/Documents/ED_code'
dir_output = dir_code + "/../ED/data/EPIC_DATA/ED_EPIC_DATA"
dir_data = dir_output + '/' + 'ED_EPIC_DATA_JUNE2018_JUNE24_2019'


# Impor the necessary modules
import os
from datetime import datetime
from calendar import monthrange
import re
import pandas as pd
import numpy as np
#import sys

# Set folder directory
os.chdir(dir_code + '../')

# Load in the support functions
from ED_support_funs import str_subset, epic_date_to_data#, str_which

##################################################
# ----- (1) CLEAN UP FILES NAMES IN OUTPUT ----- #

# Change the format of the output names
fn_output = pd.Series(os.listdir(dir_output))
fn_output = fn_output[fn_output.str.contains('csv$')]#.str.replace('ED_DATA_EPIC_|.csv','').tolist()
for ff in fn_output:
    if 'ED_DATA_EPIC' not in ff:
        print('No need to adjust file name %s' % ff)
        continue
    print('Adjusting file name %s' % ff)
    dd = ff.replace('.csv','').replace('ED_DATA_EPIC_','')
    # Change the format
    oo = re.sub('[0-9]','',dd)[0].upper() + re.sub('[0-9]','',dd)[1:3].lower() + '_' + re.sub('[A-Z]','',dd)
    oo = 'EPIC_' + oo + '.csv'
    os.rename(src=dir_output + '/' + ff,dst=dir_output + '/' + oo)
# Establish new names
fn_output = str_subset(os.listdir(dir_output),'^EPIC_[A-Za-z]{3}_[0-9]{2}.csv$')

##########################################################
# ----- (2) WRITE FILES FROM DATA FOLDER TO OUTPUT ----- #

fn_data = pd.Series(os.listdir(dir_data))
fn_data = fn_data[fn_data.str.contains('xlsx$')].reset_index(drop=True).str.replace('.xlsx','')
# Get the Notes and the Clindata
fn_notes = fn_data[fn_data.str.contains('Notes')].reset_index(drop=True).tolist()
fn_clindata = fn_data[fn_data.str.contains('ClinData')].reset_index(drop=True).tolist()
# Ensure there is a matching Notes for ClinData

#ff=fn_clindata[2]
for ff in fn_clindata:
    ff_date = datetime.strftime(datetime.strptime(ff.split('_')[0][0:3] + '-' + ff.split('_')[2],'%b-%Y'),'%b_%y')
    ff_out = 'EPIC_' + ff_date + '.csv'
    if ff_date in pd.Series(fn_output).str.replace('EPIC_|.csv','').tolist():
        print('Date %s is already in output, testing whether data should be appended' % ff_date)
        # Load the dates in from the csv file
        ff_dates_csv = pd.read_csv(dir_output + '/' + ff_out,usecols=['Arrived'])['Arrived']
        ff_dates_csv = epic_date_to_data(ff_dates_csv)
        ff_Y = datetime.strftime(ff_dates_csv[0],'%Y')
        ff_b = datetime.strftime(ff_dates_csv[0],'%m')
        ff_month_max = monthrange(year=int(ff_Y),month=int(ff_b))[1]
        ff_day_max = int(datetime.strftime(ff_dates_csv.max(),'%d'))
        if ff_day_max < ff_month_max:
            print('Appending file! Output only has max day of %i out of %i' % (ff_day_max, ff_month_max))
            df_ff_excel = pd.read_excel(dir_data + '/' + ff + '.xlsx')
            df_ff_csv = pd.read_csv(dir_output + '/' + ff_out)
            df_ff_csv.drop(columns=str_subset(df_ff_csv.columns,'Unnamed'),inplace=True)
            df_ff = pd.concat([df_ff_excel, df_ff_csv]).reset_index(drop=True)          
            print('There are a total of %i duplicate CSNs' % df_ff[['CSN']].duplicated().sum())
            df_ff.to_csv(dir_output + '/' + ff_out,index=False)
        else:
            print('Date %s already has full day count' % ff_date)
            continue
    else:
        print('Writing ClinData file for %s:' % ff_date)
        # Read full file
        df_ff = pd.read_excel(dir_data + '/' + ff + '.xlsx')    
        # Write to output
        df_ff.to_csv(dir_output + '/' + ff_out,index=False)

# Re-calculate output
fn_output = str_subset(os.listdir(dir_output),'^EPIC_[A-Za-z]{3}_[0-9]{2}.csv$')

##############################################################
# ----- (3) LOOP THROUGH OUTPUT AND CREATE SINGLE FILE ----- #

df_epic = pd.DataFrame([])
for ff in fn_output:
    print('Appending %s to master file' % ff)
    df_ff_csv = pd.read_csv(dir_output + '/' + ff,encoding='ISO-8859-1')
    df_ff_csv.drop(columns=str_subset(df_ff_csv.columns,'Unnamed'),inplace=True)
    df_epic = df_epic.append(df_ff_csv,ignore_index=True)

# Sort by the date
df_epic['Arrived_dt'] = epic_date_to_data(df_epic['Arrived'])
df_epic['Disch_dt'] = epic_date_to_data(df_epic['Disch Date/Time'],pat='%d/%m/%Y %H:%M')
df_epic['Arrived_dt'] = pd.Series(np.where(df_epic['Arrived_dt'].isnull(),df_epic['Disch_dt'],df_epic['Arrived_dt']))
df_epic = df_epic.sort_values(['Arrived_dt','Disch_dt']).reset_index(drop=True)

# Save file
df_epic.to_csv(dir_output + '/' + 'df_epic_data.csv',index=False)

########################################
# ----- (x) PROCESS THE NOTES ----- #


