import glob
import os
import pandas as pd

# Assume the current directory is 
# ED/ED_code/Preprocessing

path = '../../Epic_Sepsis_Alerts/'                          # Alert data folder
pathRN = 'RN_Sepsis_Alerts'                                 # RN folder
pathMD = 'MD_Sepsis_Triggers'                               # MD folder
pathData = '../data/EPIC_DATA/EPIC.csv'                  # Original data
pathSave = '../data/EPIC_DATA/EPIC_with_alerts.csv'           # Path of new data
os.chdir(path)

# Read the original EPIC data
EPIC = pd.read_csv(pathData, encoding = 'ISO-8859-1')
# Extract CSN
csn = EPIC['CSN']

# Append the alerts to EPIC as separate columns
pathLst = [pathRN, pathMD]
colLst = ['RN.Alert', 'MD.Alert']
for i in range(2):
    # Change to the folder
    folder = pathLst[i]
    os.chdir(folder)
    # Get all filenames
    allNames = [i for i in glob.glob('*.xlsx')]
    # Combine all files in the list
    combined_csv = pd.concat( [ pd.read_excel(f) for f in allNames ] )
    # Add additional column
    colName = colLst[i]
    EPIC[colName] = 0
    for id in combined_csv['CSN']:
        # Only append if present in EPIC
        if id in csn:
            # Assign 1 if an alert is raised
            EPIC[ EPIC['CSN'] == id, colName] = 1
    # Back to the parent folder
    os.chdir('../')
    print('Added ' + colName)


print('Writing to /data/EPIC_DATA/EPIC_with_alerts.csv')
# Write new data to EPICwithAlerts.csv
EPIC.to_csv(pathSave, index = False)
