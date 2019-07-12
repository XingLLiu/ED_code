# ----------------------------------------------------
# Path and savePath need to be configured before running
# ----------------------------------------------------


import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sklearn.preprocessing as skprep
import re
import numpy as np
import os
from scipy import stats

plt.style.use('bmh')

# Load file
# Path of file
path = '/home/xingliu/Documents/ED/data/EPIC_DATA/preprocessed_EPIC.csv'
EPIC = pd.read_csv(path, encoding = 'ISO-8859-1')

# Set current wd (for saving figures)
savePath = '/home/xingliu/Documents/code/figures'


# ----------------------------------------------------
# Overview of the dataset (6298 x 51)
print('Dimension of data:', EPIC.shape)
EPIC.info()

# Show data types
print( list(set(EPIC.dtypes.tolist())) )
numCols = EPIC.select_dtypes(include = ['float64', 'int64']).columns
catCols = [col for col in EPIC.columns if col not in numCols]

print('Categorical features:\n', catCols)
print('Numerical features:\n', numCols)

# ----------------------------------------------------
# Unify format of column names
colNames = list(EPIC.columns)
for i in range(len(colNames)):
    name = colNames[i]
    # Change names of the form 'XX XXX'
    if len(name.split(' ')) > 1:
        name = name.replace(' ', '.')
    # Change names of the form 'XxXxx'
    elif len(name.split('.')) == 1:
        nameList = re.findall('[A-Z][a-z]+', name)
        name = '.'.join(nameList)
    colNames[i] = name


# Assign FSA back
colNames[17] = 'FSA'
# Re-assign col names
EPIC.columns = colNames


# ----------------------------------------------------
# Percentage of missing data
missingPer = EPIC.isna().sum()
missingPer = missingPer / len(EPIC)
_ = missingPer.plot(kind = 'barh')


# ----------------------------------------------------
# Write summary of each feature into ./colSum.txt
file = open('colSum.txt', 'w')
# table = EPIC['Gender'].describe()
for col in colNames:
    _ = file.write('\n\n')
    _ = file.write(col)
    _ = file.write('\n')
    _ = file.write(EPIC[col].describe().to_string())

file.close()


# ----------------------------------------------------
# Discard the following features for modelling
colRem = ['First.ED.Provider', 'Last.ED.Provider', 'ED.Longest.Attending.ED.Provider',
            '']
colsWithMissingVal = list(missingPer.loc[missingPer > 0].index)






