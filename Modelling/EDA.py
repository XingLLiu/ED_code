# ----------------------------------------------------
# Path and savePath need to be configured before running
# ----------------------------------------------------
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sklearn.preprocessing as skprep
from sklearn import model_selection
from sklearn import linear_model, impute, ensemble
import sklearn as sk
from imblearn.over_sampling import SMOTE
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

# Discard the following features for modelling
colRem = ['First.ED.Provider', 'Last.ED.Provider', 'ED.Longest.Attending.ED.Provider',
            'Day.of.Arrival', 'Lab.Status', 'Rad.Status', 'Admitting.Provider'
            'Distance.To.Sick.Kids', 'Distance.To.Walkin', 'Distance.To.Hospital']
EPIC =  EPIC.drop(colRem, axis = 1)          


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


# Assign CC and FSA back
colNames[5] = 'CC'
colNames[17] = 'FSA'
# Re-assign col names
EPIC.columns = colNames

# Change 'Disch.Date.Time' and 'Roomed' to categorical
EPIC['Disch.Date.Time'] = [str(i) for i in EPIC['Disch.Date.Time']]
EPIC['Roomed'] = [str(i) for i in EPIC['Roomed']]

# Show data types. Select categorical and numerical features
print( list(set(EPIC.dtypes.tolist())) )
numCols = list(EPIC.select_dtypes(include = ['float64', 'int64']).columns)
catCols = [col for col in EPIC.columns if col not in numCols]
# Remove response variable
catCols.remove('Primary.Dx')

print('Categorical features:\n', catCols)
print('Numerical features:\n', numCols)


# ----------------------------------------------------
# Percentage of missing data
missingPer = EPIC.isna().sum()
missingPer = missingPer / len(EPIC)
_ = missingPer.plot(kind = 'barh')


# ----------------------------------------------------
'''
# Write summary of each feature into ./colSum.txt
file = open('colSum.txt', 'w')
# table = EPIC['Gender'].describe()
for col in colNames:
    _ = file.write('\n\n')
    _ = file.write(col)
    _ = file.write('\n')
    _ = file.write(EPIC[col].describe().to_string())

file.close()

'''
# ----------------------------------------------------
# Check if Primary.Dx contains Sepsis or related classes
ifSepsis = EPIC['Primary.Dx'].str.contains('Sepsis')
print('Number of sepsis or sepsis-related cases:', ifSepsis.sum())

# Convert into binary class
# Note: patients only healthy w.r.t. Sepsis
EPIC['Primary.Dx'][-ifSepsis] = 0
EPIC['Primary.Dx'][ifSepsis] = 1


# ----------------------------------------------------
# Data imputation
# Separate input features and target
y = EPIC['Primary.Dx']
X = EPIC.drop('Primary.Dx', axis = 1)
XCat = X.drop(numCols, axis = 1)
XNum = X.drop(catCols, axis = 1)

# Find all features with missing values
colWithMissing = list(missingPer.loc[missingPer > 0].index)
colWithMissing = [col for col in colWithMissing if col not in colRem]

# First try simple imputation
meanImp = sk.impute.SimpleImputer(strategy = 'mean')
freqImp = sk.impute.SimpleImputer(strategy = 'most_frequent')

# Impute categorical features with the most frequent class
freqImp.fit(XCat)
XCatImp = freqImp.transform(XCat)

# Impute numerical features with mean
meanImp.fit(XNum)
XNumImp = meanImp.transform(XNum)

EPIC[catCols] = XCatImp
EPIC[numCols] = XNumImp

print( 'Total no. of missing values after imputation:', 
        EPIC.isna().values.sum() + EPIC.isna().values.sum() )

# ----------------------------------------------------
# One-hot encode the categorical variables
EPIC_enc = EPIC.copy()
EPIC_enc = pd.get_dummies(EPIC_enc, columns = catCols)

# Encode the response as binary
EPIC_enc['Primary.Dx'] = EPIC_enc['Primary.Dx'].astype('int')

# EPIC = EPIC_enc
# EPIC = EPIC.drop('Primary.Dx')

# # Final dataset
# EPIC = pd.concat([EPIC, EPIC_enc], axis = 1)


# ----------------------------------------------------
# Logistic regression
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)

# Setting up testing and training sets
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27)
# Fit logistic regression
lr = sk.linear_model.LogisticRegression(solver = 'liblinear', 
                                        max_iter = 1000).fit(XTrain, yTrain)

lrPred = lr.predict(XTest)

sk.metrics.f1_score(yTest, lrPred)
sk.metrics.recall_score(yTest, lrPred)

# Random forest
rfc = sk.ensemble.RandomForestClassifier(n_estimators=10).fit(XTrain, yTrain)
# predict on test set
rfcPred = rfc.predict(XTest)

sk.metrics.recall_score(yTest, rfcPred)


# ----------------------------------------------------
# Oversampling

# concatenate our training data back together
EPIC_train = pd.concat([XTrain, yTrain], axis=1)

# separate minority and majority classes
isSepsis = EPIC_train[EPIC_train['Primary.Dx'] == 1]
notSepsis = EPIC_train[EPIC_train['Primary.Dx'] == 0]

# upsample minority
sepsisUpSampled = sk.utils.resample(isSepsis,
                                    replace = True, 
                                    n_samples = len(notSepsis), 
                                    random_state = 27) 

# combine majority and upsampled minority
upSampled = pd.concat([notSepsis, sepsisUpSampled])

# check new class counts
upSampled['Primary.Dx'].value_counts()

# Separate response and covariates
y = upSampled['Primary.Dx']
X = upSampled.drop('Primary.Dx', axis = 1)
# Setting up testing and training sets
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27)

# Fit logistic regression
lr = sk.linear_model.LogisticRegression(solver = 'liblinear', 
                                        max_iter = 1000).fit(XTrain, yTrain)

lrPred = lr.predict(XTest)

print('Logistic regression:')
sk.metrics.f1_score(yTest, lrPred)
sk.metrics.recall_score(yTest, lrPred)



# Random forest
rfc = sk.ensemble.RandomForestClassifier(n_estimators = 2).fit(XTrain, yTrain)
# predict on test set
rfcPred = rfc.predict(XTest)

print('Random forest with 2 estimators:')
sk.metrics.f1_score(yTest, rfcPred)
sk.metrics.recall_score(yTest, rfcPred)


# ----------------------------------------------------
# SMOTE

# Separate input features and target
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
# setting up testing and training sets
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27)

sm = SMOTE(random_state = 27, sampling_strategy = 0.4)
XTrain, yTrain = sm.fit_sample(XTrain, yTrain)
 
 

# Fit logistic regression
lr = sk.linear_model.LogisticRegression(solver = 'liblinear', 
                                        max_iter = 1000).fit(XTrain, yTrain)

lrPred = lr.predict(XTest)

print('Logistic regression:')
sk.metrics.precision_score(yTest, lrPred)
sk.metrics.f1_score(yTest, lrPred)
sk.metrics.recall_score(yTest, lrPred)


# Random forest
rfc = sk.ensemble.RandomForestClassifier(n_estimators = 2, max_depth = 10).fit(XTrain, yTrain)
# predict on test set
rfcPred = rfc.predict(XTest)

print('Random forest with 2 estimators:')
sk.metrics.precision_score(yTest, rfcPred)
sk.metrics.f1_score(yTest, rfcPred)
sk.metrics.recall_score(yTest, rfcPred)




