# -------------------------------------------------------
# Path and savePath need to be configured before running
# -------------------------------------------------------
from ED_support_module import *  


# Load file
# Path of file
path = '/home/xingliu/Documents/ED/data/EPIC_DATA/preprocessed_EPIC.csv'
EPIC = pd.read_csv(path, encoding = 'ISO-8859-1')

notes = ['Notes', 'Provider.Notes', 'Triage.Notes']
EPIC_CUI = EPIC[notes]
# EPIC = EPIC.drop(notes, axis = 1)

# Set current wd (for saving figures)
savePath = '/home/xingliu/Documents/code/figures'


# ----------------------------------------------------
# Convert three cols of notes to list
for col in notes:
    noteLst = pd.Series( map( lambda note: note[2:-2].split('\', \''), EPIC_CUI[col] ) )
    EPIC[col] = noteLst


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
EPIC['Disch.Date.Time'] = EPIC['Disch.Date.Time'].astype('object')
EPIC['Roomed'] = EPIC['Roomed'].astype('object')

# Change 'Will.Return' to binary
EPIC['Will.Return'] = EPIC['Will.Return'].astype('object')


# ----------------------------------------------------
# Overview of the dataset (6298 x 51)
print('Dimension of data:', EPIC.shape)

# Discard the following features in modelling
colRem = ['Care.Area', 'First.ED.Provider', 'Last.ED.Provider', 'ED.Longest.Attending.ED.Provider',
            'Day.of.Arrival', 'Arrival.Month', 'FSA', 'Discharge.Admit.Time', 'Name.Of.Walkin', 'Name.Of.Hospital', 
            'Admitting.Provider', 'Disch.Date.Time', 
            'Distance.To.Sick.Kids', 'Distance.To.Walkin', 'Distance.To.Hospital', 'Systolic',
            'Size.Of.Treatment.Team', 'Arrival.to.Room',
            'Roomed', 'Same.First.And.Last']
# (Some) features obtained after triage
colAT = ['Lab.Status', 'Rad.Status', 'ED.PIA.Threshold', 'Same.First.And.Last', 'Dispo', 'Size.Of.Treatment.Team',
         'Number.Of.Prescriptions', 'Length.Of.Stay.In.Minutes', 'Arrival.to.Room']
# colRem = colRem + colAT
EPIC =  EPIC.drop(colRem, axis = 1)          


# ----------------------------------------------------
## Previous choice that produced good logistirc regression results:
## only changed Pref.Language and CC
# Pref.Language: Keep top 4 languages + others
topLangs = EPIC['Pref.Language'].value_counts().index[:4]
ifTopLangs = [not language in topLangs for language in EPIC['Pref.Language'].values]
EPIC['Pref.Language'].loc[ ifTopLangs ] = 'Other'

# CC: Keep top 19 + others
topCC = EPIC['CC'].value_counts().index[:19]
ifTopCC = [not complaint in topCC for complaint in EPIC['CC'].values]
EPIC['CC'].loc[ ifTopCC ] = 'Other'

# Arrival method: combine 'Unknown' and 'Other' and keep top 9 + others
EPIC.loc[EPIC['Arrival.Method'] == 'Unknown', 'Arrival.Method'] = 'Other'
topMethods = EPIC['Arrival.Method'].value_counts().index[:9]
ifTopMethods = [not method in topMethods for method in EPIC['Arrival.Method'].values]
EPIC['Arrival.Method'].loc[ ifTopMethods ] = 'Other'

# # log transform arrival to room. -inf in the transformed feature means 0 waiting time
# waitingTime = np.log(EPIC['Arrival.to.Room'] + 1)
# waitingTime[waitingTime == -np.inf] = 0


# ----------------------------------------------------
# Show data types. Select categorical and numerical features
print( list(set(EPIC.dtypes.tolist())) )
numCols = list(EPIC.select_dtypes(include = ['float64', 'int64']).columns)
catCols = [col for col in EPIC.columns if (col not in numCols) and (col not in notes)]
# Remove response variable
catCols.remove('Primary.Dx')

# print('Categorical features:\n', catCols)
# print('Numerical features:\n', numCols)


# ----------------------------------------------------
# Check if Primary.Dx contains Sepsis or related classes
ifSepsis = EPIC['Primary.Dx'].str.contains('epsis')
# print('Number of sepsis or sepsis-related cases:', ifSepsis.sum())

# Convert into binary class
# Note: patients only healthy w.r.t. Sepsis
EPIC['Primary.Dx'][-ifSepsis] = 0
EPIC['Primary.Dx'][ifSepsis] = 1
EPIC['Primary.Dx'] = EPIC['Primary.Dx'].astype('int')


# ----------------------------------------------------
# Abnormal cases. Suspected to be typos

# Old patients. All non-Sepsis 
cond1 = EPIC['Age.at.Visit'] > 40

# Temperature > 50 or < 30
cond2 = (EPIC['Temp'] > 50) | (EPIC['Temp'] < 30)

# Blood pressure > 240
cond3 = (EPIC['Diastolic'] > 240)

# Resp > 300
cond4 = EPIC['Resp'] > 300

# Pulse > 300
cond5 = EPIC['Pulse'] > 300

# Remove these outlisers
cond = cond1 | cond2 | cond3 | cond4 | cond5
sepRmNum = EPIC.loc[cond]['Primary.Dx'].sum()
EPIC = EPIC.loc[~cond]
EPIC_CUI = EPIC_CUI.loc[~cond]

print( 'Removed {} obvious outliers from the dataset'.format( cond.sum() ) )
print( '{} of these are Sepsis or related cases'.format(sepRmNum) )


# ----------------------------------------------------
# Percentage of missing data
missingPer = EPIC.isna().sum()
missingPer = missingPer / len(EPIC)
# _ = missingPer.plot(kind = 'barh')


# ----------------------------------------------------
'''
# Heatmap
corrMat = EPIC.corr()
_ = sns.heatmap(corrMat, cmap = 'YlGnBu', square = True)
# _ = plt.xticks(rotation = 30)
plt.subplots_adjust(left = 0.125, top = 0.95)
plt.xticks(rotation = 45, fontsize = 6)
plt.show()
'''

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

'''
# Pairplots
sns.pairplot(EPIC[numCols[:4]])
sns.pairplot(EPIC[numCols[4:8]])
sns.pairplot(EPIC[numCols[8:12]])
sns.pairplot(EPIC[numCols[12:16]])

# Countplots (for categorical variables):
for j in range(3):
    for i in range(6):
        name = catCols[i + 6* j]
        _ = plt.subplot(2, 3, i + 1)
        _ = sns.countplot(y = name, data = EPIC, 
                        order = EPIC[name].value_counts().index)
    plt.subplots_adjust(wspace = 1.2)
    plt.show()

for i in range(4):
    name = catCols[i + 18]
    _ = plt.subplot(2, 2, i + 1)
    _ = sns.countplot(y = name, data = EPIC, 
                    order = EPIC[name].value_counts().index)

plt.subplots_adjust(wspace = 0.6)
plt.show()
'''


# ----------------------------------------------------
# Data imputation
# Separate input features and target
y = EPIC['Primary.Dx']
X = EPIC.drop('Primary.Dx', axis = 1)
X = X.drop(notes, axis = 1)
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

print( 'Total no. of missing values after imputation:', EPIC.isna().values.sum() )


# ----------------------------------------------------
# Separate 3 notes columns from EPIC
EPIC_CUI = EPIC[notes]
EPIC = EPIC.drop(notes, axis = 1)


# ----------------------------------------------------
# One-hot encode the categorical variables
EPIC_enc = EPIC.copy()
EPIC_enc = pd.get_dummies(EPIC_enc, columns = catCols, drop_first = True)

# Encode the response as binary
EPIC_enc['Primary.Dx'] = EPIC_enc['Primary.Dx'].astype('int')


# ----------------------------------------------------
print('\n----------------------------------------------------')
print('\nSourcing completed \n')
