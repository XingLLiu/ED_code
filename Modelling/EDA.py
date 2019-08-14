# -------------------------------------------------------
# Path and savePath need to be configured before running
# -------------------------------------------------------
from ED_support_module import *  


# Load file
# Path of file
path = '/home/xingliu/Documents/ED/data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv'
EPIC = pd.read_csv(path, encoding = 'ISO-8859-1')

# Set current wd (for saving figures)
savePath = '/home/xingliu/Documents/code/figures'

# ----------------------------------------------------
# Separate 3 notes columns from EPIC
notes = ['Notes', 'Provider.Notes', 'Triage.Notes']
EPIC_CUI = EPIC[notes]
EPIC = EPIC.drop(notes, axis = 1)

# Separate MRN and arrival date
EPIC_arrival = EPIC[['MRN', 'Arrived']]
EPIC = EPIC.drop(['MRN', 'Arrived'], axis = 1)

# Separate diagnoses columns
diagnosis = EPIC['Diagnosis']
diagnoses = EPIC['Diagnoses']
EPIC = EPIC.drop(['Diagnosis', 'Diagnoses'], axis = 1)


# ----------------------------------------------------
# Convert three cols of notes to list
for col in notes:
    noteLst = pd.Series( map( lambda note: note[2:-2].split('\', \''), EPIC_CUI[col] ) )
    EPIC_CUI[col] = noteLst


# ----------------------------------------------------
# Unify format of column names
colNames = list(EPIC.columns)
# Special cases
indexVec = np.linspace(0, len(colNames) - 1, len(colNames), dtype = 'int')
ifCC = int(indexVec[EPIC.columns == 'CC'])
ifFSA = int(indexVec[EPIC.columns == 'FSA'])

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


# Assign special cases back
colNames[ifCC] = 'CC'
colNames[ifFSA] = 'FSA'

# Re-assign col names
EPIC.columns = colNames

# Print error warning if there is empty colname remaining
for name in colNames:
    if len(colNames) == 0:
        print('Empty column name warning! Column name assignment may be wrong!')


# Change 'Disch.Date.Time' and 'Roomed' to categorical
EPIC['Disch.Date.Time'] = EPIC['Disch.Date.Time'].astype('object')
EPIC['Roomed'] = EPIC['Roomed'].astype('object')

# Change 'Will.Return' to binary
EPIC['Will.Return'] = EPIC['Will.Return'].astype('object')


# ----------------------------------------------------
# Overview of the dataset 
print('Dimension of data:', EPIC.shape)

# Discard the following features in modelling
colRem = ['First.ED.Provider', 'Last.ED.Provider', 'ED.Longest.Attending.ED.Provider',
            'Day.of.Arrival', 'Arrival.Month', 'FSA', 'Name.Of.Walkin', 'Name.Of.Hospital',
            'Admitting.Provider', 'Disch.Date.Time', 'Discharge.Admit.Time',
            'Distance.To.Sick.Kids', 'Distance.To.Walkin', 'Distance.To.Hospital', 'Systolic',
            'Pulse']
# (Some) features obtained after triage
colAT = ['Lab.Status', 'Rad.Status', 'ED.PIA.Threshold', 'Same.First.And.Last', 'Dispo', 'Size.Of.Treatment.Team',
         'Number.Of.Prescriptions', 'Length.Of.Stay.In.Minutes', 'Arrival.to.Room', 'Roomed']
colRem = colRem + colAT
EPIC =  EPIC.drop(colRem, axis = 1)


# ----------------------------------------------------
## Previous choice that produced good logistirc regression results:
## only changed Pref.Language and CC
# Pref.Language: Keep top 4 languages + others
topLangs = EPIC['Pref.Language'].value_counts().index[:4]
ifTopLangs = [not language in topLangs for language in EPIC['Pref.Language'].values]
EPIC['Pref.Language'].loc[ ifTopLangs ] = 'Others'

# CC: Keep top 19 + others
topCC = EPIC['CC'].value_counts().index[:49]
ifTopCC = [not complaint in topCC for complaint in EPIC['CC'].values]
EPIC['CC'].loc[ ifTopCC ] = 'Others'

# Arrival method: combine 'Unknown' and 'Others' and keep top 9 + others
EPIC.loc[EPIC['Arrival.Method'] == 'Unknown', 'Arrival.Method'] = 'Others'
topMethods = EPIC['Arrival.Method'].value_counts().index[:14]
ifTopMethods = [not method in topMethods for method in EPIC['Arrival.Method'].values]
EPIC['Arrival.Method'].loc[ ifTopMethods ] = 'Others'

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


# ----------------------------------------------------
# Check if Primary.Dx contains Sepsis or related classes
ifSepsis1 = EPIC['Primary.Dx'].str.contains('epsis')
# Check if Diagnosis contains Sepsis or related classes
ifSepsis2 = diagnosis.str.contains('epsis')
# Check if Diagnoses contains Sepsis or related classes
ifSepsis3 = diagnoses.str.contains('epsis')
# Lable as sepsis if any of the above contains Sepsis
ifSepsis = ifSepsis1 | ifSepsis2 | ifSepsis3
# print('Number of sepsis or sepsis-related cases:', ifSepsis.sum())

# Convert into binary class
# Note: patients only healthy w.r.t. Sepsis
EPIC.loc[-ifSepsis, 'Primary.Dx'] = 0
EPIC.loc[ifSepsis, 'Primary.Dx'] = 1
EPIC['Primary.Dx'] = EPIC['Primary.Dx'].astype('int')


# ----------------------------------------------------
# Abnormal cases. Suspected to be typos

# Old patients. All non-Sepsis 
cond1 = EPIC['Age.at.Visit'] > 40

# Temperature > 50 or < 30
cond2 = (EPIC['Temp'] > 50) | (EPIC['Temp'] < 30)

# Blood pressure > 240
cond3 = (EPIC['Diastolic'] > 240)

# # Resp > 300
cond4 = EPIC['Resp'] > 300

# Pulse > 300
# cond5 = EPIC['Pulse'] > 300

# Remove these outlisers
# cond = cond1 | cond2 | cond3 | cond4 | cond5
cond = cond1 | cond2 | cond3 | cond4
sepRmNum = EPIC.loc[cond]['Primary.Dx'].sum()
EPIC = EPIC.loc[~cond]
EPIC_CUI = EPIC_CUI.loc[~cond]
EPIC_arrival = EPIC_arrival.loc[~cond]

print( 'Removed {} obvious outliers from the dataset'.format( cond.sum() ) )
print( '{} of these are Sepsis or related cases'.format(sepRmNum) )


# ----------------------------------------------------
# Percentage of missing data
missingPer = EPIC.isna().sum()
missingPer = missingPer / len(EPIC)
# _ = missingPer.plot(kind = 'barh')


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

print( 'Total no. of missing values after imputation:', EPIC.isna().values.sum() )


# ----------------------------------------------------
# One-hot encode the categorical variables
EPIC_enc = EPIC.copy()
EPIC_enc = pd.get_dummies(EPIC_enc, columns = catCols, drop_first = True)

# Encode the response as binary
EPIC_enc['Primary.Dx'] = EPIC_enc['Primary.Dx'].astype('int')

# ----------------------------------------------------
# EPIC with arrival date
# Fill in NAN with adjacent records
nullDate = EPIC_arrival.index[EPIC_arrival['Arrived'].isnull()].astype('int')
EPIC_arrival['Arrived'][nullDate] = EPIC_arrival.loc[nullDate - 1, 'Arrived']

# Append arrival date
EPIC_arrival = pd.concat([EPIC_enc, EPIC_arrival['Arrived'].astype('int')], axis = 1)


# ----------------------------------------------------
print('\n----------------------------------------------------')
print('\nSourcing completed \n')
