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
# Path on ssh
# path = '/hpf/largeprojects/agoldenb/lauren/ED/data/EPIC_DATA/ED_EPIC_DATA/ED_DATA_EPIC_OCTOBER18.csv'
path = '/home/xingliu/Documents/ED/data/EPIC_DATA/EPIC.csv'
# path = '/home/xingliu/Documents/EPIC_DATA/ED_EPIC_DATA/ED_DATA_EPIC_OCTOBER18.csv'
EPIC = pd.read_csv(path, encoding = 'ISO-8859-1')

# Get current wd (for saving figures)
savePath = '/home/xingliu/Documents/code/figures_old'


# ----------------------------------------------------
# Overview of the dataset (6298 x 51)
print('Dimension of data:', EPIC.shape)
EPIC.info()

# Show data types
print( list(set(EPIC.dtypes.tolist())) )
numCols = EPIC.select_dtypes(include = ['float64', 'int64']).columns
catCols = [col for col in EPIC.columns if col not in numCols]

# ----------------------------------------------------
# Interested in whether Diagnoses = Sepsis or not.
# Remove those with Diagnoses = NA
# n = len(EPIC)
# EPIC = EPIC.dropna(subset = ['Diagnoses'])
# print('Cases removed =', n - len(EPIC))

# Convert into binary classification
ifSepsis = EPIC['Primary.Dx'].str.contains('Sepsis')
EPIC['Primary.Dx'][-ifSepsis] = 0
print('Number of sepsis or sepsis-related', ifSepsis.sum())

datSepOZ = EPIC
datSepOZ['Diagnoses'][ifSepsis] = 1

# ----------------------------------------------------
# Percentage of missing data
missingPer = EPIC.isna().sum()
missingPer = missingPer / len(EPIC)
missingPer.plot(kind = 'barh')

# Some entries have no date written; used the date of the previous entry instead
monthVec = []
dateTime = EPIC['Disch.Date.Time']
for i in range(len(dateTime)):
    month = dateTime[i]
    
    if month != ' ':
        monthVec.append(month.split('/')[1])
    else:
        monthVec.append(monthVec[i - 1])

monthVec = pd.Series(monthVec)
EPIC['Month'] = monthVec

# Group data by month
keys = monthVec.unique()
keys.sort()
dataDic = {}
for i in keys:
    dataDic[i] = EPIC[EPIC['Month'] == i]

# Show we have sorted all data
print( 'No. of sorted data {}. No. of original data {}'.format(
        sum([ len(data) for data in dataDic.values() ]), len(EPIC)) )

# Store missing values by month to a dataframe
missingData = pd.DataFrame(missingPer)
missingData.columns = ['Average']
for i in range(12):
    # Retrieve data for this month
    dataMonth = dataDic[keys[i]]
    
    # Compute percentage of issing values
    missingData[keys[i]] = dataMonth.isna().sum() / len(dataMonth)

# Add extra column indicating month
missingData = missingData.transpose()
missingData['Month'] = missingData.index

# Plot percentages of missing vals by month
'''
fig1, ax1 = plt.subplots(6, 5, constrained_layout = True)
for i in range(missingData.shape[1] - 1):
    colName = missingData.columns[i]
    
    # Shown in two plots
    if i < 30:
        plt.subplot(6, 5, i + 1)
    elif i == 30:
        # Initialize new plot
        plt.subplots_adjust(left = 0.125, bottom = 0.125, hspace= 0.8)
        fig2, ax2 = plt.subplots(5, 5, constrained_layout = True)
        plt.subplot(5, 5, i - 29)
    else:
        plt.subplot(5, 5, i - 29)
    
    _ = plt.bar(missingData['Month'][1::], missingData[colName][1::])
    plt.axhline(missingData.iloc[:, i][0], color = 'k', linestyle='dotted')
    plt.title(colName, fontdict= {'fontsize':8})
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 7)


fig1.subplots_adjust(left = 0.125, bottom = 0.125, hspace= 0.8)
plt.show()
'''

# ----------------------------------------------------
# Correlation between missing values
# plt.matshow(missingData.corr())
_ = sns.heatmap(missingData.corr(), 
                cmap='viridis', linewidths=0.1,
                square=True, xticklabels = missingData.columns[:-1], 
                yticklabels = missingData.columns[:-1])
sns.set(font_scale = 0.6)
plt.subplots_adjust(top = 0.998, bottom = 0.24)
plt.show()

plt.xticks(range(missingData.shape[1]), missingData.columns, fontsize=6, rotation=90)
plt.yticks(range(missingData.shape[1]), missingData.columns, fontsize=8)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.subplots_adjust(left = 0.125)
# plt.savefig('../figures/corrForMissingVals.png')


# ----------------------------------------------------
# Pull out data with missing values in Pref.Language
prefLanData = EPIC.loc[EPIC['Pref.Language'].isna()]
prefLanData.to_csv('prefLanData.csv')

lastWeight = prefLanData['Last.Weight']
# Missing values denoted as None
lastWeightNum = [float( re.search('[0-9.]+', weight).group() )
                for weight in lastWeight if not weight == 'None']

# Repeat for data with Pre.Language
prefLanData2 = EPIC.loc[EPIC['Pref.Language'].notna()]
lastWeight2 = prefLanData2['Last.Weight']
lastWeightNum2 = [float( re.search('[0-9.]+', weight).group() )
                for weight in lastWeight2 if not weight == 'None']


# Plot distribution of weights with missing Pref.Language
ax4 = sns.distplot(lastWeightNum, label = 'With missing vals')
_ = sns.distplot(lastWeightNum2, label = 'Without missing vals')
ax4.set_title('Distributions for weights with/without \n missing vals in Pref.Language')
ax4.set(xlabel = 'weights in kg', ylabel = 'density')
_ = plt.legend()
plt.show()


# qq plot
def qqplot(data1, data2, xlabel = None, ylabel = None, title = None):
    q = np.linspace(0, 100, 101)
    quantile1 = np.percentile(data1, q)
    quantile2 = np.percentile(data2, q)
    q = np.linspace( np.min( ( quantile1.min(), quantile2.min() ) ),
                    np.min( ( quantile1.max(), quantile2.max() ) ) )
    _, ax5 = plt.subplots()
    _ = ax5.scatter(quantile1, quantile2)
    _ = plt.plot(q, q, color = 'grey')
    ax5.set(xlabel = xlabel, 
            ylabel = ylabel)
    ax5.set_title(title)
    plt.show()

qqplot(lastWeightNum, lastWeightNum2,
        xlabel = 'With missing values',
        ylabel = 'Without missing value',
        title = 'QQ plot for weights with and without \n missing values in Pref.Language')

# KS test
print("p-value of the KS test:", 
    stats.ks_2samp(lastWeightNum, lastWeightNum2)[1])


# ----------------------------------------------------
def ageConverter(data = None):
    '''
    Change Age.at.Visit to number of years (float)
    Input:  EPIC dataset as pd.DataFrame or pd.Series 
    Output: Same dataset with Age.at.Visit overwritten by number of years (float)
    
    Debug help: This function may encounter TypeError if applied a second time to the 
                same dataset.
    '''
    ageList = list(data['Age.at.Visit'])
    ageArr = np.zeros(len(ageList))
    units = {'wk':52, 'm':12, 'y':1}
    for i in range(len(ageList)):
        age = ageList[i]
        if 'days' in age:
            ageNum = float( re.search('[0-9]', age).group() ) / 365
        elif any(unit in age for unit in units.keys()): 
            ageNum = float( re.search('[0-9]+', age).group() )
            unit = re.search('[a-z]+', age).group()
            # Convert to years
            divisor = units.get(unit)
            ageNum = ageNum / divisor 
        else:
            ageNum = float(age)
        ageArr[i] = ageNum
    data['Age.at.Visit'] = ageArr
    return(data)


# Replace the original format of age
prefLanData = ageConverter(prefLanData)
prefLanData2 = ageConverter(prefLanData2)

sns.set(font_scale = 1.2)
ax5 = sns.distplot(prefLanData['Age.at.Visit'], label = 'with missing vals', hist_kws={'edgecolor':'black'})
_ = sns.distplot(prefLanData2['Age.at.Visit'], label = 'without missing vals')
_ = ax5.set_title('Distributions for ages with/without \n missing vals in Pref.Language')
_ = ax5.set(xlabel = 'ages in year', ylabel = 'density')
_ = plt.legend()
plt.show()

# QQ plot
ageArr = prefLanData['Age.at.Visit']
ageArr2 = prefLanData2['Age.at.Visit']
qqplot(prefLanData.loc[ageArr < 100, 'Age.at.Visit'], 
        prefLanData2['Age.at.Visit'],
        xlabel = 'With missing values',
        ylabel = 'Without missing value',
        title = 'QQ plot for ages with and without \n missing values in Pref.Language')


# Number of people with/without missing values in Pref.Language by age
EPIC = ageConverter(EPIC)
EPIC.loc[EPIC['Pref.Language'].notna(), 'Pref.Language'] = 1
EPIC.loc[EPIC['Pref.Language'].isna(), 'Pref.Language'] = 0

EPIC.loc[EPIC['Pref.Language'] == 0, 'Pref.Language'] = '0'
EPIC.loc[EPIC['Pref.Language'] == 1, 'Pref.Language'] = '1'

missingPropAge = np.zeros(6)
totalNum = np.zeros(6, dtype = int)
for i in range(6):
    ageData = EPIC.loc[ (EPIC['Age.at.Visit'] <= 3 * (i + 1)) & (EPIC['Age.at.Visit'] > 3 * i)]
    divisor = len(ageData)
    totalNum[i] = divisor
    if divisor != 0:
        missingPropAge[i] = 1 - ageData['Pref.Language'].sum() / divisor 
    else:
        missingPropAge[i] = 0

xticksPos = np.linspace(1.5, 16.5, 6)
xticksList = []
ax6 = plt.bar(xticksPos, missingPropAge, width = 3)
for i in range( len(missingPropAge) ):
    val = missingPropAge[i]
    _ = plt.text(3 * i + 1, val, str(round(100 * val, 1)) + '%', color='black', fontsize = 11)
    xticksList.append( str(int(xticksPos[i] - 1.5)) + '-' + str(int(xticksPos[i] + 1.5)) )

xticksList[-1] = '> 15'
_ = plt.title('Percentage of missing values in Pref.Language \n by age group')
_ = plt.xlabel('ages groups')
_ = plt.ylabel('proporation')
_ = plt.xticks(xticksPos, xticksList, fontsize = 11)
plt.show()



# for p in ax5.patches:
#         percentage = '{:.1f}%'.format(100 * p.get_width() / len(EPIC))
#         x = p.get_x() + p.get_width() + 0.02
#         y = p.get_y() + p.get_height()/2
#         ax5.annotate(percentage, (x, y))


# desc1 = prefLanData['Age.at.Visit'].describe()
# desc2 = prefLanData2['Age.at.Visit'].describe()

# # Create a subplot without frame
# plot = plt.subplot(111, frame_on=False)

# # Remove axis
# plot.xaxis.set_visible(False) 
# plot.yaxis.set_visible(False) 

# # Create the table plot and position it in the upper left corner
# pd.plotting.table(plot, desc1,loc='upper right')
# pd.plotting.table(plot, desc2,loc='upper left')



# ----------------------------------------------------
# Find all cols with NA larger than or equal to 5%
rmCols = [col for col in EPIC.columns if EPIC[col].count().sum() / len(EPIC) < 0.95]
# Add ID into the list(october only)
keepCols = [col for col in EPIC.columns if col not in rmCols]
print('Removed {:d} columns:\n'.format(len(rmCols)), '\n'.join(rmCols), end = '\n')
catCols = EPIC.select_dtypes(include = ['object']).columns


# Suspecious id columns:
# Encounter Number; CSN; Registration Number; MRN


# ----------------------------------------------------
# Investigate the suspected id cols in detail
# Encounter Number
EPIC['Encounter Number'].value_counts()
print('Conclusion: most of the values in this entry is 2000 and this is clearly not a useful feature. \n')

# CSN
print( 'Number of instances in \n 1.Registration Number: \
{:d}, 2.dataset: {:d}'.format( len( EPIC['CSN'] ) , len(EPIC) ))
print('The number of distinct instances equal to the number of total cases. Hence this is a unique number \
for every visit \n')

# Registration Number
print( 'Number of instances in \n 1.Registration Number: \
{:d}, 2.dataset: {:d}'.format( len( EPIC['Registration Number'] ) , len(EPIC) ))
print('The number of distinct instances equal to the number of total cases. Hence this is a unique number \
for every visit \n')

# MRN
EPIC['MRN'].value_counts()
print('Some values appear multiple times. Hence this is a unique number for every patient. \n')

# Remove these id features for now
idCols = ['Encounter Number', 'CSN', 'Registration Number', 'MRN']
rmCols.append(idCols)


# ----------------------------------------------------
# Correlation between some randomly selected features
pd.crosstab(EPIC.Acuity, EPIC.Diagnoses)
pd.crosstab(EPIC.CC, EPIC.Diagnoses)


# Extract only the numerical temperature
tempList = list( EPIC['Temp'].dropna() )
# Extract all cases that begin with numbers
tempList = [re.findall('^[0-9.]+', temp)[0] for temp in tempList if re.findall('^[0-9.]+', temp) ]
tempList = [float(temp) for temp in tempList]


for i in range(len(tempList)):
    print(i)
    # float(re.findall('[0-9\.]+', tempList[i])[0])
    tempList = [float(re.findall('[0-9.]+', temp)[0]) for temp in tempList if re.findall('[0-9.]+', temp) ]

# plt.figure(figsize = (10, 6))
# ax = sns.boxplot(x='Diagnoses', y='Temp', data=EPIC)
# plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
# plt.xticks(rotation=45)

ind = 0
for i in tempList:
    print(ind, i)
    print(float(i))
    ind += 1


# Run the following code without removing any entries
tempInd, tempArray = np.zeros(len(EPIC)), np.zeros(len(EPIC))
tempNotNull = EPIC['Temp'].notnull().values
for i in range( len(tempInd) ):
    temp = EPIC['Temp'].values[i]
    
    if tempNotNull[i] and not re.findall('[0-9.]+', temp):
        tempInd[i] = 1

# NEED TO FIGURE OUT HOW TO SELECT ROWS
datTemp = EPIC.loc[tempInd, :]
cPalette = ['tab:blue', 'tab:orange']
quantitativeSummarized(dataframe= datTemp, y = 'Temp', x = 'Diagnoses', 
                        palette=cPalette, verbose=False, swarm=True)


plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='Diagnoses', y='Temp', data=datTemp)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)


# ----------------------------------------------------
# Numerical features
# Function for analysis
def quantitativeSummarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):
    '''
    Helper function that gives a quick summary of quantattive data
    
    Input:
            dataframe: pandas dataframe
            x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
            y: str. vertical axis to plot the quantitative data
            hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
            palette: array-like. Colour of the plot
            swarm: if swarm is set to True, a swarm plot would be overlayed
    Output: 
    =======
    Quick Stats of the data and also the box plot of the distribution
    '''
    series = dataframe[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())
    sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)
    
    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)
    plt.show()

EPIC[numCols].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()


# ----------------------------------------------------
# Categorical features

# Function for analysis. NEED TO CREATE A SEPARATE FILE 
def categoricalSummarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot
    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())
    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
    plt.show()

# Show the sepsis cases only
datSep = EPIC[EPIC['Diagnoses'] != 0]
categoricalSummarized(datSep, y = 'Diagnoses')

# Print out types of sepsis
sepList = EPIC['Diagnoses'].unique().tolist()
print('Types of Sepsis:')
for item in sepList:
    print(item)

# Show how sepsis is correlated with other covariates
# Correlation with primary dx
cPalette = ['tab:blue', 'tab:orange']
categoricalSummarized(datSepOZ, y = 'Primary.Dx', hue = 'Diagnoses', palette = cPalette)





