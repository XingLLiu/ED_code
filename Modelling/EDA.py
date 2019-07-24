# -------------------------------------------------------
# Path and savePath need to be configured before running
# -------------------------------------------------------
from ED_support_module import *  

plt.style.use('bmh')

# Load file
# Path of file
path = '/home/xingliu/Documents/ED/data/EPIC_DATA/preprocessed_EPIC.csv'
EPIC = pd.read_csv(path, encoding = 'ISO-8859-1')

# Set current wd (for saving figures)
savePath = '/home/xingliu/Documents/code/figures'


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
EPIC.info()

# Discard the following features in modelling
colRem = ['Care.Area', 'First.ED.Provider', 'Last.ED.Provider', 'ED.Longest.Attending.ED.Provider',
            'Day.of.Arrival', 'Arrival.Month', 'FSA', 'Name.Of.Walkin', 'Name.Of.Hospital', 'Lab.Status', 'Rad.Status', 
            'Admitting.Provider', 'Disch.Date.Time', 'Roomed',
            'Distance.To.Sick.Kids', 'Distance.To.Walkin', 'Distance.To.Hospital', 'Systolic']
# colRem = ['First.ED.Provider', 'Last.ED.Provider', 'ED.Longest.Attending.ED.Provider',
#             'Day.of.Arrival', 'Lab.Status', 'Rad.Status', 'Admitting.Provider'
#             'Distance.To.Sick.Kids', 'Distance.To.Walkin', 'Distance.To.Hospital']
EPIC =  EPIC.drop(colRem, axis = 1)          


# ----------------------------------------------------
## Previous choice that produced good logistirc regression results:
## only changed Pref.Language and CC
# Pref.Language: Keep top 4 languages + others
topLangs = EPIC['Pref.Language'].value_counts().index[:4]
ifTopLangs = [not language in topLangs for language in EPIC['Pref.Language'].values]
EPIC['Pref.Language'].loc[ ifTopLangs ] = 'Other'

# CC: Keep top 9 + others
topCC = EPIC['CC'].value_counts().index[:9]
ifTopCC = [not complaint in topCC for complaint in EPIC['CC'].values]
EPIC['CC'].loc[ ifTopCC ] = 'Other'

# Arrival method: combine 'Unknown' and 'Other' and keep top 4 + others
EPIC.loc[EPIC['Arrival.Method'] == 'Unknown', 'Arrival.Method'] = 'Other'
topMethods = EPIC['Arrival.Method'].value_counts().index[:4]
ifTopMethods = [not method in topMethods for method in EPIC['Arrival.Method'].values]
EPIC['Arrival.Method'].loc[ ifTopMethods ] = 'Other'

# # log transform arrival to room. -inf in the transformed feature means 0 waiting time
# waitingTime = np.log(EPIC['Arrival.to.Room'] + 1)
# waitingTime[waitingTime == -np.inf] = 0


# ----------------------------------------------------
# Show data types. Select categorical and numerical features
print( list(set(EPIC.dtypes.tolist())) )
numCols = list(EPIC.select_dtypes(include = ['float64', 'int64']).columns)
catCols = [col for col in EPIC.columns if col not in numCols]
# Remove response variable
catCols.remove('Primary.Dx')

print('Categorical features:\n', catCols)
print('Numerical features:\n', numCols)


# ----------------------------------------------------
# Check if Primary.Dx contains Sepsis or related classes
ifSepsis = EPIC['Primary.Dx'].str.contains('Sepsis')
print('Number of sepsis or sepsis-related cases:', ifSepsis.sum())

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

# Blood pressure > 200
cond3 = (EPIC['Diastolic'] > 200)

# Resp > 100
cond4 = EPIC['Resp'] > 100

# Pulse > 300
cond5 = EPIC['Pulse'] > 300

# Remove these outlisers
# cond = cond1 | cond2 | cond3 | cond4
cond = cond1 | cond2 | cond3 | cond4 | cond5
sepRmNum = EPIC.loc[cond]['Primary.Dx'].sum()
EPIC = EPIC.loc[~cond]

print( 'Removed {} obvious outliers from the dataset'.format( cond.sum() ) )
print('{} of these are Sepsis or related cases'.format(sepRmNum))


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


# KNN imputer




# ----------------------------------------------------
# One-hot encode the categorical variables
EPIC_enc = EPIC.copy()
EPIC_enc = pd.get_dummies(EPIC_enc, columns = catCols)

# Encode the response as binary
EPIC_enc['Primary.Dx'] = EPIC_enc['Primary.Dx'].astype('int')


# ----------------------------------------------------
print('\n----------------------------------------------------')
print('\n \nSourcing completed')


# # ----------------------------------------------------
# # Logistic regression
# y = EPIC_enc['Primary.Dx']
# X = EPIC_enc.drop('Primary.Dx', axis = 1)

# # Setting up testing and training sets
# XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27)
# # Fit logistic regression
# lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l1',
#                                         max_iter = 1500).fit(XTrain, yTrain)

# lrPred = lr.predict(XTest)

# roc_plot(yTest, lrPred)

# # Random forest
# rfc = sk.ensemble.RandomForestClassifier(n_estimators = 10, max_depth = 20).fit(XTrain, yTrain)
# # predict on test set
# rfcPred = rfc.predict(XTest)

# roc_plot(yTest, rfcPred)


# # ----------------------------------------------------
# # Oversampling

# # concatenate our training data back together
# EPIC_train = pd.concat([XTrain, yTrain], axis=1)

# # separate minority and majority classes
# isSepsis = EPIC_train[EPIC_train['Primary.Dx'] == 1]
# notSepsis = EPIC_train[EPIC_train['Primary.Dx'] == 0]

# # upsample minority
# sepsisUpSampled = sk.utils.resample(isSepsis,
#                                     replace = True, 
#                                     n_samples = len(notSepsis), 
#                                     random_state = 27) 

# # combine majority and upsampled minority
# upSampled = pd.concat([notSepsis, sepsisUpSampled])

# # check new class counts
# upSampled['Primary.Dx'].value_counts()

# # Separate response and covariates
# y = upSampled['Primary.Dx']
# X = upSampled.drop('Primary.Dx', axis = 1)
# # Setting up testing and training sets
# XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27)

# # Fit logistic regression
# lr = sk.linear_model.LogisticRegression(solver = 'liblinear', 
#                                         max_iter = 1000).fit(XTrain, yTrain)

# lrPred = lr.predict(XTest)

# print('Logistic regression:')
# sk.metrics.f1_score(yTest, lrPred)
# sk.metrics.recall_score(yTest, lrPred)



# # Random forest
# rfc = sk.ensemble.RandomForestClassifier(n_estimators = 2).fit(XTrain, yTrain)
# # predict on test set
# rfcPred = rfc.predict(XTest)

# print('Random forest with 2 estimators:')
# sk.metrics.f1_score(yTest, rfcPred)
# sk.metrics.recall_score(yTest, rfcPred)


# # ----------------------------------------------------
# # SMOTE

# # Separate input features and target
# y = EPIC_enc['Primary.Dx']
# X = EPIC_enc.drop('Primary.Dx', axis = 1)

# XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)

# sm = SMOTE(random_state = 27, sampling_strategy = 0.4)
# XTrain, yTrain = sm.fit_sample(XTrain, yTrain)

# # # Stratified sampling
# # XNormal, XIll = EPIC_enc.loc[y == 0], EPIC_enc.loc[y == 1]
# # yNormal, yIll = y.loc[y == 0], y.loc[y == 1]
# # # setting up testing and training sets
# # XTrainNormal, XTestNormal, yTrainNormal, yTestNormal = sk.model_selection.train_test_split(XNormal, yNormal, 
# #                                                                                         test_size=0.25, random_state=27)
# # XTrainIll, XTestIll, yTrainIll, yTestIll = sk.model_selection.train_test_split(XIll, yIll, 
# #                                                                             test_size=0.25, random_state=27)
# # XTrain = pd.concat([XTrainNormal, XTrainIll])
# # yTrain = pd.concat([yTrainNormal, yTrainIll])


# # ----------------------------------------------------
# # Logistic regression
# lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l1',
#                                         max_iter = 1000).fit(XTrain, yTrain)


# lrPred = lr.predict(XTest)
# roc_plot(yTest, lrPred)




# # Remove the zero coefficients
# ifZero = (lr.coef_ == 0).reshape(-1)
# notZero = (lr.coef_ != 0).reshape(-1)
# coeffs = [X.columns[i] for i in range(X.shape[1]) if not ifZero[i]]
# coeffsRm = [X.columns[i] for i in range(X.shape[1]) if ifZero[i]]

# # Refit 
# whichKeep = pd.Series( range( len( notZero ) ) )
# whichKeep = whichKeep.loc[notZero]
# XTrain, XTest = XTrain.iloc[:, whichKeep], XTest.iloc[:, whichKeep]
# lr2 = sk.linear_model.LogisticRegression(max_iter = 1000).fit(XTrain, yTrain)

# lrPred2 = lr2.predict(XTest)
# roc_plot(yTest, lrPred2)




# # Random forest
# # rfc = sk.ensemble.RandomForestClassifier(n_estimators = 1000, max_features = 8, max_depth = 15).fit(XTrain, yTrain)
# # rfc = sk.ensemble.RandomForestClassifier(n_estimators = 2000, max_depth = 12).fit(XTrain, yTrain)
# # rfc = sk.ensemble.RandomForestClassifier(n_estimators = 1000, max_depth = 12, max_features = 50).fit(XTrain, yTrain)
# rfc = sk.ensemble.RandomForestClassifier(n_estimators = 500, max_depth = 12, max_features = 50).fit(XTrain, yTrain)
# # predict on test set
# rfcPred = rfc.predict(XTest)
# print('Random forest:')
# roc_plot(yTest, rfcPred)


# # CART
# # cart = sk.tree.DecisionTreeClassifier(max_features = 40, class_weight = {1:200}, random_state = 27)
# cart = sk.tree.DecisionTreeClassifier(max_features = 50, class_weight = {1:150, 0:1}, random_state = 27)
# cart = cart.fit(XTrain, yTrain)
# cartPred = cart.predict(XTest)
# roc_plot(yTest, cartPred)


# # ----------------------------------------------------
# # Unsupervised methods

# # double_scatter('Pulse', 'Last.Weight', data = EPIC_enc)

# # Split into train and test set (without oversampling)
# y = EPIC_enc['Primary.Dx']
# X = EPIC_enc.drop('Primary.Dx', axis = 1)
# XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)

# # Split train set into normal abnormal instances
# XTrainNormal = XTrain[ yTrain == 0 ]
# XTrainOutliers = XTrain[ yTrain == 1 ]

# # One-class SVM
# outlierProp = len(XTrainOutliers) / len(XTrainNormal)
# algorithm = sk.svm.OneClassSVM(kernel ='rbf', nu = 0.98, gamma = 0.001)
# svmModel = algorithm.fit(XTrainNormal)

# svmPred = svmModel.predict(XTest) 
# svmPred[svmPred == -1] = 0       # Set labels to (0, 1) rather than (-1, 1)



# # Use the feature-reduced dataset
# outlierProp = len(XTrainOutliers) / len(XTrainNormal)
# algorithm = sk.svm.OneClassSVM(kernel ='rbf', nu = 0.005, gamma = 0.01)
# svmModel = algorithm.fit(XTrainNormal.iloc[:, whichKeep])

# svmPred = svmModel.predict(XTest.iloc[:, whichKeep]) 
# svmPred[svmPred == -1] = 0       # Set labels to (0, 1) rather than (-1, 1)



# # colors = np.array(['#377eb8', '#ff7f00'])
# # _ = plt.scatter(XTest['Age.at.Visit'], XTest['Last.Weight'], alpha = 0.7, c = colors[(svmPred + 1) // 2])
# # _ = plt.xlabel('Age')
# # _ = plt.ylabel('Weight')

# print('One-class SVM:')
# roc_plot(yTest, svmPred)


# # Isolation forest
# # training the model
# isof = sk.ensemble.IsolationForest(n_estimators = 500, max_samples = 1024, random_state = 27, contamination = 0.998)
# isof.fit(XTrainNormal)

# # predictions
# # yPredTrain = isof.predict(XTrain)
# # yPredTrain[yPredTrain == -1] = 0
# isofPredTest = isof.predict(XTest)
# isofPredTest[isofPredTest == -1] = 0

# # roc_plot(yTrain, yPredTrain)
# roc_plot(yTest, isofPredTest)



# # Extended isolation forest
# eisof = eif.iForest(XTrainNormal.values, 
#                      ntrees = 500, 
#                      sample_size = 2048, 
#                      ExtensionLevel = XTrainNormal.shape[1] - 1)

# # calculate anomaly scores
# anomalyScores = eisof.compute_paths(X_in = XTest.values)
# # sort the scores
# anomalyScoresSorted = np.argsort(anomalyScores)
# # retrieve indices of anomalous observations
# indicesWithPreds = anomalyScoresSorted[-int(np.ceil( 0.002 * XTest.shape[0] )):]   # 0.002 is the anomaly ratio
# # create predictions 
# eisofPred = yTest * 0
# eisofPred.iloc[indicesWithPreds] = 1

# roc_plot(yTest, eisofPred)


# # ----------------------------------------------------
# # DBSCAN
# # Scale the data
# y = EPIC_enc['Primary.Dx']
# X = EPIC_enc.drop('Primary.Dx', axis = 1)

# # Put Will.Return into list of categorical features
# robust = sk.preprocessing.RobustScaler()
# X[numCols] = robust.fit_transform(X[numCols])

# # Fit DBSCAN
# dbscan = sk.cluster.DBSCAN(eps = 3, min_samples = 4, metric = 'euclidean')
# dbscanModel = dbscan.fit(X)

# labels = dbscanModel.labels_
# coreSamples = np.zeros_like(labels, dtype = bool)
# coreSamples[dbscan.core_sample_indices_] = True
# nClusters = len(set(labels)) - (1 if -1 in labels else 0)
# print('Silhouette coefficient: %0.3f' %sk.metrics.silhouette_score(X, labels))



# _ = plt.scatter(X['Pulse'], X['Temp'], alpha = 0.2, c = y)
# _ = plt.scatter(X.loc[y == 1, 'Pulse'], X.loc[y == 1, 'Temp'], c = 'goldenrod', alpha = 0.8)
# _ = plt.scatter(X.loc[labels == -1, 'Pulse'], X.loc[labels == -1, 'Temp'], marker = '+', color = 'slategrey', alpha = 0.8) 
# _ = plt.xlabel('Pulse')
# _ = plt.ylabel('Temp')
# plt.show()




# # ----------------------------------------------------
# # Cross validation for logistic regression
# lrResults = {}
# for i in range(100):
#     XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, stratify = y)
#     sm = SMOTE(random_state = 27, sampling_strategy = 0.4)
#     XTrain, yTrain = sm.fit_sample(XTrain, yTrain)
#     lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l1',
#                                         max_iter = 1000).fit(XTrain, yTrain)
#     lrPred = lr.predict(XTest)
#     metrics = roc_plot(yTest, lrPred, plot = False)
#     lrResults[str(i)] = metrics
#     if i % 10 == 0:
#         print('Current iteration:', i)


# f1Vec = [scores[2] for scores in lrResults.values()]
# _ = sns.scatterplot(range(100), f1Vec)
# aucVec = [scores[3] for scores in lrResults.values()]
# _ = sns.scatterplot(range(100), aucVec)
# plt.show()


# # Tuning random forest
# nEstimators = [50, 100, 200, 500, 1000, 2000, 5000]
# maxDepth = [5, 10, 12, 15, 20, 50, None]
# rfResults = {}
# for nEst in nEstimators:
#     print('Training for nEstimator =', nEst)
#     for maxDep in maxDepth:
#         rfc = sk.ensemble.RandomForestClassifier(n_estimators = nEst, max_depth = maxDep, max_features = 50).fit(XTrain, yTrain)
#         # predict on test set
#         rfcPred = rfc.predict(XTest)
#         metrics = roc_plot(yTest, rfcPred,  plot = False)
#         parameters = str(nEst) + '; ' + str(maxDep)
#         rfResults[parameters] = metrics


# optimParams = {'nEst':0, 'maxDep':0}
# optimIndex = aucVec.index( max(aucVec) ) // len(maxDepth) % len(nEstimators)
# optimParams['nEst'] = nEstimators[ optimIndex ]
# optimParams['maxDep'] = maxDepth[ optimIndex ]

# f1Vec = [scores[2] for scores in rfResults.values()]
# _ = sns.scatterplot(range(len(rfResults)), f1Vec)
# aucVec = [scores[3] for scores in rfResults.values()]
# _ = sns.scatterplot(range(len(rfResults)), aucVec)
# _ = plt.xlabel('No. of estimators and max depth')
# _ = plt.ylabel('f1 score')
# plt.show()

# # 3D plot
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection = '3d')
# # x, y, z = np.meshgrid(nEstimators, maxDepth, aucVec)
# # _ = ax.scatter(x, y, z)
# # _ = plt.xlabel('No. of estimators')
# # _ = plt.ylabel('Max depth')
# # _ = plt.title('f1 score')
# # plt.show()






