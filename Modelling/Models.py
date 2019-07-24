# ----------------------------------------------------
from ED_support_module import *                      # Source required modules and functions
from EDA import EPIC, EPIC_enc, numCols, catCols     # Source datasets from EDA.py


# ----------------------------------------------------
# PCA on the numerical features
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)

# Transform the numerical features
XTrainNum, XTestNum = pd.DataFrame(XTrain)[numCols], pd.DataFrame(XTest)[numCols]
transformer = sk.preprocessing.RobustScaler().fit(XTrainNum)
XTrainNum = transformer.transform(XTrainNum)

# Choose PCA such that 0.95 of the variance is explained
pca = sk.decomposition.PCA(0.95).fit(XTrainNum)

# Transform the train and test sets to principle components
XTrainNum = pd.DataFrame(pca.transform(XTrainNum))
XTestNum = pd.DataFrame(pca.transform(XTestNum))
XTrainNum.index, XTestNum.index = XTrain.index, XTest.index

# Replace the numerical features with the principle components
XTrain = pd.concat([pd.DataFrame(XTrainNum), pd.DataFrame(XTrain.drop(numCols, axis = 1))], axis = 1, sort = False)
XTest = pd.concat([pd.DataFrame(XTestNum), pd.DataFrame(XTest.drop(numCols, axis = 1))], axis = 1, sort = False)


# ----------------------------------------------------
# Logistic regression without oversampling
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)

# Setting up testing and training sets
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27)
# Fit logistic regression
lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l1',
                                        max_iter = 1500).fit(XTrain, yTrain)

lrPred = lr.predict(XTest)

roc_plot(yTest, lrPred)

# Random forest
rfc = sk.ensemble.RandomForestClassifier(n_estimators = 10, max_depth = 20).fit(XTrain, yTrain)
# predict on test set
rfcPred = rfc.predict(XTest)

roc_plot(yTest, rfcPred)


# ----------------------------------------------------
# Logistic regression with oversampling
lr = sk.linear_model.LogisticRegression(solver = 'liblinear', 
                                        max_iter = 1000).fit(XTrain, yTrain)

lrPred = lr.predict(XTest)

print('Logistic regression:')
sk.metrics.f1_score(yTest, lrPred)
sk.metrics.recall_score(yTest, lrPred)



# Random forest with oversampling
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

XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)

sm = SMOTE(random_state = 27, sampling_strategy = 'auto') # 0.4
XTrain, yTrain = sm.fit_sample(XTrain, yTrain)


# SMOTE with PCA
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
colNames = X.columns
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)
sm = SMOTE(random_state = 27, sampling_strategy = 'auto')
XTrain, yTrain = sm.fit_sample(XTrain, yTrain)
XTrain, XTest = pd.DataFrame(XTrain), pd.DataFrame(XTest)
XTrain.columns, XTest.columns = colNames, colNames

XTrainNum, XTestNum = pd.DataFrame(XTrain)[numCols], pd.DataFrame(XTest)[numCols]
transformer = sk.preprocessing.RobustScaler().fit(XTrainNum)
XTrainNum = transformer.transform(XTrainNum)
pca = sk.decomposition.PCA(0.95).fit(XTrainNum)
XTrainNum = pd.DataFrame(pca.transform(XTrainNum))
XTestNum = pd.DataFrame(pca.transform(XTestNum))
XTrainNum.index, XTestNum.index = XTrain.index, XTest.index
XTrain = pd.concat([pd.DataFrame(XTrainNum), pd.DataFrame(XTrain.drop(numCols, axis = 1))], axis = 1, sort = False)
XTest = pd.concat([pd.DataFrame(XTestNum), pd.DataFrame(XTest.drop(numCols, axis = 1))], axis = 1, sort = False)


def numPCA(XTrain, XTest, numCols):
    '''
    Scale and transfer the numerical features of XTrain and XTest using PCA and the Euclidean metric.
    Input:  XTrain: train set without the response (pd.DataFrame)
            XTest: test set without the response (pd.DataFrame)
    Output: XTrain and XTest after PCA as a list
    '''
    XTrainNum, XTestNum = pd.DataFrame(XTrain)[numCols], pd.DataFrame(XTest)[numCols]
    transformer = sk.preprocessing.RobustScaler().fit(XTrainNum)
    XTrainNum = transformer.transform(XTrainNum)
    pca = sk.decomposition.PCA(0.95).fit(XTrainNum)
    XTrainNum = pd.DataFrame(pca.transform(XTrainNum))
    XTestNum = pd.DataFrame(pca.transform(XTestNum))
    XTrainNum.index, XTestNum.index = XTrain.index, XTest.index
    XTrain = pd.concat([pd.DataFrame(XTrainNum), pd.DataFrame(XTrain.drop(numCols, axis = 1))], axis = 1, sort = False)
    XTest = pd.concat([pd.DataFrame(XTestNum), pd.DataFrame(XTest.drop(numCols, axis = 1))], axis = 1, sort = False)
    return([XTrain, XTest])


# ----------------------------------------------------
# # SMOTENC for mixed data
# y = EPIC['Primary.Dx']
# X = EPIC.drop('Primary.Dx', axis = 1)
# XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)

# ifCat = [name in catCols for name in X.columns]
# sm = SMOTENC(random_state = 27, sampling_strategy = 0.4, categorical_features = ifCat)
# XTrain, yTrain = sm.fit_sample(XTrain, yTrain)

# # Stratified sampling
# XNormal, XIll = EPIC_enc.loc[y == 0], EPIC_enc.loc[y == 1]
# yNormal, yIll = y.loc[y == 0], y.loc[y == 1]
# # setting up testing and training sets
# XTrainNormal, XTestNormal, yTrainNormal, yTestNormal = sk.model_selection.train_test_split(XNormal, yNormal, 
#                                                                                         test_size=0.25, random_state=27)
# XTrainIll, XTestIll, yTrainIll, yTestIll = sk.model_selection.train_test_split(XIll, yIll, 
#                                                                             test_size=0.25, random_state=27)
# XTrain = pd.concat([XTrainNormal, XTrainIll])
# yTrain = pd.concat([yTrainNormal, yTrainIll])


# ----------------------------------------------------
# Logistic regression with SMOTE
lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l1',
                                        max_iter = 1000).fit(XTrain, yTrain)


lrPred = lr.predict(XTest)
roc_plot(yTest, lrPred)


# Remove the zero coefficients
ifZero = (lr.coef_ == 0).reshape(-1)
notZero = (lr.coef_ != 0).reshape(-1)
coeffs = [X.columns[i] for i in range(X.shape[1]) if not ifZero[i]]
coeffsRm = [X.columns[i] for i in range(X.shape[1]) if ifZero[i]]

# Refit 
whichKeep = pd.Series( range( len( notZero ) ) )
whichKeep = whichKeep.loc[notZero]
XTrain, XTest = XTrain.iloc[:, whichKeep], XTest.iloc[:, whichKeep]
lr2 = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l2', 
                                            max_iter = 1000).fit(XTrain, yTrain)

lrPred2 = lr2.predict(XTest)
roc_plot(yTest, lrPred2)




# Random forest
# rfc = sk.ensemble.RandomForestClassifier(n_estimators = 1000, max_features = 8, max_depth = 15).fit(XTrain, yTrain)
# rfc = sk.ensemble.RandomForestClassifier(n_estimators = 2000, max_depth = 12).fit(XTrain, yTrain)
# rfc = sk.ensemble.RandomForestClassifier(n_estimators = 1000, max_depth = 12, max_features = 50).fit(XTrain, yTrain)
# rfc = sk.ensemble.RandomForestClassifier(n_estimators = 500, max_depth = 12, max_features = 50).fit(XTrain, yTrain)
rfc = sk.ensemble.RandomForestClassifier(n_estimators = 4000, max_depth = 5, max_features = 30).fit(XTrain, yTrain)  # no max feature: 0.75
# predict on test set
rfcPred = rfc.predict(XTest)
print('Random forest:')
roc_plot(yTest, rfcPred)


# CART
# cart = sk.tree.DecisionTreeClassifier(max_features = 40, class_weight = {1:200}, random_state = 27)
cart = sk.tree.DecisionTreeClassifier(max_features = 40, class_weight = 'balanced', random_state = 27)
cart = cart.fit(XTrain, yTrain)
cartPred = cart.predict(XTest)
roc_plot(yTest, cartPred)



# One-class SVM
algorithm = sk.svm.OneClassSVM(kernel ='rbf', nu = 0.2, gamma = 0.001) 
svmModel = algorithm.fit(XTrain)

svmPred = svmModel.predict(XTest) 
svmPred = -0.5 * svmPred + 0.5      # Set labels to (1, 0) rather than (-1, 1)

print('One-class SVM:')
roc_plot(yTest, svmPred)

# Train lr, rf, svm, if

# ----------------------------------------------------
# ----------------------------------------------------
# Unsupervised methods
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop(['Primary.Dx', 'Diastolic'], axis = 1)  # EPIC_enc.drop(coeffsRm, axis = 1) 
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)


# Isolation forest
# training the model
isof = sk.ensemble.IsolationForest(n_estimators = 100, max_samples = 128, random_state = 27, 
                                    contamination = 0.1, behaviour = 'new')
_ = isof.fit(XTrain)

# predictions
# isofPredTrain = isof.predict(XTrain)
# isofPredTrain = -0.5 * isofPredTrain + 0.5
isofPredTest = isof.predict(XTest)
isofPredTest = -0.5 * isofPredTest + 0.5


# roc_plot(yTrain, isofPredTrain)
roc_plot(yTest, isofPredTest)

# Visulize the scores
_ = sns.scatterplot(x = range(len(yTest)), y = isof.score_samples(XTest), label = 'Normal')
_ = sns.scatterplot(x = np.linspace(1, len(yTest), len(yTest))[yTest == 1], 
                    y = isof.score_samples(XTest)[yTest == 1], label = 'Sepsis')
_ = sns.scatterplot(x = np.linspace(1, len(yTest), len(yTest))[(isofPredTest == 1) & (yTest != 1)], 
                    y = isof.score_samples(XTest)[(isofPredTest == 1) & (yTest != 1)], label = 'False positives')
_ = plt.legend()
_ = plt.ylabel('Score')
_ = plt.title('Isolation forest score for each instance')
plt.show()


# Extended isolation forest
eisof = eif.iForest(XTrain.values, 
                     ntrees = 50, 
                     sample_size = 128, 
                     ExtensionLevel = XTrain.shape[1] - 1)

# calculate anomaly scores
anomalyScores = eisof.compute_paths(X_in = XTest.values)
# sort the scores
anomalyScoresSorted = np.argsort(anomalyScores)
# retrieve indices of anomalous observations
indicesWithPreds = anomalyScoresSorted[-int(np.ceil( 0.19 * XTest.shape[0] )):]   # 0.002 is the anomaly ratio
# create predictions 
eisofPred = yTest * 0
eisofPred.iloc[indicesWithPreds] = 1

roc_plot(yTest, eisofPred)


# Visulize the scores
_ = sns.scatterplot(x = range(len(yTest)), y = anomalyScores)
_ = sns.scatterplot(x = np.linspace(1, len(yTest), len(yTest))[yTest == 1], 
                    y = anomalyScores[yTest == 1])
_ = sns.scatterplot(x = np.linspace(1, len(yTest), len(yTest))[eisofPred == 1], 
                    y = anomalyScores[eisofPred == 1])
plt.show()


# ----------------------------------------------------
# DBSCAN
# Scale the data
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)

# Put Will.Return into list of categorical features
robust = sk.preprocessing.RobustScaler()
X[numCols] = robust.fit_transform(X[numCols])

# Fit DBSCAN
'''
dbscan = sk.cluster.DBSCAN(eps = 3, min_samples = 4, metric = 'euclidean')
dbscanModel = dbscan.fit(X)

labels = dbscanModel.labels_
coreSamples = np.zeros_like(labels, dtype = bool)
coreSamples[dbscan.core_sample_indices_] = True
nClusters = len(set(labels)) - (1 if -1 in labels else 0)
print('Silhouette coefficient: %0.3f' %sk.metrics.silhouette_score(X, labels))

_ = plt.scatter(X['Pulse'], X['Temp'], alpha = 0.2, c = y)
_ = plt.scatter(X.loc[y == 1, 'Pulse'], X.loc[y == 1, 'Temp'], c = 'goldenrod', alpha = 0.8)
_ = plt.scatter(X.loc[labels == -1, 'Pulse'], X.loc[labels == -1, 'Temp'], marker = '+', color = 'slategrey', alpha = 0.8) 
_ = plt.xlabel('Pulse')
_ = plt.ylabel('Temp')
plt.show()

dbscanPred = labels
dbscanPred[dbscanPred == -1] = 1
dbscanPred[dbscanPred != -1] = 0
'''


# ----------------------------------------------------
# Multiple runs for logistic regression
lrResults = {}
for i in range(2000):
    XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, stratify = y)
    sm = SMOTE(random_state = 27, sampling_strategy = 'auto')
    XTrain, yTrain = sm.fit_sample(XTrain, yTrain)
    lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l2',
                                        max_iter = 1000).fit(XTrain, yTrain)
    lrPred = lr.predict(XTest)
    metrics = roc_plot(yTest, lrPred, plot = False, show_results = False)
    lrResults[str(i)] = metrics
    if i % 10 == 0:
        print('Current iteration:', i)


f1Vec = [scores[2] for scores in lrResults.values()]
_ = sns.lineplot(range(2000), f1Vec, label = 'F1')
aucVec = [scores[3] for scores in lrResults.values()]
_ = sns.lineplot(range(2000), aucVec, label = 'AUC')
recallVec = [scores[1] for scores in lrResults.values()]
_ = sns.lineplot(range(2000), recallVec, label = 'Recall')
precVec = [scores[0] for scores in lrResults.values()]
_ = sns.lineplot(range(2000), precVec, label = 'Precision')
_ = plt.ylim(-0.1, 1)
_ = plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Metrics for 2000 runs (Logistic Regression)')
plt.show()


# Tuning random forest
nEstimators = [50, 100, 200, 500, 1000, 2000, 5000]
maxDepth = [5, 10, 12, 15, 20, 50, None]
rfResults = {}
for nEst in nEstimators:
    print('Training for nEstimator =', nEst)
    for maxDep in maxDepth:
        rfc = sk.ensemble.RandomForestClassifier(n_estimators = nEst, max_depth = maxDep, max_features = 30).fit(XTrain, yTrain)
        # predict on test set
        rfcPred = rfc.predict(XTest)
        metrics = roc_plot(yTest, rfcPred,  plot = False, show_results = False)
        parameters = str(nEst) + '; ' + str(maxDep)
        rfResults[parameters] = metrics


optimParams = {'nEst':0, 'maxDep':0}
optimIndex = aucVec.index( max(aucVec) ) // len(maxDepth) % len(nEstimators)
optimParams['nEst'] = nEstimators[ optimIndex ]
optimParams['maxDep'] = maxDepth[ optimIndex ]

f1Vec = [scores[2] for scores in rfResults.values()]
_ = sns.scatterplot(range(len(rfResults)), f1Vec)
aucVec = [scores[3] for scores in rfResults.values()]
_ = sns.scatterplot(range(len(rfResults)), aucVec)
_ = plt.xlabel('No. of estimators and max depth')
_ = plt.ylabel('f1 score')
plt.show()

# 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# x, y, z = np.meshgrid(nEstimators, maxDepth, aucVec)
# _ = ax.scatter(x, y, z)
# _ = plt.xlabel('No. of estimators')
# _ = plt.ylabel('Max depth')
# _ = plt.title('f1 score')
# plt.show()


# Tuning SVM
nuList = [0.002, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5]
gammaList = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
svmResults = {}
for nu in nuList:
    for gamma in gammaList:
        algorithm = sk.svm.OneClassSVM(kernel ='rbf', nu = nu, gamma = gamma)
        svmModel = algorithm.fit(XTrain)
        svmPred = svmModel.predict(XTest) 
        svmPred[svmPred == -1] = 0       
        metrics = roc_plot(yTest, svmPred,  plot = False, show_results = False)
        parameters = str(nu) + '; ' + str(gamma)
        svmResults[parameters] = metrics
    print('\n \n \n nu =', nu, '\n \n \n')


f1Vec = [scores[2] for scores in svmResults.values()]
_ = sns.scatterplot(range(len(svmResults)), f1Vec)
aucVec = [scores[3] for scores in svmResults.values()]
_ = sns.scatterplot(range(len(svmResults)), aucVec)
_ = plt.xlabel('No. of estimators and max depth')
_ = plt.ylabel('f1 score')
plt.show()


# Tune isolation forest
nEstimators = [100] # [50, 100, 200, 500, 1000, 2000]
maxSamples = [128] # [128, 256, 512, 1024, 2048, 5012]
contamination = [0.002, 0.006, 0.01, 0.02, 0.04, 0.08, 0.1, 0.15, 0.2, 0.3]
isofResults = {}
for nEst in nEstimators:
    for maxSam in maxSamples:
        for contam in contamination:
            print('Training for nEstimator =', nEst, 'contamination:', contam)
            isof = sk.ensemble.IsolationForest(n_estimators = nEst, max_samples = maxSam, random_state = 27, 
                                                contamination = contam, behaviour = 'new')
            _ = isof.fit(XTrain)
            # predict on test set
            isofPred = isof.predict(XTest)
            isofPred = -0.5 * isofPred + 0.5
            metrics = roc_plot(yTest, isofPred,  plot = False, show_results = False)
            parameters = str(nEst) + '; ' + str(maxSam) + '; ' + str(contam)
            isofResults[parameters] = metrics


# Tune extended isolation forest
nEstimators = [50] # [50, 100, 200, 500]
maxSamples = [128] # [128, 256, 512, 1024]
contamination = [0.01, 0.015, 0.02, 0.05, 0.1, 0.15, 0.2]
eisofResults = {}
for nEst in nEstimators:
    for maxSam in maxSamples:
        for contam in contamination:
            print('Training for nEstimator =', nEst, 'contamination:', contam)
            eisof = eif.iForest(XTrain.values, 
                     ntrees = nEst, 
                     sample_size = maxSam, 
                     ExtensionLevel = XTrain.shape[1] - 1)
            anomalyScores = eisof.compute_paths(X_in = XTest.values)
            anomalyScoresSorted = np.argsort(anomalyScores)
            indicesWithPreds = anomalyScoresSorted[-int(np.ceil( contam * XTest.shape[0] )):]   
            eisofPred = yTest * 0
            eisofPred.iloc[indicesWithPreds] = 1
            # Store metrics
            metrics = roc_plot(yTest, eisofPred,  plot = False, show_results = False)
            parameters = str(nEst) + '; ' + str(maxSam) + '; ' + str(contam)
            eisofResults[parameters] = metrics




# ----------------------------------------------------
# 4-fold cross validation
# Logistic regression
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
cv = sk.model_selection.KFold(n_splits = 4, shuffle = True, random_state = 27)
lrResults = {}
i = 0
for trainIndex, testIndex in cv.split(X):
    print("Epoch:", i + 1)   
    XTrain, XTest, yTrain, yTest = X.iloc[trainIndex], X.iloc[testIndex], y.iloc[trainIndex], y.iloc[testIndex]
    sm = SMOTE(random_state = 27, sampling_strategy = 'auto')
    XTrain, yTrain = sm.fit_sample(XTrain, yTrain)
    lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l2',
                                        max_iter = 1000).fit(XTrain, yTrain)
    lrPred = lr.predict(XTest)
    metrics = roc_plot(yTest, lrPred, plot = False)
    lrResults[str(i)] = metrics
    i += 1


saveModel(lrResults, './saved_results/lrCV')


# Random forest
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
cv = sk.model_selection.KFold(n_splits = 4, shuffle = True, random_state = 27)
rfResults = {}
i = 0
for trainIndex, testIndex in cv.split(X):
    print("Epoch:", i + 1)   
    XTrain, XTest, yTrain, yTest = X.iloc[trainIndex], X.iloc[testIndex], y.iloc[trainIndex], y.iloc[testIndex]
    sm = SMOTE(random_state = 27, sampling_strategy = 'auto')
    XTrain, yTrain = sm.fit_sample(XTrain, yTrain)
    rfc = sk.ensemble.RandomForestClassifier(n_estimators = nEst, max_depth = maxDep, max_features = None).fit(XTrain, yTrain)
    # predict on test set
    rfcPred = rfc.predict(XTest)
    metrics = roc_plot(yTest, rfcPred,  plot = False)
    rfResults[parameters] = metrics
    i += 1


saveModel(rfResults, './saved_results/rfCV')


# SVM 
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
cv = sk.model_selection.KFold(n_splits = 4, shuffle = True, random_state = 27)
rfResults = {}
i = 0
for trainIndex, testIndex in cv.split(X):
    print("Epoch:", i + 1)   
    XTrain, XTest, yTrain, yTest = X.iloc[trainIndex], X.iloc[testIndex], y.iloc[trainIndex], y.iloc[testIndex]
    sm = SMOTE(random_state = 27, sampling_strategy = 'auto')
    XTrain, yTrain = sm.fit_sample(XTrain, yTrain)
    rfc = sk.ensemble.RandomForestClassifier(n_estimators = nEst, max_depth = maxDep, max_features = None).fit(XTrain, yTrain)
    # predict on test set
    rfcPred = rfc.predict(XTest)
    metrics = roc_plot(yTest, rfcPred,  plot = False)
    rfResults[parameters] = metrics
    i += 1


saveModel(rfResults, './saved_results/rfCV')



# ----------------------------------------------------
# Compare ROC plots
# Logistic regression with customized threshold
lrProba = lr.predict_proba(XTest)[:, 1]
lrPred = lrProba > 0.9
roc_plot(yTest, lrPred)
lrRoc = lr_roc_plot(yTest, lrProba, title = '(Logistic Regression)')
lrTpr = lrRoc['tpr']
lrFpr = lrRoc['fpr']
lr_roc_auc = sk.metrics.auc(lrFpr, lrTpr)


# Isolation forest
fprLst, tprLst = [], []
threshold = np.linspace(0, 1, 21)
for i in range(21):
    isof = sk.ensemble.IsolationForest(n_estimators = 100, max_samples = 128, random_state = 27, 
                                    contamination = threshold[i], behaviour = 'new')
    _ = isof.fit(XTrain)
    isofPred = isof.predict(XTest)
    isofPred = -0.5 * isofPred + 0.5
    fpr, tpr, _ = sk.metrics.roc_curve(yTest, isofPred)
    fprLst.append(fpr[1])
    tprLst.append(tpr[1])
    if i % 4 == 0:
        print('Current iteration:', i)


fprLst[-1], tprLst[-1] = 0, 0
fprLst[0], fprLst[0] = 1, 1
sortInd = np.argsort(fprLst)
isofFpr = np.sort(fprLst)
isofTpr = [tprLst[item] for item in sortInd]
if_roc_auc = sk.metrics.auc(isofFpr, isofTpr)
_ = plt.title('Receiver Operating Characteristic (iForest)')
_ = plt.plot(isofFpr, isofTpr, 'b', label = 'AUC = %0.2f' % if_roc_auc)
_ = plt.legend(loc = 'lower right')
_ = plt.plot([0, 1], [0, 1],'r--')
_ = plt.xlim([0, 1])
_ = plt.ylim([0, 1.01])
_ = plt.ylabel('True Positive Rate')
_ = plt.xlabel('False Positive Rate')
plt.show()


# Extended isolation forest
fprLst, tprLst = [], []
threshold = np.linspace(0, 1, 21)
for i in range(21):
    eisof = eif.iForest(XTrain.values, 
                     ntrees = 50, 
                     sample_size = 128, 
                     ExtensionLevel = XTrain.shape[1] - 1)
    anomalyScores = eisof.compute_paths(X_in = XTest.values)
    anomalyScoresSorted = np.argsort(anomalyScores)
    indicesWithPreds = anomalyScoresSorted[-int(np.ceil( threshold[i] * XTest.shape[0] )):]  
    eisofPred = yTest * 0
    eisofPred.iloc[indicesWithPreds] = 1
    fpr, tpr, _ = sk.metrics.roc_curve(yTest, eisofPred)
    fprLst.append(fpr[1])
    tprLst.append(tpr[1])
    if i % 4 == 0:
        print('Current iteration:', i)


fprLst[-1], tprLst[-1] = 0, 0
fprLst[0], fprLst[0] = 1, 1
sortInd = np.argsort(fprLst)
eisofFpr = np.sort(fprLst)
eisofTpr = [tprLst[item] for item in sortInd]
eif_roc_auc = sk.metrics.auc(eisofFpr, eisofTpr)
_ = plt.title('Receiver Operating Characteristic (Extended iForest)')
_ = plt.plot(isofFpr, isofTpr, 'b', label = 'AUC = %0.2f' % eif_roc_auc)
_ = plt.legend(loc = 'lower right')
_ = plt.plot([0, 1], [0, 1],'r--')
_ = plt.xlim([0, 1])
_ = plt.ylim([0, 1.01])
_ = plt.ylabel('True Positive Rate')
_ = plt.xlabel('False Positive Rate')
plt.show()


# Plot LR, IF and EIF on the same ROC
_ = plt.title('Receiver Operating Characteristic')
_ = plt.plot(lrFpr, lrTpr, 'b', label = 'Logistic regression; AUC = %0.2f' % lr_roc_auc)
_ = plt.plot(isofFpr, isofTpr, 'c-.', label = 'iForest; AUC = %0.2f' % if_roc_auc)
_ = plt.plot(eisofFpr, eisofTpr, 'm:', label = 'Extended iForest; AUC = %0.2f' % eif_roc_auc)
_ = plt.legend(loc = 'lower right')
_ = plt.plot([0, 1], [0, 1],'r--')
_ = plt.xlim([0, 1])
_ = plt.ylim([0, 1.01])
_ = plt.ylabel('True Positive Rate')
_ = plt.xlabel('False Positive Rate')
plt.show()


# ----------------------------------------------------
# Bar plot of AUC
lrPred = loadModel('./saved_results/lrPred21')
cartPred = loadModel('./saved_results/cartPred21')
rfcPred = loadModel('./saved_results/rfcPred21')
svmPred = loadModel('./saved_results/svmPred21')
isofPred = loadModel('./saved_results/isofPred21')
eisofPred = loadModel('./saved_results/eisofPred21')

lrResults = roc_plot(yTest, lrPred, plot = False, show_results = False)
cartResults = roc_plot(yTest, cartPred, plot = False, show_results = False)
rfcResults = roc_plot(yTest, rfcPred, plot = False, show_results = False)
svmResults = roc_plot(yTest, svmPred, plot = False, show_results = False)
isofResults = roc_plot(yTest, isofPred, plot = False, show_results = False)
eisofResults = roc_plot(yTest, eisofPred, plot = False, show_results = False)

modelNames = ['Logistic regression', 'CART', 'Random Forest', 'SVM', 'iForest', 'Extended iForests']
summary = {'model':modelNames, 'precision':0, 'recall':0, 'f1_score':0, 'auc':0}
metrics = list(summary.keys())
for i in range( len(summary) - 1 ):
    key = metrics[i + 1]
    ls = []
    ls.append(lrResults[i])
    ls.append(cartResults[i])
    ls.append(rfcResults[i])
    ls.append(svmResults[i])
    ls.append(isofResults[i])
    ls.append(eisofResults[i])
    summary[key] = ls


# Plot all AUC
summary = pd.DataFrame(summary)
# The AUC for lr, iforest and extended iForest should be computed from more than 1 threshold
summary.loc[[0, 4, 5], 'auc'] = [lr_roc_auc, if_roc_auc, eif_roc_auc]
_ = sns.barplot(y = summary['model'], x = summary['auc'], data = summary)
_ = plt.xlim([0, 1])
for i in range( len( modelNames ) ):
    auc = summary['auc']
    _ = plt.text(y = i  , x = 0.98, s = np.round(auc[i], 3), size = 10)

plt.show()

# Plot all f1 score
_ = sns.barplot(y = summary['model'], x = summary['f1_score'], data = summary)
_ = plt.xlim([0, 1])
for i in range( len( modelNames ) ):
    f1 = summary['f1_score']
    _ = plt.text(y = i  , x = 0.2, s = np.round(f1[i], 3), size = 10)

plt.show()

