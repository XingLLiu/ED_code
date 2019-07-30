# ----------------------------------------------------
from ED_support_module import *                      # Source required modules and functions
from EDA import EPIC, EPIC_enc, EPIC_CUI, numCols, catCols     # Source datasets from EDA.py


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

smote = SMOTE(random_state = 27, sampling_strategy = 'auto') # 0.4
XTrain, yTrain = smote.fit_sample(XTrain, yTrain)


# SMOTE with PCA
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
colNames = X.columns
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)
smote = SMOTE(random_state = 27, sampling_strategy = 'auto')
XTrain, yTrain = smote.fit_sample(XTrain, yTrain)
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
XTrain = pd.DataFrame(XTrain, columns = X.columns)
XTrain, XTest = XTrain.iloc[:, whichKeep], XTest.iloc[:, whichKeep]
lr2 = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l2', 
                                            max_iter = 1000).fit(XTrain, yTrain)

lrPred2 = lr2.predict(XTest)
roc_plot(yTest, lrPred2)

# Feature importance using permutation test
impVals, impAll = mlxtend.evaluate.feature_importance_permutation(
                    predict_method = lr2.predict, 
                    X = np.array(XTest),
                    y = np.array(yTest),
                    metric = 'accuracy',
                    num_rounds = 10,
                    seed = 27)


std = np.std(impAll, axis=1)
indices = np.argsort(impVals)[::-1]
# Plot importance values
_ = plt.figure()
_ = plt.title("Logistic regression feature importance via permutation importance w. std. dev.")
_ = sns.barplot(y = XTest.columns[indices], x = impVals[indices],
                xerr = std[indices])
_ = plt.yticks(fontsize = 8)
plt.show()

# Plot beta values
nonZeroCoeffs = lr.coef_[lr.coef_ != 0]
indices = np.argsort(abs(nonZeroCoeffs))[::-1][:50]
_ = plt.figure()
_ = plt.title("Logistic regression feature importance via permutation importance w. std. dev.")
_ = sns.barplot(y = XTest.columns[indices], x = np.squeeze(nonZeroCoeffs)[indices])
_ = plt.yticks(fontsize = 8)
plt.show()


# Random forest
# rfc = sk.ensemble.RandomForestClassifier(n_estimators = 1000, max_depth = 12, max_features = 50).fit(XTrain, yTrain)
# rfc = sk.ensemble.RandomForestClassifier(n_estimators = 500, max_depth = 12, max_features = 50).fit(XTrain, yTrain)
# rfc = sk.ensemble.RandomForestClassifier(n_estimators = 4000, max_depth = 5, max_features = 30).fit(XTrain, yTrain)
rfc = sk.ensemble.RandomForestClassifier(n_estimators = 4000, max_depth = 5, max_features = 'auto', min_samples_split = 2).fit(XTrain, yTrain)  # no max feature: 0.75
# predict on test set
rfcPred = rfc.predict(XTest)
print('Random forest:')
roc_plot(yTest, rfcPred)

# Random forest feature importance
# i) using Gini impurity
importanceVals = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importanceVals)[::-1]
# Plot the feature importances of the forest
_ = plt.figure()
_ = plt.title("Random Forest feature importance (impurity)")
_ = sns.barplot(y = XTrain.columns[indices], x = importanceVals[indices],
                xerr = std[indices])
_ = plt.yticks(fontsize = 8)
plt.show()

# ii) using permutation test
impVals, impAll = mlxtend.evaluate.feature_importance_permutation(
                    predict_method = rfc.predict, 
                    X = np.array(XTest),
                    y = np.array(yTest),
                    metric = 'accuracy',
                    num_rounds = 10,
                    seed = 27)


std = np.std(impAll, axis=1)
indices = np.argsort(impVals)[::-1]
# Plot importance values
_ = plt.figure()
_ = plt.title("Random Forest feature importance via permutation importance w. std. dev.")
_ = sns.barplot(y = XTrain.columns[indices], x = impVals[indices],
                xerr = std[indices])
_ = plt.yticks(fontsize = 8)
plt.show()


# CART
# cart = sk.tree.DecisionTreeClassifier(max_features = 40, class_weight = {1:200}, random_state = 27)
cart = sk.tree.DecisionTreeClassifier(max_features = 40, class_weight = 'balanced', random_state = 27)
cart = cart.fit(XTrain, yTrain)
cartPred = cart.predict(XTest)
roc_plot(yTest, cartPred)



# ----------------------------------------------------
# ----------------------------------------------------
# Unsupervised methods
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop(['Primary.Dx', 'Diastolic'], axis = 1)  # EPIC_enc.drop(coeffsRm, axis = 1) 
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)
XTrainNormal = XTrain.loc[yTrain == 0, :]


# One-class SVM
# algorithm = sk.svm.OneClassSVM(kernel ='rbf', nu = 0.2, gamma = 0.001) 
algorithm = sk.svm.OneClassSVM(kernel ='rbf', nu = 0.1, gamma = 0.0001) 
svmModel = algorithm.fit(XTrain.loc[yTrain == 0, :])

svmPred = svmModel.predict(XTest) 
svmPred = -0.5 * svmPred + 0.5      # Set labels to (1, 0) rather than (-1, 1)

print('One-class SVM:')
roc_plot(yTest, svmPred)


# Isolation forest
# training the model
isof = sk.ensemble.IsolationForest(n_estimators = 100, max_samples = 128, random_state = 27, 
                                    contamination = 0.1, behaviour = 'new')
_ = isof.fit(XTrainNormal)

# predictions
# isofPredTrain = isof.predict(XTrain)
# isofPredTrain = -0.5 * isofPredTrain + 0.5
isofPredTest = isof.predict(XTest)
isofPredTest = np.array(-0.5 * isofPredTest + 0.5, dtype = 'int')

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

# ii) using permutation test
impVals, impAll = mlxtend.evaluate.feature_importance_permutation(
                    predict_method = isof.predict, 
                    X = np.array(XTest),
                    y = np.array(1 -2 * yTest),
                    metric = 'accuracy',
                    num_rounds = 10,
                    seed = 27)


std = np.std(impAll, axis=1)
indices = np.argsort(impVals)[::-1]
# Plot importance values
_ = plt.figure()
_ = plt.title("Random Forest feature importance via permutation importance w. std. dev.")
_ = sns.barplot(y = XTrain.columns[indices], x = impVals[indices],
                xerr = std[indices])
_ = plt.yticks(fontsize = 8)
plt.show()


# Extended isolation forest
# # Normalizing the numerical features
# y = EPIC_enc['Primary.Dx']
# X = EPIC_enc.drop('Primary.Dx', axis = 1)
# XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)

# # Transform the numerical features
# XTrainNum, XTestNum = pd.DataFrame(XTrain)[numCols], pd.DataFrame(XTest)[numCols]
# transformer = sk.preprocessing.RobustScaler().fit(XTrainNum)
# XTrainNum = transformer.transform(XTrainNum)
# XTestNUm = transformer.transform(XTestNum)
# XTrainNum, XTestNum = pd.DataFrame(XTrainNum), pd.DataFrame(XTestNum)
# XTrainNum.index, XTestNum.index = XTrain.index, XTest.index

# # Replace the numerical features with the transfered features
# XTrain = pd.concat([pd.DataFrame(XTrainNum), pd.DataFrame(XTrain.drop(numCols, axis = 1))], axis = 1, sort = False)
# XTest = pd.concat([pd.DataFrame(XTestNum), pd.DataFrame(XTest.drop(numCols, axis = 1))], axis = 1, sort = False)
# XTrainNormal = XTrain.loc[yTrain == 0, :]

# eisof = eif.iForest(XTrain.values, 
#                      ntrees = 50, 
#                      sample_size = 128, 
#                      ExtensionLevel = XTrain.shape[1] - 1)
eisof = eif.iForest(XTrainNormal.values, 
                     ntrees = 25, 
                     sample_size = 128, 
                     ExtensionLevel = 40)

# calculate anomaly scores
anomalyScores = eisof.compute_paths(X_in = XTest.values)
# sort the scores
anomalyScoresSorted = np.argsort(anomalyScores)
# retrieve indices of anomalous observations
indicesWithPreds = anomalyScoresSorted[-int(np.ceil( 0.1 * XTest.shape[0] )):]   # 0.002 is the anomaly ratio
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

dbscanPred = labels.copy()
dbscanPred[dbscanPred != -1] = 0
dbscanPred[dbscanPred == -1] = 1
roc_plot(y, dbscanPred)

# Proportion of class 1 lables for each cluster
for i in pd.Series(labels).unique():
    print(i)
    print( sum((dbscanPred == i) & (y == 1)) / sum(dbscanPred == i) )


'''


# ----------------------------------------------------
# Multiple runs for logistic regression
lrResults = {}
for i in range(2000):
    XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, stratify = y)
    smote = SMOTE(random_state = 27, sampling_strategy = 'auto')
    XTrain, yTrain = smote.fit_sample(XTrain, yTrain)
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


# ----------------------------------------------------
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



# Number of trees in random forest
n_estimators = [200, 500, 1000, 2000]
n_estimators.append(3000)
n_estimators.append(4000)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2]   # [2, 5, 10]

randomGrid = {'n_estimators': n_estimators, 'max_features': max_features, 
              'min_samples_split': min_samples_split, 
              'max_depth': max_depth}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rfc = sk.ensemble.RandomForestClassifier()
# Random search of parameters, using 4 fold cross validation, 
rfRandom = sk.model_selection.GridSearchCV(estimator = rfc, param_grid = randomGrid, cv = 4,
                                           verbose = 2, n_jobs = -1, scoring = 'f1')
# Fit the random search model
_ = rfRandom.fit(XTrain, yTrain)
print('Best params RFC: ', rfRandom.best_params_)


# ----------------------------------------------------
# Tuning SVM
nuList = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5]
gammaList = [0.00001, 0.0001, 0.001, 0.01, 0.1]
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



# nu
nu = [0.01, 0.02, 0.1, 0.2, 0.3, 0.5]
# Gamma for the extend of fit
gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 'auto_deprecated']
# kernel
kernel = ['rbf']

randomGrid = {'nu': nu, 'gamma': gamma, 'kernel': kernel}

# First create the base model to tune
svm = sk.svm.OneClassSVM()
# Random search of parameters, using 4 fold cross validation, 
svmRandom = sk.model_selection.GridSearchCV(estimator = svm, param_grid = randomGrid, cv = 4,
                                            verbose = 2, n_jobs = -1, scoring = 'f1')
# Fit the random search model
_ = svmRandom.fit(XTrainNormal, yTrain.loc[yTrain == 0] + 1)
print('Best params SVM: ', svmRandom.best_params_)


# With poly kernel
# kernel
kernel = ['poly']
# Degree
degree = [1, 3, 5, 7]
randomGrid = {'nu': nu, 'gamma': gamma, 'kernel': kernel, 'degree': degree}
# First create the base model to tune
svm = sk.svm.OneClassSVM()
# Random search of parameters, using 4 fold cross validation, 
svmRandom = sk.model_selection.GridSearchCV(estimator = svm, param_grid = randomGrid, cv = 4, 
                                                  verbose = 2, n_jobs = -1, scoring = 'f1')
# Fit the random search model
_ = svmRandom.fit(XTrainNormal, yTrain.loc[yTrain == 0] + 1)
print('Best params SVM: ', svmRandom.best_params_)


# ----------------------------------------------------
# Tuning isolation forest
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


# ----------------------------------------------------
# Tuning extended isolation forest
nEstimators = [25, 50, 100, 400] # [50] 
maxSamples = [128, 256, 1024] # [128] 
contamination = [0.01, 0.08, 0.2]
extensionLevel = [0, 10, 20, 40, XTrain.shape[1] - 1]
eisofResults = {}
for nEst in nEstimators:
    for maxSam in maxSamples:
        for contam in contamination:
            for extensionLev in extensionLevel:
                print('Training for nEstimator =', nEst, 'Ex Level:', extensionLev)
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
                parameters = str(nEst) + '; ' + str(maxSam) + '; ' + str(contam) + ';' + str(extensionLev)
                eisofResults[parameters] = metrics


f1Vec = [scores[2] for scores in eisofResults.values()]
_ = sns.scatterplot(range(len(eisofResults)), f1Vec)
aucVec = [scores[3] for scores in eisofResults.values()]
_ = sns.scatterplot(range(len(eisofResults)), aucVec)
_ = plt.xlabel('No. of estimators and max depth')
_ = plt.ylabel('f1 score')
plt.show()

params = list(eisofResults.keys())
bestParams = params[aucVec.index(max(aucVec))]



# ----------------------------------------------------
# 4-fold cross validation
# Logistic regression
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
cv = sk.model_selection.KFold(n_splits = 4, shuffle = True, random_state = 27)
lrResults = {}
i = 0; tpr = 0; tnr = 0;
for trainIndex, testIndex in cv.split(X):
    print("Epoch:", i + 1)   
    XTrain, XTest, yTrain, yTest = X.iloc[trainIndex], X.iloc[testIndex], y.iloc[trainIndex], y.iloc[testIndex]
    smote = SMOTE(random_state = 27, sampling_strategy = 'auto')
    XTrain, yTrain = smote.fit_sample(XTrain, yTrain)
    lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l2',
                                        max_iter = 1000).fit(XTrain, yTrain)
    lrProba = lr.predict_proba(XTest)[:, 1]
    lrPred = lrProba > 0.5
    metrics = roc_plot(yTest, lrPred, plot = False)
    lrResults[str(i)] = metrics
    i += 1
    tpr += metrics[1]
    tnr += (lrPred == 0).sum() / (yTest == 0).sum()


print(tpr/4, tnr/4)

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
    smote = SMOTE(random_state = 27, sampling_strategy = 'auto')
    XTrain, yTrain = smote.fit_sample(XTrain, yTrain)
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
    smote = SMOTE(random_state = 27, sampling_strategy = 'auto')
    XTrain, yTrain = smote.fit_sample(XTrain, yTrain)
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
lrPred = lrProba > 0.5
roc_plot(yTest, lrPred, plot = False)
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
    _ = isof.fit(XTrainNormal)
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
    eisof = eif.iForest(XTrainNormal.values, 
                     ntrees = 25, 
                     sample_size = 128, 
                     ExtensionLevel = 40)
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
_ = plt.plot(eisofFpr, eisofTpr, 'b', label = 'AUC = %0.2f' % eif_roc_auc)
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
_ = plt.plot(eisofFpr, np.sort(eisofTpr), 'm:', label = 'Extended iForest; AUC = %0.2f' % eif_roc_auc)
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


# AUC plot
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



# ----------------------------------------------------
'''
# LR
XTrainMat = sm.add_constant(XTrain)
lr2 = sm.GLM(yTrain, XTrainMat, family = sm.families.Binomial()).fit(maxiter = 1000)
print(lr2.summary())
lrPred2 = lr2.mu


fig, ax = plt.subplots()

ax.scatter(lrPred2, lr2.resid_pearson)
ax.hlines(0, 0, 1)
ax.set_xlim(0, 1)
ax.set_title('Residual Dependence Plot')
ax.set_ylabel('Pearson Residuals')
ax.set_xlabel('Fitted values')
'''


# ----------------------------------------------------
# Clinical notes
# Show term frequencies for Sepsis
ifSepsis = EPIC['Primary.Dx'] == 1
CUISepsis = EPIC_CUI.iloc[ifSepsis.values]

triageNotes = {}
# The following for loops can be simplified
for i in CUISepsis.index:
    cuiLst = [cui for cui in CUISepsis.loc[i, 'Triage.Notes']]
    for cui in cuiLst:
        triageNotes[cui] = 0            


# Count number of occurance
for i in CUISepsis.index:
    cuiLst = [cui for cui in CUISepsis.loc[i, 'Triage.Notes']]
    for cui in cuiLst:
        try:
            triageNotes[cui] += 1
        except:
            continue


# Create TF-IDF dataframe
triageDf = pd.DataFrame(index = range(len(EPIC_CUI)), columns = range(len(triageNotes)), dtype = 'float')
triageDf.iloc[:, :] = 0
triageDf.columns = triageNotes.keys()
triageDf.index = EPIC_enc.index

# Compute TF and IDF
corpusLen = sum(triageNotes.values())
for i in triageDf.index:
    notes = EPIC_CUI.loc[i, 'Triage.Notes']
    for cui in notes:
        if cui in triageNotes.keys():
            # TF 
            tf = sum([term == cui for term in notes]) / len(notes)
            # IDF 
            idf = np.log( corpusLen / triageNotes[cui] )
            # Store TF-IDF
            triageDf.loc[i, cui] = tf * idf


# Append to EPIC_enc
EPIC_enc = pd.concat([EPIC_enc, triageDf], axis = 1, sort = False)

# Split to train and test
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop(['Primary.Dx', 'Diastolic'], axis = 1)  # EPIC_enc.drop(coeffsRm, axis = 1) 
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)
XTrainNormal = XTrain.loc[yTrain == 0, :]





# Full triage note
# Append triage notes to X
triageNotes = {}
for i in EPIC_enc.index:
    cuiLst = [cui for cui in EPIC_CUI.loc[i, 'Triage.Notes']]
    for cui in cuiLst:
        try:
            triageNotes[cui] += 1
        except:
            triageNotes[cui] = 1

