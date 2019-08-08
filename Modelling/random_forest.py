# ----------------------------------------------------
from ED_support_module import *                                # Source required modules and functions
from EDA import EPIC, EPIC_enc, EPIC_CUI, numCols, catCols     # Source datasets from EDA.py


# ----------------------------------------------------
# Path to save figures
path = '/'.join(os.getcwd().split('/')[:3]) + '/Pictures/random_forest/'
# Create folder if not already exist
if not os.path.exists(path):
    os.makedirs(path)


# ----------------------------------------------------
# Input random seed. seed = 27 by default.
try: 
    seed = int(sys.argv[1])
except:
    seed = 27


# ----------------------------------------------------
# SMOTE with PCA
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
colNames = X.columns
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=seed, stratify = y)
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


# ----------------------------------------------------
# Fit random forest
rfc = sk.ensemble.RandomForestClassifier(n_estimators = 4000, max_depth = 5, max_features = 'auto', min_samples_split = 2).fit(XTrain, yTrain)  # no max feature: 0.75
# predict on test set
rfcPred = rfc.predict(XTest)
print('Random forest:')
roc_plot(yTest, rfcPred)

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


# ----------------------------------------------------
# Tuning
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
