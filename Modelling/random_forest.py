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


# Tuning mode
try:
    mode = sys.argv[2]
except:
    mode = 'normal'


# Class weight
try:
    weight = int(sys.argv[3])
except:
    weight = 2000

# ----------------------------------------------------
# Create a directory if not exists
results_path = 'saved_results/random_forest/' + mode
plot_path = '/'.join(os.getcwd().split('/')[:3]) + '/Pictures/random_forest/'
if not os.path.exists(results_path):
    os.makedirs(results_path)


if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    

# ----------------------------------------------------
# SMOTE with PCA
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
colNames = X.columns
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=seed, stratify = y)

# Use smote if weight = 1
if weight == 1:
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
if mode == 'normal':
    print('Start fitting random forest ...\n')
    # Fit random forest
    # rfc = sk.ensemble.RandomForestClassifier(n_estimators = 4000, max_depth = 5, max_features = 'auto', min_samples_split = 2).fit(XTrain, yTrain)  # no max feature: 0.75
    # rfc = sk.ensemble.RandomForestClassifier(n_estimators = 200, max_depth = 50, max_features = 'auto', min_samples_split = 2).fit(XTrain, yTrain)
    rfc = sk.ensemble.RandomForestClassifier(n_estimators = 4000, class_weight = {0:1, 1:weight}).fit(XTrain, yTrain)  # no max feature: 0.75
    print('Complete\n')
    
    # predict on test set
    rfcPred = rfc.predict(XTest)
    rfcProbs = rfc.predict_proba(XTest)[:, 1]

    # Show results
    print('Random forest:')
    rfcRoc = lr_roc_plot(yTest, rfcProbs, save_path = plot_path + 'roc.eps')
    rfcTpr = rfcRoc['tpr']
    rfcFpr = rfcRoc['fpr']
    rfc_roc_auc = sk.metrics.auc(rfcFpr, rfcTpr)
    print( '\nWith TNR:{}, TPR:{}'.format( round( 1 - rfcFpr[5], 4), round(rfcTpr[5], 4) ) )

    importanceVals = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
                axis=0)
    indices = np.argsort(importanceVals)[::-1]

    # Feature importances of the forest
    _ = plt.figure()
    _ = plt.title("Random Forest feature importance (impurity)")
    _ = sns.barplot(y = XTrain.columns[indices], x = importanceVals[indices],
                    xerr = std[indices])
    _ = plt.yticks(fontsize = 8)
    plt.savefig(path + 'feature_importance.eps', format='eps', dpi=800)
    plt.show()


# ----------------------------------------------------
if mode != 'normal':
    # Tuning
    # Number of trees in random forest
    n_estimators = [200, 500, 1000, 2000, 3000, 4000]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2]
    # Class weight
    class_weight = [{0:1, 1:1000}, {0:1, 1:2000}, {0:1, 1:3000}]

    randomGrid = {'n_estimators': n_estimators, 'max_features': max_features, 
                'min_samples_split': min_samples_split,
                'max_depth': max_depth,
                'class_weight':class_weight}


    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rfc = sk.ensemble.RandomForestClassifier()
    # Random search of parameters, using 4 fold cross validation, 
    rfRandom = sk.model_selection.GridSearchCV(estimator = rfc, param_grid = randomGrid, cv = 4,
                                            verbose = 2, n_jobs = -1, scoring = 'f1')
    # Fit the random search model
    _ = rfRandom.fit(XTrain, yTrain)
    print('Best params RFC: ', rfRandom.best_params_)
