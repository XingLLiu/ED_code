# ----------------------------------------------------
from ED_support_module import *
from EDA import EPIC, EPIC_enc, EPIC_CUI, numCols, catCols

# ----------------------------------------------------
# Input random seed. seed = 27 by default.
try: 
    seed = int(sys.argv[1])
except:
    seed = 27


# ----------------------------------------------------
# SMOTE
# Separate input features and target
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state = 27, sampling_strategy = 'auto')
XTrain, yTrain = smote.fit_sample(XTrain, yTrain)

# ----------------------------------------------------
# Logistic regression with L1 loss
lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l1',
                                        max_iter = 1000).fit(XTrain, yTrain)
lrPred = lr.predict(XTest)
roc_plot(yTest, lrPred)

# Remove the zero coefficients
ifZero = (lr.coef_ == 0).reshape(-1)
notZero = (lr.coef_ != 0).reshape(-1)
coeffs = [X.columns[i] for i in range(X.shape[1]) if not ifZero[i]]
coeffsRm = [X.columns[i] for i in range(X.shape[1]) if ifZero[i]]
print('\nFeatures with zero coefficients:\n', coeffsRm, '\n')

# Refit 
whichKeep = pd.Series( range( len( notZero ) ) )
whichKeep = whichKeep.loc[notZero]
XTrain = pd.DataFrame(XTrain, columns = X.columns)
XTrain, XTest = XTrain.iloc[:, whichKeep], XTest.iloc[:, whichKeep]
lr2 = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l2',
                                            max_iter = 1000).fit(XTrain, yTrain)

lrPred2 = lr2.predict(XTest)
roc_plot(yTest, lrPred2)


# ----------------------------------------------------
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
_ = plt.title("Logistic regression feature importance via permutation importance with std. dev.")
_ = sns.barplot(y = XTest.columns[indices], x = impVals[indices],
                xerr = std[indices])
_ = plt.yticks(fontsize = 8)
plt.show()

# Plot beta values
nonZeroCoeffs = lr2.coef_[lr2.coef_ != 0]
indices = np.argsort(abs(nonZeroCoeffs))[::-1][:50]
_ = plt.figure()
_ = plt.title("Logistic regression values of coefficients.")
_ = sns.barplot(y = XTest.columns[indices], x = np.squeeze(nonZeroCoeffs)[indices])
_ = plt.yticks(fontsize = 8)
plt.show()


# ----------------------------------------------------
# Plot AUC
lrProba = lr2.predict_proba(XTest)[:, 1]
lrRoc = lr_roc_plot(yTest, lrProba, title = '(Logistic Regression)', n_pts = 101)
lrTpr = lrRoc['tpr']
lrFpr = lrRoc['fpr']
lr_roc_auc = sk.metrics.auc(lrFpr, lrTpr)
print( '\nWith TNR:{}, TPR:{}'.format( round( 1 - lrFpr[5], 4), round(lrTpr[5], 4) ) )