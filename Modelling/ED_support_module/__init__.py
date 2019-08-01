import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sklearn.preprocessing as skprep
from sklearn import model_selection
from sklearn import linear_model, impute, ensemble, svm
import sklearn as sk
from imblearn.over_sampling import SMOTE
import re
import numpy as np
import os
from scipy import stats
import pickle
import eif
from mpl_toolkits import mplot3d   # For 3D plots
import mlxtend.evaluate            # For feature importance

plt.style.use('seaborn')


def double_scatter(x_feature = None, y_feature = None, classes = 'Primary.Dx', data = None):
    '''
    Scatter plot for y_feature against x_feature, coloured accourding to classes.
    Input:  x_feature: name of feature on the x axis (str)
            y_feature: name of feature on the y axis (str)
            class: multi-class categorical variable used for colouring
            EPIC: the EPIC dataset
    Output: plt.scatter
    '''
    plt.scatter(data[x_feature], data[y_feature], alpha = 0.7, c = data[classes])
    plt.scatter(data.loc[data[classes] == 1, x_feature], data.loc[data[classes] == 1, y_feature], c = 'goldenrod')
    _ = plt.xlabel(x_feature)
    _ = plt.ylabel(y_feature)
    plt.show()


# ROC curve
def roc_plot(yTest = None, pred = None, plot = True, show_results = True):
    '''
    Plot the roc curve of a given test set and predictions
    Input:  yTest = test set (pd.dataframe or series)
            pred = predictions (pd.dataframe or seires)
    Output: ROC plot
    '''
    precision = sk.metrics.precision_score(yTest, pred)
    f1_score = sk.metrics.f1_score(yTest, pred)
    recall = sk.metrics.recall_score(yTest, pred)
    confMat = sk.metrics.confusion_matrix(yTest, pred)
    fpr, tpr, _ = sk.metrics.roc_curve(yTest, pred)
    roc_auc = sk.metrics.auc(fpr, tpr)
    if plot == True:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    if show_results:
        print(' precision: {} \n recall:    {} \n f1_score:  {} '.format(precision, recall, f1_score)) 
        print(' Confusion matrix: \n')
        print(confMat)
    return([precision, recall, f1_score, roc_auc])


# Full ROC curve for logistic regression
def lr_roc_plot(yTest = None, proba = None, plot = True, title = None, n_pts = 51):
    '''
    Plot the roc curve of a trained logistic regression model.
    Input:  yTest = test set (pd.dataframe or series)
            proba = predicted probability (np.array)
    Output: ROC plot
    '''
    fprLst, tprLst = [], []
    threshold = np.linspace(0, 1, n_pts)
    scoreSorted = np.argsort(proba)
    for i in range(n_pts):
        indicesWithPreds = scoreSorted[-int(np.ceil( threshold[i] * yTest.shape[0] )):] 
        pred = yTest * 0
        pred.iloc[indicesWithPreds] = 1
        fpr, tpr, _ = sk.metrics.roc_curve(yTest, pred)
        fprLst.append(fpr[1])
        tprLst.append(tpr[1])
    fprLst[-1], tprLst[-1] = 1, 1
    fprLst[0], tprLst[0] = 0, 0
    roc_auc = sk.metrics.auc(fprLst, tprLst)
    if plot == True:
        plt.title('Receiver Operating Characteristic ' + title)
        plt.plot(fprLst, tprLst, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    return({'tpr':tprLst, 'fpr':fprLst})


def if_roc_plot(yTest = None, score = None, plot = True, 
                title = None, n_pts = 51, extended = False):
    '''
    Plot the roc curve of a trained isolation forest model.
    Input:  yTest = test set (pd.dataframe or series)
            proba = predicted probability (np.array)
    Output: ROC plot
    '''
    fprLst, tprLst = [], []
    threshold = np.linspace(0, 1, n_pts)
    scoreSorted = np.argsort(score)
    for i in range(n_pts):
        if not extended:
            indicesWithPreds = scoreSorted[:int(np.ceil( threshold[i] * yTest.shape[0] ))] 
        else:
            indicesWithPreds = scoreSorted[-int(np.ceil( threshold[i] * yTest.shape[0] )):] 
        pred = yTest * 0
        pred.iloc[indicesWithPreds] = 1
        fpr, tpr, _ = sk.metrics.roc_curve(yTest, pred)
        fprLst.append(fpr[1])
        tprLst.append(tpr[1])
    fprLst[-1], tprLst[-1] = 1, 1
    fprLst[0], tprLst[0] = 0, 0
    sortInd = np.argsort(fprLst)
    fpr = np.sort(fprLst)
    tpr = [tprLst[item] for item in sortInd]
    roc_auc = sk.metrics.auc(fpr, tpr)
    if plot == True:
        _ = plt.title('Receiver Operating Characteristic ' + title)
        _ = plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        _ = plt.legend(loc = 'lower right')
        _ = plt.plot([0, 1], [0, 1],'r--')
        _ = plt.xlim([0, 1])
        _ = plt.ylim([0, 1.01])
        _ = plt.ylabel('True Positive Rate')
        _ = plt.xlabel('False Positive Rate')
        plt.show()
    return({'tpr':tprLst, 'fpr':fprLst})


# Show various metrics
def metricsPlot(results, model_name):
    '''
    Show plots for various evaluation metrics.
    Input:  results: a dictionary with precision, recall, f1 score and AUC (in that order).
            model_name: name of the model used (str; to be shown in the title).
    Output: plot of these metrics
    '''
    xVec = range(len(results))
    f1Vec = [scores[2] for scores in results.values()]
    _ = sns.lineplot(xVec, f1Vec, label = 'F1')
    aucVec = [scores[3] for scores in results.values()]
    _ = sns.lineplot(xVec, aucVec, label = 'AUC')
    recallVec = [scores[1] for scores in results.values()]
    _ = sns.lineplot(xVec, recallVec, label = 'Recall')
    precVec = [scores[0] for scores in results.values()]
    _ = sns.lineplot(xVec, precVec, label = 'Precision')
    _ = plt.ylim(-0.1, 1)
    _ = plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title('ROC ' + model_name)
    plt.show()


# Save models
def saveModel(model, filename):
    '''
    Save sklearn model as 'filename' in current directory.
    Input:  model: sklearn model
            filename: name of the model file (str)
    '''
    pickle.dump(model, open(filename, 'wb'))


# Load models
def loadModel(filename):
    '''
    Load sklearn model 'filename' in current directory.
    Input:  model: sklearn model
            filename: name of the model file (str)
    '''
    return(pickle.load(open(filename, 'rb')))



