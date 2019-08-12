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
import mlxtend.evaluate            # For feature importance
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import sys
import datetime
from tqdm import tqdm, trange


plt.style.use('seaborn')


# Pair plot
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
def roc_plot(yTest = None, pred = None, plot = True, show_results = True,
             save_path = None):
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
        if not save_path == None:
            plt.savefig(save_path, format='eps', dpi=1000)
        plt.show()
    if show_results:
        print(' precision: {} \n recall:    {} \n f1_score:  {} '.format(precision, recall, f1_score)) 
        print(' \nConfusion matrix:')
        print(confMat)
    return([precision, recall, f1_score, roc_auc])


# Full ROC curve for logistic regression
def lr_roc_plot(yTest = None, proba = None, plot = True, title = ' ', n_pts = 51, save_path = None):
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
        if not save_path == None:
            plt.savefig(save_path, format='eps', dpi=1000)
        plt.show()
    return({'tpr':tprLst, 'fpr':fprLst})


def if_roc_plot(yTest = None, score = None, plot = True,
                title = ' ', n_pts = 51, extended = False):
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


# Append TF-IDF
def TFIDF(EPIC_CUI, EPIC_enc):
    # Find all Sepsis
    ifSepsis = EPIC_enc['Primary.Dx'] == 1
    CUISepsis = EPIC_CUI.iloc[ifSepsis.values]
    # Get all unique CUIs
    triageNotes = {}
    for i in CUISepsis.index:
        cuiLst = [cui for cui in CUISepsis.loc[i, 'Triage.Notes']]
        for cui in cuiLst:
            if cui not in triageNotes.keys():
                triageNotes[cui] = 0
    # For each unique CUI, count the number of documents that contains it
    for notes in EPIC_CUI['Triage.Notes']:
        for cui in triageNotes.keys():
            if cui in notes:
                triageNotes[cui] += 1
    # Create TF-IDF dataframe
    triageDf = pd.DataFrame(index = range(len(EPIC_CUI)), columns = range(len(triageNotes)), dtype = 'float')
    triageDf.iloc[:, :] = 0
    triageDf.columns = triageNotes.keys()
    triageDf.index = EPIC_enc.index
    # Compute TF and IDF
    # Vectorize this!
    print('Start computing TF-IDF ...')
    corpusLen = len(EPIC_CUI)
    for i in triageDf.index:
        notes = EPIC_CUI.loc[i, 'Triage.Notes']
        for cui in notes:
            # Compute TF-IDF if cui is in vocab
            if cui in triageNotes.keys():
                # TF 
                tf = sum([term == cui for term in notes]) / len(notes)
                # IDF 
                idf = np.log( corpusLen / triageNotes[cui] )
                # Store TF-IDF
                triageDf.loc[i, cui] = tf * idf
    # Append to EPIC_enc
    print('Complete')
    cuiCols = triageDf.columns
    EPIC_enc = pd.concat([EPIC_enc, triageDf], axis = 1, sort = False)
    return EPIC_enc, cuiCols


# Prediction with VAE
def vaePredict(loss_train = None, loss_test = None, batch_size = None, sample_size = 1000, k = 1, percent = 0.1):
    '''
    Make prediction based on the train loss and the test loss.
    Threshold is set to be mu + k * std, where mu and std are computed
    from the last sample_size batches.
    Input : loss_train = loss of the train set (np.array or pd.Series)
            loss_test = loss of the test set (np.array or pd.Series)
            batch_size = batch size used in the training
            sample_size = size of the sample used to set the threshold
            k = parameter for setting the threshold
    '''
    # Set threshold
    testLossSorted = np.sort(loss_test)
    threshold = testLossSorted[-int( np.ceil( percent * len(loss_test) ) )]
    yPred = loss_test > threshold
    return(yPred, threshold)


# Train-test split by arrival date
def time_split(data, threshold = 201903, dynamic = False, pred_span = 1):
    '''
    Sort data by the feature 'Arrived' and output train and test sets
    as specified by threshold.
    Input : data = EPIC dataset with feature 'Arrived'
            threshold = time of the train/test split
    Output: XTrain, XTest, yTrain, yTest
    '''
    # Sort by arrival date
    data = data.sort_values(by = ['Arrived'])
    # Split by threshold
    train = data.loc[data['Arrived'] <= threshold]
    if not dynamic:
        test = data.loc[data['Arrived'] > threshold]
    else:
        # Make ending month to the format XX
        end_threshold = str( int(str(threshold)[-2:]) % 12 + pred_span ).zfill(2)
        end_threshold = int( str(threshold)[:-2] + end_threshold )
        selector = data['Arrived'] == end_threshold
        if selector.sum() == 0:
            raise ValueError('No data between {} and {}'.format(threshold, end_threshold))
        test = data.loc[selector]
    yTrain = train['Primary.Dx']
    XTrain = train.drop(['Primary.Dx'], axis = 1)
    yTest = test['Primary.Dx']
    XTest = test.drop(['Primary.Dx'], axis = 1)
    # Drop arrival date
    XTrain = XTrain.drop(['Arrived'], axis = 1)
    XTest = XTest.drop(['Arrived'], axis = 1)
    return(XTrain, XTest, yTrain, yTest)

