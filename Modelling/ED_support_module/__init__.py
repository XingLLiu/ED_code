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
import argparse
import logging


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
             save_path = None, eps = False):
    '''
    Plot the roc curve of a given test set and predictions
    Input : yTest = test set (pd.dataframe or series)
            pred = predictions (pd.dataframe or seires)
    Output: ROC plot
    '''
    precision = sk.metrics.precision_score(yTest, pred)
    f1_score = sk.metrics.f1_score(yTest, pred)
    recall = sk.metrics.recall_score(yTest, pred)
    confMat = sk.metrics.confusion_matrix(yTest, pred)
    fpr, tpr, _ = sk.metrics.roc_curve(yTest, pred)
    roc_auc = sk.metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if save_path != None:
        if eps:
            plt.savefig(save_path, format="eps", dpi=800)
        else:
            plt.savefig(save_path)
        plt.close()
    if plot == True:
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
    plt.title('Receiver Operating Characteristic ' + title)
    plt.plot(fprLst, tprLst, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if save_path != None:
        plt.savefig(save_path, format='eps', dpi=1000)
        plt.close()
    if plot == True:
        plt.show()
    return({'TPR':tprLst, 'FPR':fprLst})


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
    return({'TPR':tprLst, 'FPR':fprLst})


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
    triageDf = pd.DataFrame(index = range(len(EPIC_CUI)),
                            columns = range(len(triageNotes)),
                            dtype = 'float')
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
def vaePredict(loss_train = None, loss_test = None, batch_size = None, 
               sample_size = 1000, k = 1, percent = 0.1):
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
def time_split(data, threshold = 201903, dynamic = True, pred_span = 1):
    '''
    Sort data by the feature 'Arrived' and output train and test sets
    as specified by threshold. This can be seen as a special version
    of sk.model_selection.train_test_split.
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
        # Add one year if appropriate
        if end_threshold == '01':
            year = int(str(threshold)[:-2]) + 1
        else:
            year = int(str(threshold)[:-2])
        end_threshold = int( str(year) + end_threshold )
        test = data.loc[data['Arrived'] == end_threshold]
    yTrain = train['Primary.Dx']
    XTrain = train.drop(['Primary.Dx'], axis = 1)
    yTest = test['Primary.Dx']
    XTest = test.drop(['Primary.Dx'], axis = 1)
    # Drop arrival date
    XTrain = XTrain.drop(['Arrived'], axis = 1)
    XTest = XTest.drop(['Arrived'], axis = 1)
    return(XTrain, XTest, yTrain, yTest)


# Save summary results for dynamic training/testing
def dynamic_summary(summary, p_num, n_num):
    '''
    Return a dataframe with the TPR, FPR, TP, FN, FP, TN .
    Input : summary = a dataframe with TPR and FPR (colnames: tpr, fpr, respectively.)
            p_num = number of positives in the test set
            n_num = number of negatives in the test set
    '''
    # Compute no. of TP and FN
    summary['TP'] = (summary['TPR'] * p_num).round().astype('int')
    summary['FN'] = p_num - summary['TP']
    # Compute no. of FP and TN
    summary['FP'] = (summary['FPR'] * n_num).round().astype('int')
    summary['TN'] = n_num - summary['FP']
    return(summary)



# Upstream function for train/test/(valid) split.
def splitter_upstream():
    '''
    Downstream of the train/test/(valid) splitting. For developer's use.
    '''


# Downstream function for train/test/(valid) split.
def splitter_downstream(XTrain, XTest, yTrain, yTest, num_cols, keep_cols, mode,
                 cui_cols=None, valid_size=None, pca_components=None, seed=None):
    '''
    Downstream of the train/test/(valid) splitting. For developer's use.
    '''
    if valid_size != None:
        # Prepare validation set
        XTrain, XValid, yTrain, yValid = sk.model_selection.train_test_split(XTrain, yTrain, test_size=valid_size,
                                        random_state=seed, stratify=yTrain)
    # Separate the numerical features
    if mode in ['c', 'e', 'f']:
        num_cols = num_cols + list(cui_cols)
    # Extract the numerical columns to be transformed
    XTrainNum = XTrain[num_cols]
    XTestNum = XTest[num_cols]
    if valid_size != None:
        XValidNum = XValid[num_cols]
    # PCA on the numerical entries   # 27, 11  # Without PCA: 20, 18
    if mode in ['b', 'd', 'e', 'f']:
        if mode in ['f']:
            if type(pca_components) != float:
                raise ValueError("pca_components is of type {} but must be float for Sparse PCA.".format(type(pca_components)))
            # Sparse PCA 
            pca = sk.decomposition.SparsePCA(int(np.ceil(XTrainNum.shape[1]/2))).fit(XTrainNum)
        else:
            # Usual PCA
            pca = sk.decomposition.PCA(pca_components).fit(XTrainNum)
        # Assign the transformed values back
        XTrainNum = pd.DataFrame( pca.transform( XTrainNum ) )
        XTestNum = pd.DataFrame( pca.transform( XTestNum ) )
        XTrainNum.index = XTrain.index
        XTestNum.index = XTest.index
        if valid_size != None:
            # Transform validation set
            XValidNum = pd.DataFrame( pca.transform( XValidNum ) )
            XValidNum.index = XValidNum.index
    # Assign the transformed values back
    XTrain = pd.concat( [ XTrain[keep_cols], XTrainNum ], axis=1 )
    XTest = pd.concat( [ XTest[keep_cols], XTestNum ], axis=1 )
    if valid_size != None:
        XValid = pd.concat( [ XValid[keep_cols], XValidNum ], axis=1 )
        return XTrain, XTest, XValid, yTrain, yTest, yValid
    else:
        return XTrain, XTest, yTrain, yTest


# Split EPIC_enc into train/test/(valid)
def splitter(EPIC_enc, num_cols, mode, test_size,
             time_threshold=None, EPIC_CUI=None, valid_size=None, pca_components=None, seed=None):
    '''
    Split EPIC_enc into train/test/(valid) with/without PCA/Sparse PCA (see 'mode'). This can be
    used as a substitute of sklearn.model_selection.TrainTestSplit.
    Input : num_cols = [list or pd.Index] names of numerical cols to be transformed.
            cui_cols = [list or pd.Index] names of CUI cols to be transformed if
                       EPIC_CUI is not None.
            valid_size = [float] proportion of train set to be split into valid set. None if
                          no validation is required.
            mode = [str] must be one of the following:
                            a -- No PCA, no TF-IDF
                            b -- PCA, no TF-IDF
                            c -- No PCA, TF-IDF
                            d -- PCA, but not on TF-IDF
                            e -- PCA, TF-IDF
                            f -- Sparse PCA, TF-IDF
    Output: XTrain, XTest, (XValid), yTrain, yTest, (yValid)
    '''
    # Prepare taining set
    if mode not in ['a', 'b']:
        try:
            EPIC_enc, cui_cols = TFIDF(EPIC_CUI, EPIC_enc)
        except:
            raise ValueError("EPIC_CUI must be given when including TF-IDF")
    else:
        cui_cols = None
    # Separate input features and target
    y = EPIC_enc['Primary.Dx']
    X = EPIC_enc.drop('Primary.Dx', axis = 1)
    # Columns that are not transformed
    keep_cols = [col for col in X.columns if col not in num_cols]
    # Prepare train and test sets
    if time_threshold == None:
        if test_size == None:
            raise ValueError("test_size cannot be None for stratified train/test split.")
        XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=test_size,
                                        random_state=seed, stratify=y)
    else:
        if "Arrived" in EPIC_enc.columns:
            XTrain, XTest, yTrain, yTest = time_split(EPIC_enc, threshold=time_threshold, dynamic=True)
            # Remove the time column
            keep_cols.remove("Arrived")
        else:
            raise ValueError("Feature \'Arrived\' must be in EPIC_enc to split by time.")
    return splitter_downstream(XTrain, XTest, yTrain, yTest, num_cols, keep_cols, mode,
                               cui_cols=cui_cols, valid_size=valid_size, pca_components=pca_components, seed=seed)


# Given FPR, find the corresponding predicted response
def threshold_predict(pred_prob, y_data, fpr=0.05):
    '''
    Predict y values by controling the false positive rate.
    Input : pred_prob = [Series] predicted scores. Higher for class 1.
            y_data = [Series] true response vector.
    '''
    # Initialization
    num_fp = int( np.round( len( y_data ) * fpr ) )
    y_data = pd.Series(y_data)
    fprLst, tprLst = [], []
    threshold = np.linspace(0, 1, 501)
    score_sorted = np.argsort(pred_prob)
    # Find threshold
    for i in range(len(threshold)):
        indices_with_preds = score_sorted[ -int( np.ceil( threshold[i] * y_data.shape[0] ) ): ]
        y_pred = y_data * 0
        y_pred.iloc[indices_with_preds] = 1
        # Compute FPR for the current predicted response vector
        current_fpr = false_positive_rate(y_data, y_pred)
        # Stop if the current FPR is just over the desired FPR
        if current_fpr >= fpr and i > 0:
            break
    # Return the predicted response vector
    if i == len(threshold):
        Warning("All thresholds give a FPR lower than the given value. Make sure it is from 0 to 1.")
    return y_pred


# Transform predicted resposne from -1, 1 to 0, 1
def predict_transform(self, x_data):
    '''
    Predict and transform the predicted response to 0 or 1 instead of -1 and 1.
    '''
    # Predicted response
    y_pred = self.predict(x_data)
    # Transform
    y_pred = -0.5 * y_pred + 0.5
    return y_pred.astype(int)




# Helper function for threshold_predict. For developer's use
def false_positive_rate(y_true, y_pred):
    true_negative = ( (y_pred == 0) & (y_true == 0) ).sum()
    negative = len(y_true) -  y_true.sum()
    if negative != 0:
        fpr = 1 - float(true_negative) / negative
    else:
        fpr = 0
        Warning("No positives detected. Filled in by 0 for FPR instead.")
    return fpr


# Compute TPR
def true_positive_rate(y_true, y_pred):
    '''
    Compute TPR.
    Input : y_true = [Series or array] true response vector.
            y_pred = [Series or array] predicted response vector.
    Output: tpr = [float] true positive rate.
    '''
    true_positive = ( (y_pred == 1) & (y_true == 1) ).sum()
    positive = y_true.sum()
    if positive != 0:
        tpr = float(true_positive) / positive
    else:
        tpr = 0
        Warning("No positives detected. Filled in by 0 for FPR instead.")
    return tpr



# def time_time_split(EPIC_enc, num_cols, mode,
#                     EPIC_CUI=None, valid_size=None, pca_components=None, seed=None,
#                     time_threshold=None):
#     '''
#     Split EPIC_enc into train/test/(valid) with/without PCA/Sparse PCA (see 'mode'). This can be
#     used as a substitute of sklearn.model_selection.TrainTestSplit.
#     Input : num_cols = [list or pd.Index] names of numerical cols to be transformed.
#             cui_cols = [list or pd.Index] names of CUI cols to be transformed if
#                        EPIC_CUI is not None.
#             valid_size = [float] proportion of train set to be split into valid set. None if
#                           no validation is required.
#             mode = [str] must be one of the following:
#                             a -- No PCA, no TF-IDF
#                             b -- PCA, no TF-IDF
#                             c -- No PCA, TF-IDF
#                             d -- PCA, but not on TF-IDF
#                             e -- PCA, TF-IDF
#                             f -- Sparse PCA, TF-IDF
#     Output: XTrain, XTest, (XValid), yTrain, yTest, (yValid)
#     '''
#     # Prepare taining set
#     if mode not in ['a', 'b']:
#         try:
#             EPIC_enc, cui_cols = TFIDF(EPIC_CUI, EPIC_enc)
#         except:
#             raise ValueError("EPIC_CUI must be given when including TF-IDF")
#     # Separate input features and target
#     y = EPIC_enc['Primary.Dx']
#     X = EPIC_enc.drop('Primary.Dx', axis = 1)
#     # Prepare train and test sets
#     if "Arrived" in X.columns:
#         XTrain, XTest, yTrain, yTest = time_split(EPIC_enc, threshold=time_threshold, dynamic=True)
#     else:
#         raise ValueError("\'Arrived\' must be included as a feature in input EPIC_enc.")
#     if valid_size != None:
#         # Prepare validation set
#         XTrain, XValid, yTrain, yValid = sk.model_selection.train_test_split(XTrain, yTrain, test_size=valid_size,
#                                         random_state=seed, stratify=yTrain)
#     # Separate the numerical features
#     if mode in ['c', 'e', 'f']:
#         num_cols = num_cols + list(cui_cols)
#     # Extract the numerical columns to be transformed
#     XTrainNum = XTrain[num_cols]
#     XTestNum = XTest[num_cols]
#     if valid_size != None:
#         XValidNum = XValid[num_cols]
#     # PCA on the numerical entries   # 27, 11  # Without PCA: 20, 18
#     if mode in ['b', 'd', 'e', 'f']:
#         if mode in ['f']:
#             if type(pca_components) != float:
#                 raise ValueError("pca_components is of type {} but must be float for Sparse PCA.".format(type(pca_components)))
#             # Sparse PCA 
#             pca = sk.decomposition.SparsePCA(int(np.ceil(XTrainNum.shape[1]/2))).fit(XTrainNum)
#         else:
#             # Usual PCA
#             pca = sk.decomposition.PCA(pca_components).fit(XTrainNum)
#         # Assign the transformed values back
#         XTrainNum = pd.DataFrame( pca.transform( XTrainNum ) )
#         XTestNum = pd.DataFrame( pca.transform( XTestNum ) )
#         XTrainNum.index = XTrain.index
#         XTestNum.index = XTest.index
#         if valid_size != None:
#             # Transform validation set
#             XValidNum = pd.DataFrame( pca.transform( XValidNum ) )
#             XValidNum.index = XValidNum.index
#     # Assign the transformed values back
#     keep_cols = [col for col in X.columns if col not in num_cols]
#     XTrain = pd.concat( [ XTrain[keep_cols], XTrainNum ], axis=1 )
#     XTest = pd.concat( [ XTest[keep_cols], XTestNum ], axis=1 )
#     if valid_size != None:
#         XValid = pd.concat( [ XValid[keep_cols], XValidNum ], axis=1 )
#         return XTrain, XTest, XValid, yTrain, yTest, yValid
#     else:
#         return XTrain, XTest, yTrain, yTest






# # Dynamic prediction ROC
# def dynamic_roc():
# # Create a directory if not exists
# plot_path = '/'.join(os.getcwd().split('/')[:3]) + '/Pictures/neural_net/'
# dynamic_plot_path = plot_path + 'dynamic/'

# # Create subplot
# for i, month in enumerate(timeSpan[1:-2]):
#     csv_name = dynamic_plot_path + f'summary_{month}.csv'
#     summary = pd.read_csv(csv_name)
#     _ = plt.subplot(3, 3, i + 1)
#     # ROC plot
#     tpr = summary['TPR']
#     fpr = summary['FPR']
#     roc_auc = sk.metrics.auc(fpr, tpr)
#     month_pred = timeSpan[i + 3]
#     _ = plt.title(f'ROC {month_pred}')
#     _ = plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#     _ = plt.legend(loc = 'lower right')
#     _ = plt.plot([0, 1], [0, 1],'r--')
#     _ = plt.xlim([0, 1])
#     _ = plt.ylim([0, 1])
#     _ = plt.ylabel('True Positive Rate')
#     _ = plt.xlabel('False Positive Rate')


# plt.tight_layout()
# plt.savefig(dynamic_plot_path + 'aggregate_roc.eps', format='eps', dpi=1000)
# plt.show()


