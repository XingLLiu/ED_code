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

    Input:  
            x_feature: name of feature on the x axis (str)
            y_feature: name of feature on the y axis (str)
            class: multi-class categorical variable used for colouring
            EPIC: the EPIC dataset
    Output: 
            plot object
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
    Plot the roc curve of a given test set and predictions.

    Input : 
            yTest = test set (pd.dataframe or series)
            pred = predictions (pd.dataframe or seires)
    Output: 
            Plot object
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

    Input:  
            yTest = test set (pd.dataframe or series)
            proba = predicted probability (np.array)
    Output: 
            Plot object
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

    Input:  
            yTest = test set (pd.dataframe or series)
            proba = predicted probability (np.array)
    Output: 
            Plot object
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

    Input:  
            results: a dictionary with precision, recall, f1 score and AUC (in that order).
            model_name: name of the model used (str; to be shown in the title).
    Output: 
            plot of these metrics
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

    Input:  
            model: sklearn model
            filename: name of the model file (str)
    '''
    pickle.dump(model, open(filename, 'wb'))


# Load models
def loadModel(filename):
    '''
    Load sklearn model 'filename' in current directory.

    Input:  
            model: sklearn model
            filename: name of the model file (str)
    '''
    return(pickle.load(open(filename, 'rb')))


# Append TF-IDF
def TFIDF(EPIC_CUI, EPIC_enc):
    '''
    Compute and append the TF-IDF, using the CUIs.

    Input :
            EPIC_CUI = [DataFrame] notes in the form of CUIs.
            EPIC_enc = [DataFrame] one-hot encoded design matrix.
    Output:
            EPIC_enc = [DataFrame] EPIC_enc with TF-IDF appended.
            cui_cols = [list] column names of the CUI TF-IDFs.
    '''
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


# Prediction with Variational Auto Encoders
def vaePredict(loss_train = None, loss_test = None, batch_size = None, 
               sample_size = 1000, k = 1, percent = 0.1):
    '''
    Make prediction for Variational Auto Encoders based on the train
    loss and the test loss. Threshold is set to be mu + k * std, where
    mu and std are computed from the last sample_size batches.

    Input : 
            loss_train = loss of the train set (np.array or pd.Series)
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
def time_split(data, threshold = 201903, dynamic = True, pred_span = 1,
                keep_time = False):
    '''
    Sort data by the feature 'Arrived' and output train and test sets
    as specified by threshold. This can be used as a special version
    of sk.model_selection.train_test_split.

    Input : 
            data = EPIC dataset with feature 'Arrived'
            threshold = time of the train/test split
            keep_time = whether to keep 'Arrived' in the returned data
    Output: 
            XTrain, XTest, yTrain, yTest
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
    # Give warning if future data is included in train set
    if any(XTrain['Arrived'] > end_threshold):
        raise Warning('Fture data (after {}) is contained in train set. May be overfitting!'.format(end_threshold))
    # Drop arrival date
    if not keep_time:
        XTrain = XTrain.drop(['Arrived'], axis = 1)
        XTest = XTest.drop(['Arrived'], axis = 1)
    return(XTrain, XTest, yTrain, yTest)


# Save summary results for dynamic training/testing
def dynamic_summary(summary, p_num, n_num):
    '''
    Return a dataframe with the TPR, FPR, TP, FN, FP, TN.

    Input : 
            summary = a dataframe with TPR and FPR (colnames: tpr, fpr, respectively.)
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


# Downstream function for train/test/(valid) split.
def splitter_downstream(XTrain, XTest, yTrain, yTest, num_cols, mode,
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

    # PCA on the numerical entries
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
    cat_cols = [col for col in XTrain.columns if col not in num_cols]
    XTrain = pd.concat( [ XTrain[cat_cols], XTrainNum ], axis=1 )
    XTest = pd.concat( [ XTest[cat_cols], XTestNum ], axis=1 )

    if valid_size != None:
        XValid = pd.concat( [ XValid[cat_cols], XValidNum ], axis=1 )
        return XTrain, XTest, XValid, yTrain, yTest, yValid
    else:
        return XTrain, XTest, yTrain, yTest


# Split EPIC_enc into train/test/(valid)
def splitter(EPIC_enc, num_cols, mode, test_size,
             time_threshold=None, EPIC_CUI=None, valid_size=None, pca_components=None, seed=None,
             keep_time=False):
    '''
    Split EPIC_enc into train/test/(valid) with/without PCA/Sparse PCA (see 'mode'). This can be
    used as a substitute of sklearn.model_selection.TrainTestSplit.

    Input : 
            num_cols = [list or pd.Index] names of numerical cols to be transformed.
            cui_cols = [list or pd.Index] names of CUI cols to be transformed if
                       EPIC_CUI is not None.
            test_size = [float] must be None if time_threshold is give.
            valid_size = [float] proportion of train set to be split into valid set by stratified
                         sampling. None if no validation is required.
            mode = [str] must be one of the following:
                            a -- No PCA, no TF-IDF
                            b -- PCA, no TF-IDF
                            c -- No PCA, TF-IDF
                            d -- PCA, but not on TF-IDF
                            e -- PCA, TF-IDF
                            f -- Sparse PCA, TF-IDF
    Output: 
            XTrain, XTest, (XValid), yTrain, yTest, (yValid)
    '''
    # Split by time
    if time_threshold is not None:
        if "Arrived" in EPIC_enc.columns:
            XTrain, XTest, yTrain, yTest = time_split(EPIC_enc, threshold=time_threshold, dynamic=True,
                                                        keep_time=keep_time)
        else:
            raise ValueError("Feature \'Arrived\' must be in EPIC_enc to split by time.")

    # Add TFIDF
    if mode not in ['a', 'b']:
        try:
            EPIC_enc, cui_cols = TFIDF(EPIC_CUI.loc[ pd.concat( [XTrain, XTest], axis=0 ).index, : ],
                                        EPIC_enc.loc [ pd.concat( [XTrain, XTest], axis=0 ).index, : ] )
        except:
            raise ValueError("EPIC_CUI must be given when including TF-IDF")
    else:
        cui_cols = None

    # Separate input features and target
    y = EPIC_enc['Primary.Dx']
    X = EPIC_enc.drop('Primary.Dx', axis = 1)

    # Prepare train and test sets
    if time_threshold == None:
        if test_size == None:
            raise ValueError("test_size cannot be None for stratified train/test split.")
        XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=test_size,
                                        random_state=seed, stratify=y)
    else:
        if "Arrived" in EPIC_enc.columns:
            XTrain, XTest, yTrain, yTest = time_split(EPIC_enc, threshold=time_threshold, dynamic=True,
                                                        keep_time=keep_time)
        else:
            raise ValueError("Feature \'Arrived\' must be in EPIC_enc to split by time.")

    return splitter_downstream(XTrain, XTest, yTrain, yTest, num_cols, mode,
                               cui_cols=cui_cols, valid_size=valid_size, pca_components=pca_components, seed=seed)


# Given FPR, find the corresponding predicted response
def threshold_predict(pred_prob, y_data, fpr=0.05):
    '''
    Predict y values by controling the false positive rate.

    Input : 
            pred_prob = [Series] predicted scores. Higher for class 1.
            y_data = [Series] true response vector.
            fpr = [float] false positive rate threshold.
    Output:
            predicted response vector that has the specified false positive rate.
    '''
    # Initialization
    if pred_prob.shape != y_data.shape:
        raise Warning("Shapes of predicted probs ({}) do not agree with the true data {}."
                        .format(pred_prob.shape, y_data.shape))

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

    Input :
            x_data = [DataFrame or array] design matrix.
    Output:
            predicted response vector, 0 for class 0 and 1 for class 1.
    '''
    # Predicted response
    y_pred = self.predict(x_data)
    # Transform
    y_pred = -0.5 * y_pred + 0.5
    return y_pred.astype(int)


# Compute FPR. Helper function for threshold_predict.
def false_positive_rate(y_true, y_pred):
    '''
    Compute FPR.

    Input :
            y_true = [DataFrame or array] real response vector.
            y_pred = [DataFrame or array] predicted response vector.
    Output:
            fpr = [float] false positive rate.
    '''
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

    Input : 
            y_true = [Series or array] true response vector.
            y_pred = [Series or array] predicted response vector.
    Output: 
            tpr = [float] true positive rate.
    '''
    true_positive = ( (y_pred == 1) & (y_true == 1) ).sum()
    positive = y_true.sum()
    if positive != 0:
        tpr = float(true_positive) / positive
    else:
        tpr = 0
        Warning("No positives detected. Filled in by 0 for FPR instead.")
    return tpr


# Feature importance function
def feature_importance_permutation(X, y, predict_method,
                                   metric, num_rounds=1, seed=None,
                                   fpr_threshold=0.1):
    """Feature importance imputation via permutation importance
       This function only makes sense if the model is able to output
       probabilities for the predicted responses.
       Adapted from mlxtend.evaluate.feature_importance_permutation
    Parameters
    ----------

    X : NumPy array, shape = [n_samples, n_features]
        Dataset, where n_samples is the number of samples and
        n_features is the number of features.

    y : NumPy array, shape = [n_samples]
        Target values.

    predict_method : prediction function
        A callable function that predicts the target values
        from X.

    metric : str, callable
        The metric for evaluating the feature importance through
        permutation. By default, the strings 'accuracy' is
        recommended for classifiers and the string 'r2' is
        recommended for regressors. Optionally, a custom
        scoring function (e.g., `metric=scoring_func`) that
        accepts two arguments, y_true and y_pred, which have
        similar shape to the `y` array.

    num_rounds : int (default=1)
        Number of rounds the feature columns are permuted to
        compute the permutation importance.

    seed : int or None (default=None)
        Random seed for permuting the feature columns.

    Returns
    ---------

    mean_importance_vals, all_importance_vals : NumPy arrays.
      The first array, mean_importance_vals has shape [n_features, ] and
      contains the importance values for all features.
      The shape of the second array is [n_features, num_rounds] and contains
      the feature importance for each repetition. If num_rounds=1,
      it contains the same values as the first array, mean_importance_vals.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/feature_importance_permutation/
    """
    if not isinstance(num_rounds, int):
        raise ValueError('num_rounds must be an integer.')
    if num_rounds < 1:
        raise ValueError('num_rounds must be greater than 1.')
    if not (metric in ('r2', 'accuracy') or hasattr(metric, '__call__')):
        raise ValueError('metric must be either "r2", "accuracy", '
                         'or a function with signature func(y_true, y_pred).')
    if metric == 'r2':
        def score_func(y_true, y_pred):
            sum_of_squares = np.sum(np.square(y_true - y_pred))
            res_sum_of_squares = np.sum(np.square(y_true - y_true.mean()))
            r2_score = 1. - (sum_of_squares / res_sum_of_squares)
            return r2_score
    elif metric == 'accuracy':
        def score_func(y_true, y_pred):
            return np.mean(y_true == y_pred)
    else:
        score_func = metric
    rng = np.random.RandomState(seed)
    mean_importance_vals = np.zeros(X.shape[1])
    all_importance_vals = np.zeros((X.shape[1], num_rounds))
    pred_prob = predict_method(X)
    y_pred = threshold_predict(pred_prob, y, fpr_threshold)
    baseline = score_func(y, y_pred)
    for round_idx in range(num_rounds):
        for col_idx in range(X.shape[1]):
            save_col = X[:, col_idx].copy()
            new_col = rng.choice(save_col)
            X[:, col_idx] = new_col
            pred_prob = predict_method(X)
            y_pred = threshold_predict(pred_prob, y, fpr_threshold)
            new_score = score_func(y, y_pred)
            X[:, col_idx] = save_col
            importance = baseline - new_score
            mean_importance_vals[col_idx] += importance
            all_importance_vals[col_idx, round_idx] = importance
    mean_importance_vals /= num_rounds
    return mean_importance_vals, all_importance_vals


