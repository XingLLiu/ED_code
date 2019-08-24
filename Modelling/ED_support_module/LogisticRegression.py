from ED_support_module import *


class DoubleLogisticRegression(sk.linear_model.LogisticRegression):
    def __init__(self, lr, XTrain, yTrain):
        '''
        Input : X = [DataFrame] design matrix.
                y = [DataFrame] response.
        '''
        super().__init__()
        self.X = XTrain
        self.y = yTrain
        self.lr = lr
        self.colNames = self.X.columns
    def which_zero(self):
        '''
        Check which coefficients of the LR model are zero.
        Input : lr = [object] logistic regression model.
        Output: coeffsRm = [list] names of the columns to be reomved
        '''
        # Find zero coefficients
        ifZero = (self.lr.coef_ == 0).reshape(-1)
        coeffsRm = [ self.colNames[i] for i in range(self.X.shape[1]) if ifZero[i] ]
        return coeffsRm
    def remove_zero_coeffs(self, data):
        '''
        Remove the features whose coefficients are zero.
        '''
        # Get names of columns to be kept
        notZero = (self.lr.coef_ != 0).reshape(-1)
        whichKeep = pd.Series( range( len( notZero ) ) )
        whichKeep = whichKeep.loc[notZero]
        # Remove the features 
        data = data.iloc[:, whichKeep].copy()
        return data
    # def double_fits(self, XTrain, yTrain, penalty, max_iter=None):
    #     '''
    #     Fit the logistic regression with l1 penalty and refit
    #     after removing all features with zero coefficients.
    #     Input : model: [object] fitted logistic regression model.
    #             penalty: [str] penalty to be used for the refitting.
    #             max_iter: [int] maximum no. of iterations.
    #     Output: lr_new: [object] refitted logistic regression model.
    #     '''
    #     lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = penalty,
    #                                             max_iter = max_iter).fit(XTrain, yTrain)
    #     XTrain = self.remove_zero_coeffs(XTrain)
    #     lr_new = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = penalty,
    #                                                 max_iter = max_iter).fit(XTrain, yTrain)
    #     return lr_new
    def double_fits(self, lr_new, XTrain, yTrain):
        '''
        Fit the logistic regression with l1 penalty and refit
        after removing all features with zero coefficients.
        Input : model: [object] instantiated logistic regression model.
                penalty: [str] penalty to be used for the refitting.
                max_iter: [int] maximum no. of iterations.
        Output: lr_new: [object] refitted logistic regression model.
        '''
        XTrain = self.remove_zero_coeffs(XTrain)
        lr_new_fitted = lr_new.fit(XTrain, yTrain)
        return lr_new_fitted



def which_zero(self, col_names):
    '''
    Check which coefficients of the LR model are zero.
    Input : col_names = [list or Series] column names.
    Output: coeffs_rm = [list] names of the columns to be reomved.
    '''
    # Find zero coefficients
    if_zero = (self.coef_ == 0).reshape(-1)
    coeffs_rm = [ col_names[i] for i in range( len( col_names ) ) if if_zero[i] ]
    return coeffs_rm


def remove_zero_coef_(self, x_data):
    '''
    Remove the features whose coefficients are zero.
    Input : x_data = [DataFrame] design matrix used to fit model.
    Output: x_data = [DataFrame] design matrix with the columns of zero
                     coefficients removed.
    '''
    # Get names of columns to be kept
    not_zero = (self.coef_ != 0).reshape(-1)
    which_keep = pd.Series( range( len( not_zero ) ) )
    which_keep = which_keep.loc[not_zero]
    # Remove the features
    x_data = x_data.iloc[:, which_keep].copy()
    return x_data


def double_fits(self, x_train, y_train):
    '''
    Fit the logistic regression with l1 penalty and refit
    after removing all features with zero coefficients.
    Input : model: [object] instantiated logistic regression model.
            penalty: [str] penalty to be used for the refitting.
            max_iter: [int] maximum no. of iterations.
    Output: lr_new: [object] refitted logistic regression model.
    '''
    x_train = self.remove_zero_coef_(x_train)
    return self.fit(x_train, y_train)


def predict_proba_single(self, x):
    '''
    Output the predicted probability of being of class 1
    only, as opposed to 2 columns for being of class 0 and class 1.
    '''
    return self.predict_proba(x)[:, 1]
    


# Add methods
sk.linear_model.LogisticRegression.which_zero = which_zero
sk.linear_model.LogisticRegression.remove_zero_coef_ = remove_zero_coef_
sk.linear_model.LogisticRegression.double_fits = double_fits
sk.linear_model.LogisticRegression.predict_proba_single = predict_proba_single
