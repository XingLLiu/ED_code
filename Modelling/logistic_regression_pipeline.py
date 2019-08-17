from ED_support_module import *
# from EDA import EPIC, EPIC_enc, EPIC_CUI, EPIC_arrival, numCols, catCols




# ----------------------------------------------------
class DoubleLogisticRegression(sk.linear_model.LogisticRegression):
    def __init__(self, X, y, lr):
        '''
        Input : X = [DataFrame] design matrix.
                y = [DataFrame] response.
        '''
        super().__init__()
        self.X = X
        self.y = y
        self.lr = lr
        self.colNames = X.columns
    def which_zero(self):
        '''
        Check which coefficients of the LR model are zero.
        Input : lr = [object] logistic regression model.
        Output: coeffsRm = [list] names of the columns to be reomved
        '''
        # Remove the zero coefficients
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
        Input : model: [object] fitted logistic regression model.
                penalty: [str] penalty to be used for the refitting.
                max_iter: [int] maximum no. of iterations.
        Output: lr_new: [object] refitted logistic regression model.
        '''
        XTrain = self.remove_zero_coeffs(XTrain)
        lr_new = lr_new.fit(XTrain, yTrain)
        return lr_new





# ----------------------------------------------------
# Arguments
def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed",
                        default=27,
                        required=True,
                        type=int,
                        help="Random seed.")
    parser.add_argument("---dynamic",
                        default=True,
                        required=True,
                        type=bool,
                        help="If using one-month ahead prediction.")
    parser.add_argument("--path",
                        required=True,
                        type=str,
                        help="Path to save figures.")
    return parser


# Parser arguements
parser = setup_parser()
args = parser.parse_args()



# ----------------------------------------------------
# Path to save figures
path = '/'.join(os.getcwd().split('/')[:3]) + '/Pictures/logistic_regression/'

# Create folder if not already exist
if not os.path.exists(path):
    os.makedirs(path)


# ----------------------------------------------------




# ----------------------------------------------------
XTrain, XTest, yTrain, yTest = time_split(EPIC_arrival, dynamic=True)

print('Start fitting logistic regression...\n')
lr = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l1',
                                        max_iter = 1000).fit(XTrain, yTrain)
print('Fitting complete\n')
lrPred = lr.predict(XTest)
roc_plot(yTest, lrPred, save_path = path + 'roc1.eps')

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
roc_plot(yTest, lrPred2, save_path = path + 'roc2.eps')







