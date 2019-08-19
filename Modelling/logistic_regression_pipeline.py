from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation

# ----------------------------------------------------
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





# ----------------------------------------------------
# ========= 0. Preliminary seetings =========
RANDOM_SEED = 27

# # Set logger
# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# console.setFormatter(formatter)
# logging.getLogger("").addHandler(console)
# logger = logging.getLogger("Activity")


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
# parser = setup_parser()
# args = parser.parse_args()



# ----------------------------------------------------
# Path to save figures
fig_path = "/".join(os.getcwd().split("/")[:3]) + "/Pictures/logistic_regression/"
data_path = "/home/xingliu/Documents/ED/data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"


# Create folder if not already exist
if not os.path.exists(fig_path):
    os.makedirs(fig_path)


# ----------------------------------------------------
# ========= 1. Further preprocessing =========
preprocessor = EPICPreprocess.Preprocess(data_path)
EPIC, EPIC_enc, EPIC_CUI, EPIC_arrival = preprocessor.streamline()

# Get numerical columns (for later transformation)
num_cols = preprocessor.which_numerical(EPIC)
num_cols.remove("Primary.Dx")
num_cols.remove("Will.Return")

# Get time span
time_span = EPIC_arrival['Arrived'].unique().tolist()


# ----------------------------------------------------
# ========= 2.a. One-month ahead prediction =========
print("====================================")
print("Dynamically evaluate the model ...\n")


for j, time in enumerate(time_span[2:-1]):
    # Month to be predicted
    time_pred = time_span[j + 3]
    # Create folder if not already exist
    dynamic_path = fig_path + "dynamic/" + f"{time_pred}/"
    if not os.path.exists(dynamic_path):
        os.makedirs(dynamic_path)
    # Prepare train/test sets
    XTrain, XTest, yTrain, yTest= splitter(EPIC_arrival, num_cols, "a", time_threshold=time, test_size=None,
                                        EPIC_CUI=EPIC_CUI, seed=RANDOM_SEED)
    print("Training for data up to {} ...".format(time))
    print( "Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}]."
                .format( len(yTrain), len(yTest), yTrain.sum(), yTest.sum() ) )
            

    # ========= 2.a.i. Model =========
    # Apply SMOTE
    smote = SMOTE(random_state = RANDOM_SEED, sampling_strategy = 'auto')
    col_names = XTrain.columns
    XTrain, yTrain = smote.fit_sample(XTrain, yTrain)
    XTrain = pd.DataFrame(XTrain, columns=col_names)

    # Fit logistic regression
    model = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l2',
                                        max_iter = 1000).fit(XTrain, yTrain)

    # Re-fit after removing features of zero coefficients
    model_new = sk.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l2', max_iter = 1000)
    double_logistic_regressor = DoubleLogisticRegression(model, XTrain, yTrain)
    model_new = double_logistic_regressor.double_fits(model_new, XTrain, yTrain)

    # Remove features in test set
    XTest = double_logistic_regressor.remove_zero_coeffs(XTest)
    pred_new = model_new.predict_proba(XTest)[:, 1]


    # ========= 2.a.ii. Plot beta values =========
    # Plot the features whose coefficients are the top 50 largest in magnitude
    non_zero_coeffs = model_new.coef_[model_new.coef_ != 0]
    indices = np.argsort(abs(non_zero_coeffs))[::-1][:50]
    _ = plt.figure()
    _ = plt.title("Logistic Regression Coefficients Values")
    _ = sns.barplot(y = XTest.columns[indices], x = np.squeeze(non_zero_coeffs)[indices])
    _ = plt.yticks(fontsize = 4)
    plt.savefig(dynamic_path + f"coeffs_{time_pred}.eps", format = 'eps', dpi = 800)
    plt.close()
    # ========= 2.b. Evaluation =========
    evaluator = Evaluation.Evaluation(yTest, pred_new)
    # Save ROC plot
    _ = evaluator.roc_plot(plot = False, title = "LR", save_path = dynamic_path + f"roc_{time_pred}")
    # Save summary
    summary_data = evaluator.summary()
    summary_data.to_csv(dynamic_path + f"summary_{time_pred}.csv", index = False)
    # ========= 2.c. Feature importance =========
    # Permutation test
    imp_means, imp_vars = mlxtend.evaluate.feature_importance_permutation(
                    predict_method = model_new.predict,
                    X = np.array(XTest),
                    y = np.array(yTest),
                    metric = sk.metrics.f1_score,
                    num_rounds = 15,
                    seed = RANDOM_SEED)
    fi_evaluator = Evaluation.FeatureImportance(imp_means, imp_vars, XTest.columns, "LR")
    # Save feature importance plot
    fi_evaluator.FI_plot(save_path = dynamic_path, y_fontsize = 4, eps = True)
    # ========= End of iteration =========
    print("Completed evaluation for {}.\n".format(time_pred))



# ========= 2.c. Summary plots =========
print("Saving summary plots ...")

summary_plot_path = fig_path + "dynamic/"
# Subplots of ROCs
evaluator.roc_subplot(summary_plot_path, time_span, [3, 3])
# Aggregate ROC
aggregate_summary = evaluator.roc_aggregate(summary_plot_path, time_span)
# Save aggregate summary
aggregate_summary.to_csv(summary_plot_path + "aggregate_summary.csv", index = False)

print("Summary plots saved at {}".format(summary_plot_path))
print("====================================")
