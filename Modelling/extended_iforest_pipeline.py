from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation


# ----------------------------------------------------
# Supporting functions and classes

def predict_proba(self, x_data):
    '''
    Predict the outlier score of x_data.
    Input : x_data = [DataFrame or array] x test set
    '''
    try:
        # Get values if x_data is DataFrame
        x_data = x_data.values
    except:
        pass
    anomaly_scores = self.compute_paths(X_in = x_data)
    return anomaly_scores


def predict(self, x_data, outlier_proportion, anomaly_scores=None):
    '''
    Predict the response variable using x_data.
    Input : x_data = [DataFrame] design matrix. Omitted if anomaly_score
                                 is not None.
            outlier_proportion = [float] proportion of outliers required.
                                         Must be between 0 and 1.
            anomaly_scroes = [Series or array] anomaly scores of the
                             instances. The higher the score, the more
                             likely it is an outlier. If None, predict
                             by using x_data first.
    Output: y_pred = [Series] predicted response vector. 1 for outlier.
    '''
    if not isinstance(x_data, pd.DataFrame):
        raise TypeError("Type of x_data must be DataFrame but got {} instead."
                        .format(type(x_data)))
    if anomaly_scores is None:
        anomaly_scores = self.predict_proba(x_data)
    # sort the scores
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    # retrieve indices of anomalous observations
    outlier_num = int( np.ceil( outlier_proportion * x_data.shape[0] ) )
    indices_with_preds = anomaly_scores_sorted[ -outlier_num : ]
    # create predictions
    y_pred = x_data.iloc[:, 0] * 0
    y_pred.iloc[indices_with_preds] = 1
    return y_pred


def plot_scores(self, anomaly_scores, y_true, y_pred, save_path=None, title=None, eps=False):
    '''
    Plot the anomaly scores.
    Input : anomaly_scores = [Series or array] anomaly scores.
            y_true = [Series or array] response vector.
            title = title of the plot
    '''
    # Convert to numpy array
    anomaly_scores = np.array(anomaly_scores)
    # Plot scores
    x_vec = np.linspace(1, len(yTest), len(yTest))
    # True positives
    true_positive = (y_true == 1) & (y_pred == 1)
    _ = sns.scatterplot(x = x_vec, y = anomaly_scores)
    _ = sns.scatterplot(x = x_vec[y_true == 1],
                        y = anomaly_scores[yTest == 1],
                        label = "false negatives")
    _ = sns.scatterplot(x = x_vec[y_pred == 1],
                        y = anomaly_scores[y_pred == 1],
                        label = "false positives")
    _ = sns.scatterplot(x = x_vec[true_positive == 1],
                        y = anomaly_scores[true_positive == 1],
                        label = "true positives")
    _ = plt.title(title)
    _ = plt.legend()
    if save_path is not None:
        if eps:
            plt.savefig(save_path + "scores.eps", format="eps", dpi=800)
        else:
            plt.savefig(save_path + "scores.png")
        plt.close()
    else:
        plt.show()




eif.iForest.predict_proba = predict_proba
eif.iForest.predict = predict
eif.iForest.plot_scores = plot_scores


# For feature importance evaluation
def add_method(y_true, fpr):
    '''
    Add method to RandomForestClassifier for evaluating feature importance.
    Evaluation metric would be the TPR corresponding to the given FPR.
    Input : y_true = [list or Series] true response values.
            fpr = [float] threshold false positive rate.
    '''
    def threshold_predict_method(self, x_data, y_true=y_true, fpr=fpr):
        # Predicted probability
        pred_prob = self.predict_proba(x_data)
        # Predicted response vector
        y_pred = threshold_predict(pred_prob, y_true, fpr)
        return y_pred
    
    eif.iForest.threshold_predict = threshold_predict_method


# ----------------------------------------------------
# ========= 0. Preliminary seetings =========
MODEL_NAME = "Extended iForest"
RANDOM_SEED = 27
MODE = "a"
FPR_THRESHOLD = 0.1

N_ESTIMATORS = 25
SAMPLE_SIZE = 128
EXTENSION_LEVEL = 0
OUTLIER_PROPORTION = 0.2


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


# Path to save figures
FIG_PATH = "/".join(os.getcwd().split("/")[:3]) + "/Pictures/extended_iforest/"
DATA_PATH = "/home/xingliu/Documents/ED/data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"


# Create folder if not already exist
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)


# ----------------------------------------------------
# ========= 1. Further preprocessing =========
preprocessor = EPICPreprocess.Preprocess(DATA_PATH)
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
    DYNAMIC_PATH = FIG_PATH + "dynamic/" + f"{time_pred}/"
    if not os.path.exists(DYNAMIC_PATH):
        os.makedirs(DYNAMIC_PATH)

    # Prepare train/test sets
    XTrain, XTest, yTrain, yTest= splitter(EPIC_arrival,
                                            num_cols,
                                            MODE,
                                            time_threshold = time,
                                            test_size = None,
                                            EPIC_CUI = EPIC_CUI,
                                            seed = RANDOM_SEED)

    print("Training for data up to {} ...".format(time))
    print( "Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}]."
                .format( len(yTrain), len(yTest), yTrain.sum(), yTest.sum() ) )

    # ========= 2.a.i. Model =========
    # Fit model
    model = eif.iForest(XTrain.loc[yTrain == 0, :].values,
                        ntrees = N_ESTIMATORS,
                        sample_size = SAMPLE_SIZE,
                        ExtensionLevel = EXTENSION_LEVEL)

    # Prediction
    pred = model.predict_proba(XTest)


    # ========= 2.a.ii. Feature importance by permutation test =========
    # # Add method for feature importance evaluation
    # add_method(y_true = yTest, fpr = FPR_THRESHOLD)

    # # Permutation test
    # imp_means, imp_vars = mlxtend.evaluate.feature_importance_permutation(
    #                         predict_method = model.threshold_predict,
    #                         X = np.array(XTest),
    #                         y = np.array(yTest),
    #                         metric = true_positive_rate,
    #                         num_rounds = 5,
    #                         seed = RANDOM_SEED)

    # # Save feature importance plot
    # fi_evaluator = Evaluation.FeatureImportance(imp_means, imp_vars, XTest.columns, MODEL_NAME)
    # fi_evaluator.FI_plot(save_path = DYNAMIC_PATH, y_fontsize = 4, eps = True)


    # ========= 2.a.iii. Plot scores =========
    # Predicted response
    y_pred = model.predict(x_data = XTest,
                           outlier_proportion = OUTLIER_PROPORTION,
                           anomaly_scores = pred)
    # Save score plot
    model.plot_scores(pred, yTest, y_pred,
                      save_path = DYNAMIC_PATH + f"roc_{time_pred}",
                      title= MODEL_NAME,
                      eps=False)


    # ========= 2.b. Evaluation =========
    evaluator = Evaluation.Evaluation(y_true = yTest, pred_prob = pred)

    # Save ROC plot
    _ = evaluator.roc_plot(plot = False, title = MODEL_NAME, save_path = DYNAMIC_PATH + f"roc_{time_pred}")

    # Save summary
    summary_data = evaluator.summary()
    summary_data.to_csv(DYNAMIC_PATH + f"summary_{time_pred}.csv", index = False)

    # ========= End of iteration =========
    print("Completed evaluation for {}.\n".format(time_pred))



# ========= 2.c. Summary plots =========
print("Saving summary plots ...")

SUMMARY_PLOT_PATH = FIG_PATH + "dynamic/"
# Subplots of ROCs
evaluator.roc_subplot(SUMMARY_PLOT_PATH, time_span, [3, 3])
# Aggregate ROC
aggregate_summary = evaluator.roc_aggregate(SUMMARY_PLOT_PATH, time_span)
# Save aggregate summary
aggregate_summary.to_csv(SUMMARY_PLOT_PATH + "aggregate_summary.csv", index = False)

print("Summary plots saved at {}".format(SUMMARY_PLOT_PATH))
print("====================================")
