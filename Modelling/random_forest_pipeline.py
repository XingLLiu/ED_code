from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
# About 20 mins for each iteration

# ----------------------------------------------------
# Supporting functions and classes
# def add_method(y_true, fpr):
#     '''
#     Add method to RandomForestClassifier for evaluating feature importance.
#     Evaluation metric would be the TPR corresponding to the given FPR.
#     Input : y_true = [list or Series] true response values.
#             fpr = [float] threshold false positive rate.
#     '''
#     def threshold_predict_method(self, x_data, y_true=y_true, fpr=fpr):
#         # Predicted probability
#         pred_prob = self.predict_proba(x_data)[:, 1]
#         # Predicted response vector
#         y_pred = threshold_predict(pred_prob, y_true, fpr)
#         return y_pred
    
#     sk.ensemble.RandomForestClassifier.threshold_predict = threshold_predict_method


def predict_proba_single(self, x):
    '''
    Output the predicted probability of being of class 1
    only, as opposed to 2 columns for being of class 0 and class 1.
    '''
    return self.predict_proba(x)[:, 1]
    

sk.ensemble.RandomForestClassifier.predict_proba_single = predict_proba_single


# ----------------------------------------------------
# ========= 0. Preliminary seetings =========
MODEL_NAME = "RF"
RANDOM_SEED = 27
CLASS_WEIGHT = 500
MODE = "a"
FPR_THRESHOLD = 0.1

N_ESTIMATORS = 4000
MAX_DEPTH = 30
MAX_FEATURES = "auto"



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
FIG_PATH = "/".join(os.getcwd().split("/")[:3]) + "/Pictures/random_forest/"
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
    # Apply SMOTE only if class weight is 1
    if CLASS_WEIGHT == 1:
        smote = SMOTE(random_state = RANDOM_SEED, sampling_strategy = 'auto')
        col_names = XTrain.columns
        XTrain, yTrain = smote.fit_sample(XTrain, yTrain)
        XTrain = pd.DataFrame(XTrain, columns=col_names)

    # Fit model
    model = sk.ensemble.RandomForestClassifier(n_estimators = N_ESTIMATORS,
                                               max_depth = MAX_DEPTH,
                                               max_features = MAX_FEATURES,
                                               class_weight = {0:1, 1:CLASS_WEIGHT}).fit(XTrain, yTrain)

    # Prediction
    pred = model.predict_proba(XTest)[:, 1]

    # ========= 2.a.ii. Feature importances by Gini impurity =========
    # Get importance scores
    importance_vals = model.feature_importances_
    std = np.std( [tree.feature_importances_ for tree in model.estimators_] , axis=0 )
    indices = np.argsort(importance_vals)[::-1]
    _ = plt.figure()
    _ = plt.title("Random Forest Feature Importance (Gini)")
    _ = sns.barplot(y = XTrain.columns[indices], x = importance_vals[indices],
                    xerr = std[indices])
    _ = plt.yticks(fontsize = 4)
    plt.savefig(DYNAMIC_PATH + f"feature_imp_by_gini_{time_pred}.eps", format = 'eps', dpi = 800)
    plt.close()

    # ========= 2.a.iii. Feature importance by permutation test =========
    # # Add method for feature importance evaluation
    # add_method(y_true = yTest, fpr = FPR_THRESHOLD)

    # # Permutation test
    # imp_means, imp_vars = mlxtend.evaluate.feature_importance_permutation(
    #                         predict_method = model.threshold_predict,
    #                         X = np.array(XTest),
    #                         y = np.array(yTest),
    #                         metric = true_positive_rate,
    #                         num_rounds = 15,
    #                         seed = RANDOM_SEED)


    # Permutation test
    imp_means, imp_vars = feature_importance_permutation(
                            predict_method = model.predict_proba_single,
                            X = np.array(XTest),
                            y = np.array(yTest),
                            metric = true_positive_rate,
                            fpr_threshold = FPR_THRESHOLD,
                            num_rounds = 5,
                            seed = RANDOM_SEED)

    # Save feature importance plot
    fi_evaluator = Evaluation.FeatureImportance(imp_means, imp_vars, XTest.columns, MODEL_NAME)
    fi_evaluator.FI_plot(save_path = DYNAMIC_PATH, y_fontsize = 4, eps = True)

    # ========= 2.b. Evaluation =========
    evaluator = Evaluation.Evaluation(yTest, pred)

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
