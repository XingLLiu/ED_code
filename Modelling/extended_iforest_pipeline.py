from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module import ExtendediForest


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
FIG_PATH = "../../results/extended_iforest/"
DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"


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

    # Permutation test
    imp_means, imp_vars = feature_importance_permutation(
                            predict_method = model.predict_proba,
                            X = np.array(XTest),
                            y = np.array(yTest),
                            metric = true_positive_rate,
                            fpr_threshold = FPR_THRESHOLD,
                            num_rounds = 5,
                            seed = RANDOM_SEED)

    # Save feature importance plot
    fi_evaluator = Evaluation.FeatureImportance(imp_means, imp_vars, XTest.columns, MODEL_NAME)
    fi_evaluator.FI_plot(save_path = DYNAMIC_PATH, y_fontsize = 4, eps = True)


    # ========= 2.a.iii. Plot scores =========
    # Predicted response (0, 1)
    y_pred = model.predict(x_data = XTest,
                           outlier_proportion = OUTLIER_PROPORTION,
                           anomaly_scores = pred)

    # Save score plot
    plt.close()
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


    # ========= 2.c. Save predicted results =========
    pred = pd.DataFrame(pred, columns = ["pred_prob"])
    pred.to_csv(DYNAMIC_PATH + f"predicted_result_{time_pred}.csv", index = False)

    
    # ========= End of iteration =========
    print("Completed evaluation for {}.\n".format(time_pred))



# ========= 2.c. Summary plots =========
print("Saving summary plots ...")

SUMMARY_PLOT_PATH = FIG_PATH + "dynamic/"
# Subplots of ROCs
evaluator.roc_subplot(SUMMARY_PLOT_PATH, time_span, [3, 3])
# Aggregate ROC
aggregate_summary = evaluator.roc_aggregate(SUMMARY_PLOT_PATH, time_span, eps = True)
# Save aggregate summary
aggregate_summary.to_csv(SUMMARY_PLOT_PATH + "aggregate_summary.csv", index = False)

print("Summary plots saved at {}".format(SUMMARY_PLOT_PATH))
print("====================================")
