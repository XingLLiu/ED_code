from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module import LogisticRegression

# ----------------------------------------------------
# ========= 0.i. Supporting functions and classes =========
# NN model
 

# ----------------------------------------------------
# ========= 0.ii. Preliminary seetings =========
# Device configuration
MODEL_NAME = "LR"
RANDOM_SEED = 50
MODE = "a"
FPR_THRESHOLD = 0.1

PENALTY = "l1"   # Penalty of the first fit




# Path set-up
FIG_PATH = "../../results/neural_net_tuning/"
DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"


# Create folder if not already exist
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)


# ----------------------------------------------------
# ========= 1. Further preprocessing =========
drop_cols = ['First.ED.Provider', 'Last.ED.Provider', 'ED.Longest.Attending.ED.Provider',
                'Day.of.Arrival', 'Arrival.Month', 'FSA', 'Name.Of.Walkin', 'Name.Of.Hospital',
                'Admitting.Provider', 'Disch.Date.Time', 'Discharge.Admit.Time',
                'Distance.To.Sick.Kids', 'Distance.To.Walkin', 'Distance.To.Hospital', 'Systolic',
                'Pulse']

drop_cols = []
drop_cols = ['Distance.To.Sick.Kids', 'Distance.To.Walkin', 'Distance.To.Hospital', 'Systolic']
drop_cols = ['Distance.To.Sick.Kids', 'Distance.To.Walkin', 'Distance.To.Hospital', 'Systolic',
            'First.ED.Provider', 'Last.ED.Provider', 'ED.Longest.Attending.ED.Provider']
drop_cols = ['Name.Of.Walkin', 'Day.of.Arrival']



drop_cols = ['First.ED.Provider', 'Last.ED.Provider', 'ED.Longest.Attending.ED.Provider', 'Admitting.Provider',
            'Name.Of.Hospital']
rm_features = ['Systolic', 'Day.of.Arrival_Monday', 'Gender_M']
rm_features = []

preprocessor = EPICPreprocess.Preprocess(DATA_PATH, drop_cols = drop_cols)
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



j = 0
time = time_span[j + 2]

# ========= 2.a. Setup =========
# Month to be predicted
time_pred = time_span[j + 3]

# Create folder if not already exist
DYNAMIC_PATH = FIG_PATH + "dynamic/" + f"{time_pred}_remove{len(rm_features)}features/"
if not os.path.exists(DYNAMIC_PATH):
    os.makedirs(DYNAMIC_PATH)


# Prepare train/test sets
XTrain, XTest, XValid, yTrain, yTest, yValid= splitter(EPIC_arrival,
                                                    num_cols,
                                                    MODE,
                                                    time_threshold = time,
                                                    test_size = None,
                                                    valid_size = 0.15,
                                                    EPIC_CUI = EPIC_CUI,
                                                    seed = RANDOM_SEED)


# Remove features
XTrain = XTrain.drop(rm_features, axis = 1)
XTest = XTest.drop(rm_features, axis = 1)
XValid = XValid.drop(rm_features, axis = 1)

# ========= 2.a.i. Model =========
# Initialize the model at the first iteration

# Model
model = sk.linear_model.LogisticRegression(solver = "liblinear", max_iter = 1000,
                                            class_weight = {0:1, 1:3000}).fit(XTrain, yTrain)
# Predict
pred = model.predict_proba(XValid)[:, 1]

# Evaluate
evaluator = Evaluation.Evaluation(yValid, pred)

# Save ROC plot
_ = evaluator.roc_plot(plot = False, title = MODEL_NAME, save_path = DYNAMIC_PATH + f"roc_{time_pred}")
# Save summary
summary_data = evaluator.summary()
summary_data.to_csv(DYNAMIC_PATH + f"summary_{time_pred}.csv", index = False)

# Store predictions
pred_valid1 = pred


# Permutation test
imp_means, imp_vars = feature_importance_permutation(
                        predict_method = model.predict_proba_single,
                        X = np.array(XTest),
                        y = np.array(yTest),
                        metric = true_positive_rate,
                        fpr_threshold = FPR_THRESHOLD,
                        num_rounds = 15,
                        seed = RANDOM_SEED)

# Save feature importance plot
not_zero = imp_means != 0
cols = XValid.columns[not_zero]
sort_index = np.argsort(abs(imp_means[not_zero]))
imp_means_short = imp_means[not_zero][sort_index <= 30]
imp_vars_short = imp_vars[not_zero][sort_index <= 30]
cols_short = cols[sort_index <= 30]
fi_evaluator = Evaluation.FeatureImportance(imp_means[not_zero], imp_vars[not_zero], cols, MODEL_NAME, show_num = "all")
# fi_evaluator = Evaluation.FeatureImportance(imp_means_short, imp_vars_short, cols_short, MODEL_NAME, show_num = "all")
fi_evaluator.FI_plot(save_path = DYNAMIC_PATH, y_fontsize = 14, eps = True)


# Load data
summary1 = pd.read_csv(FIG_PATH + "dynamic/" + f"{time_pred}_remove{0}features/" + f"summary_{time_pred}.csv")
summary2 = pd.read_csv(FIG_PATH + "dynamic/" + f"{time_pred}_remove{1}features/" + f"summary_{time_pred}.csv")
summary3 = pd.read_csv(FIG_PATH + "dynamic/" + f"{time_pred}_remove{3}features/" + f"summary_{time_pred}.csv")
summary4 = pd.read_csv(FIG_PATH + "dynamic/" + f"{time_pred}_remove{16}features/" + f"summary_{time_pred}.csv")

# Plot composed ROC
roc_auc = [0] * 4
labels = ["Remove 0 feature", "Remove 4 features", "Remove 8 features"]
data_list = [summary1, summary3, summary4]
for i, data in enumerate(data_list):
    roc_auc[i] = sk.metrics.auc(data["FPR"], data["TPR"])
    # Interpolate
    if i == 0:
        plot_label = labels[i] + " TPR=%0.3f" % data.loc[5, "TPR"]
    else:
        plot_label = labels[i] + " TPR=%0.3f" % data.loc[6, "TPR"]
    _ = plt.plot(data["FPR"], data["TPR"], label = plot_label)


# _ = plt.plot([0, 1], [0, 1.01],'r--')
_ = plt.axvline(x = 0.1, color = 'k', linestyle = '--')
_ = plt.title('ROC after Feature Selection (NN)', fontsize = 18)
_ = plt.legend(loc = 'lower right', fontsize = 14)
_ = plt.xlim([0, 1])
_ = plt.ylim([0, 1.01])
_ = plt.xticks(np.arange(0, 1, 0.05), fontsize = 8)
_ = plt.yticks(fontsize = 14)
_ = plt.ylabel('True Positive Rate', fontsize = 18)
_ = plt.xlabel('False Positive Rate', fontsize = 18)
plt.savefig(FIG_PATH + "roc_feature_selection", format="eps", dpi=800)
_ = plt.show()


# Plot TPR
num_rm_features = [0, 1, 2, 16]
tpr = [0] * len(num_rm_features)
for i, num in enumerate(num_rm_features):
    summary = pd.read_csv(FIG_PATH + "dynamic/" + f"{time_pred}_remove{num}features/" + f"summary_{time_pred}.csv")
    tpr[i] = summary.loc[5, "TPR"]



_ = plt.plot(num_rm_features, tpr, linestyle='--', marker='o')
_ = plt.ylim([0, 1])
_ = plt.show()


for i, data in enumerate(data_list):
    plot_label = labels[i] + " AUC=%0.3f" % roc_auc[i]
    _ = plt.plot(data["FPR"], data["TPR"], label = plot_label)
