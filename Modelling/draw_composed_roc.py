from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module import LogisticRegression

# Data paths
DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"
# Path to save figure
FIG_PATH= "../../results/composed_roc/"
# Get time span
preprocessor = EPICPreprocess.Preprocess(DATA_PATH)
_, _, _, EPIC_arrival = preprocessor.streamline()
# Get time span
time_span = EPIC_arrival['Arrived'].unique().tolist()

# Create folder if not exist
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)


# NN paths
FIG_PATH_NN= "../../results/neural_net/"
FIG_ROOT_PATH_NN = FIG_PATH_NN + f"dynamic_750/"

# LR paths
FIG_PATH_LR = "../../results/logistic_regression/"
FIG_ROOT_PATH_LR = FIG_PATH_LR + f"dynamic_c_l1penalty/"

# RF paths
FIG_PATH_RF = "../../results/random_forest/"
FIG_ROOT_PATH_RF = FIG_PATH_RF + f"dynamic_c/"


# Isolation forest paths
FIG_PATH_IF = "../../results/extended_iforest/"
FIG_ROOT_PATH_IF = FIG_PATH_IF + f"dynamic_a_25estimators/"


# NN aggregate
evaluator_nn = Evaluation.Evaluation(1, 1)
aggregate_summary_nn = evaluator_nn.roc_aggregate(FIG_ROOT_PATH_NN, time_span, eps = True)

# LR aggregate ROC
evaluator_lr = Evaluation.Evaluation(1, 1)
aggregate_summary_lr = evaluator_lr.roc_aggregate(FIG_ROOT_PATH_LR, time_span, eps = True)

# RF aggregate
evaluator_rf = Evaluation.Evaluation(1, 1)
aggregate_summary_rf = evaluator_rf.roc_aggregate(FIG_ROOT_PATH_RF, time_span, eps = True)

# IF aggregate
evaluator_if = Evaluation.Evaluation(1, 1)
aggregate_summary_if = evaluator_if.roc_aggregate(FIG_ROOT_PATH_IF, time_span, eps = True)



# Plot composed ROC
roc_auc = [0] * 4
labels = ["NN", "LR", "RF", "IF"]
data_list = [aggregate_summary_nn, aggregate_summary_lr, aggregate_summary_rf]
# all_data = [aggregate_summary_nn, aggregate_summary_lr, aggregate_summary_rf, aggregate_summary_if]
for i, data in enumerate(data_list):
    roc_auc[i] = sk.metrics.auc(data["FPR"], data["TPR"])
    plot_label = labels[i] + " AUC=%0.3f" % roc_auc[i]
    _ = plt.plot(data["FPR"], data["TPR"], label = plot_label)


_ = plt.plot([0, 1], [0, 1.01],'r--')
_ = plt.title('Aggregate Receiver Operating Characteristic', fontsize = 18)
_ = plt.legend(loc = 'lower right', fontsize = 14)
_ = plt.xlim([0, 1])
_ = plt.ylim([0, 1.01])
_ = plt.xticks(fontsize = 14)
_ = plt.yticks(fontsize = 14)
_ = plt.ylabel('True Positive Rate', fontsize = 18)
_ = plt.xlabel('False Positive Rate', fontsize = 18)
plt.savefig(FIG_PATH + "composed_roc", format="eps", dpi=800)
_ = plt.show()
