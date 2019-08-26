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
FIG_ROOT_PATH_NN = FIG_PATH_NN + f"dynamic_1000/"
FIG_ROOT_PATH_NN_LST = [FIG_PATH_NN + f"dynamic_a_seeds10_750epochs_1600hiddenSize/",
                        FIG_PATH_NN + f"dynamic_a_seeds20_750epochs_1600hiddenSize/",
                        FIG_PATH_NN + f"dynamic_a_seeds30_750epochs_1600hiddenSize/",
                        FIG_PATH_NN + f"dynamic_a_seeds40_750epochs_1600hiddenSize/",
                        FIG_PATH_NN + f"dynamic_a_seeds50_750epochs_1600hiddenSize/",
                        FIG_PATH_NN + f"dynamic_a_seeds60_750epochs_1600hiddenSize/"]


# NN TFIDF paths
FIG_PATH_TFIDF = "../../results/neural_net/"
FIG_ROOT_PATH_TFIDF = FIG_PATH_TFIDF + f"dynamic_e_200epochs_3200hiddenSize/"

# NN with bert paths
FIG_PATH_STACKED = "../../results/stacked/"
FIG_ROOT_PATH_STACKED = FIG_PATH_STACKED + f"dynamic_400epochs_50hiddenSize/"


# NN aggregate
evaluator_nn = Evaluation.Evaluation(1, 1)
aggregate_summary_nn = evaluator_nn.roc_aggregate(FIG_ROOT_PATH_NN, time_span, eps = True)

# NN TFIDF paths
evaluator_lr = Evaluation.Evaluation(1, 1)
aggregate_summary_tfidf = evaluator_lr.roc_aggregate(FIG_ROOT_PATH_TFIDF, time_span, eps = True)

# NN with bert paths
evaluator_rf = Evaluation.Evaluation(1, 1)
aggregate_summary_bert = evaluator_rf.roc_aggregate(FIG_ROOT_PATH_STACKED, time_span, eps = True)



# Plot composed ROC
roc_auc = [0] * 4
labels = ["NN", "NN+TFIDF", "NN+BERT"]
data_list = [aggregate_summary_nn, aggregate_summary_tfidf, aggregate_summary_bert]
for i, data in enumerate(data_list):
    roc_auc[i] = sk.metrics.auc(data["FPR"], data["TPR"])
    plot_label = labels[i] + " AUC=%0.3f" % roc_auc[i]
    _ = plt.plot(data["FPR"], data["TPR"], label = plot_label)


for i, data in enumerate(data_list):
    roc_auc[i] = sk.metrics.auc(data.iloc[:51, 1], data.iloc[:51, 0])
    plot_label = labels[i] + " AUC=%0.3f" % roc_auc[i]
    ax = sns.lineplot(data["FPR"], data["TPR"], label = plot_label, err_style = "band")



_ = plt.plot([0, 1], [0, 1.01],'r--')
_ = plt.title('Aggregate Receiver Operating Characteristic', fontsize = 18)
_ = plt.legend(loc = 'lower right', fontsize = 14)
_ = plt.xlim([0, 1])
_ = plt.ylim([0, 1.01])
_ = plt.xticks(fontsize = 14)
_ = plt.yticks(fontsize = 14)
_ = plt.ylabel('True Positive Rate', fontsize = 18)
_ = plt.xlabel('False Positive Rate', fontsize = 18)
plt.savefig(FIG_PATH + "composed_roc_text", format="eps", dpi=800)
_ = plt.show()
