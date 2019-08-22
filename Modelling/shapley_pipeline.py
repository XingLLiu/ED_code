from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module.NeuralNet import NeuralNet
from ED_support_module import LogisticRegression


# ----------------------------------------------------
# ========= 0.i. Supporting functions and classes =========
# NN model
   

# ----------------------------------------------------
# ========= 0.ii. Preliminary seetings =========
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = "NN"
RANDOM_SEED = 27
CLASS_WEIGHT = 3000
MODE = "a"
FPR_THRESHOLD = 0.1

NUM_CLASS = 2
NUM_EPOCHS = 3000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
# SAMPLE_WEIGHT = 15
DROP_PROB = 0.4
HIDDEN_SIZE = 1000





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
FIG_PATH = "../../results/neural_net/"
DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"
RAW_DATA_PATH = "../../data/EPIC_DATA/EPIC.csv"


# Create folder if not already exist
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)


# ----------------------------------------------------
# ========= 1.i. Further preprocessing =========
preprocessor = EPICPreprocess.Preprocess(DATA_PATH)
EPIC, EPIC_enc, EPIC_CUI, EPIC_arrival = preprocessor.streamline()

# Get numerical columns (for later transformation)
num_cols = preprocessor.which_numerical(EPIC)
num_cols.remove("Primary.Dx")

# Get time span
time_span = EPIC_arrival['Arrived'].unique().tolist()


# ----------------------------------------------------
# ========= 1.ii. Append arrival date =========
EPIC_raw = pd.read_csv(RAW_DATA_PATH, encoding = "ISO-8859-1")
date = pd.to_datetime(EPIC_raw["Arrived"]).loc[EPIC_arrival.index]
# Change name to differentiate from Arrived
date = date.rename("Arrived.Date")
# Append date to EPIC_arrival
EPIC_arrival = pd.concat([EPIC_arrival, date.dt.day], axis = 1)



# ----------------------------------------------------
# ========= 2. Train and test sets for data Shapley =========
j = 4
time = time_span[j]
FPR_THRESHOLD = 0.2


# ========= 2.a. Setup =========
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
                                        seed = RANDOM_SEED,
                                        keep_time = True)
print("Training for data up to {} ...".format(time))
print( "Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}]."
            .format( len(yTrain), len(yTest), yTrain.sum(), yTest.sum() ) )


# Select train/valid sets for data shapley
yValid = yTrain.loc[XTrain["Arrived.Date"] > 21]
XValid = XTrain.loc[XTrain["Arrived.Date"] > 21, :]

yTrain = yTrain.loc[XTrain["Arrived.Date"] <= 21]
XTrain = XTrain.loc[XTrain["Arrived.Date"] <= 21, :]

# Remove date variable
XValid = XValid.drop(["Arrived.Date"], axis = 1)
XTrain = XTrain.drop(["Arrived.Date"], axis = 1)
XTest = XTest.drop(["Arrived.Date"], axis = 1)


# ----------------------------------------------------
# ========= 2.b. One-month ahead prediction =========
# Get data for each group
train_dict = {}
for time in XTrain["Arrived"].unique().tolist():
    x_gp = (XTrain.loc[XTrain["Arrived"] == time, :])
    # Remove month variable
    x_gp = x_gp.drop(["Arrived"], axis = 1)
    # Append response
    y_gp = (yTrain.loc[XTrain["Arrived"] == time])
    train_dict[str(time)] = pd.concat([x_gp, y_gp], axis = 1)


# XTrain1 = (XTrain.loc[XTrain["Arrived"] == time_span[0], :])
# XTrain2 = (XTrain.loc[XTrain["Arrived"] == time_span[1], :])
# XTrain3 = (XTrain.loc[XTrain["Arrived"] == time_span[2], :])

# # Remove month variable
# XTrain1 = XTrain1.drop(["Arrived"], axis = 1)
# XTrain2 = XTrain2.drop(["Arrived"], axis = 1)
# XTrain3 = XTrain3.drop(["Arrived"], axis = 1)

# # Append response


# # Prepare data input into shapley function
# train_dict = {"201807":gp1, "201808":gp2, "201809":gp3}


# Remove month variable from validation set
XValid = XValid.drop(["Arrived"], axis =1)


shapley_val = shapley_exact(model_class = sk.linear_model.LogisticRegression(solver = "liblinear",
                                                                class_weight = {0:1, 1:3000}),
                     train_dict = train_dict,
                     test_data = pd.concat([XValid, yValid], axis = 1),
                     fpr_threshold = FPR_THRESHOLD,
                     convergence_tol = 0.01,
                     performance_tol = 0.01,
                     max_iter = 50,
                     benchmark_score = None)




# shapley_vec = tmc_shapley(model_class = sk.linear_model.LogisticRegression(solver = "liblinear",
#                                                                             class_weight = {0:1, 1:3000}),
#                         train_dict = train_dict,
#                         test_data = pd.concat([XValid, yValid], axis = 1),
#                         fpr_threshold = FPR_THRESHOLD,
#                         convergence_tol = 0.01,
#                         performance_tol = 0.01,
#                         max_iter = 50,
#                         benchmark_score = None)



# ========= 2.b. Refit without the group with the lowest shapley value =========
# Get good samples
data_good = pd.concat([train_dict["201807"],
                        train_dict["201809"],
                        train_dict["201810"],
                        train_dict["201808"]], axis = 0)
yTrain_good = data_good["Primary.Dx"]
XTrain_good = data_good.drop(["Primary.Dx"], axis = 1)


# Fit with whole data
XTrain = XTrain.drop(["Arrived"], axis = 1)
XTest = XTest.drop(["Arrived"], axis = 1)

model_full = sk.linear_model.LogisticRegression(solver = "liblinear",
                                            class_weight = {0:1, 1:3000}).fit(XTrain, yTrain)
pred_prob_full = model_full.predict_proba_single(XValid)
y_pred_full = threshold_predict(pred_prob_full, yValid, fpr = FPR_THRESHOLD)
tpr_full = true_positive_rate(yValid, y_pred_full)

# Fit with subset
model_sub = sk.linear_model.LogisticRegression(solver = "liblinear",
                                            class_weight = {0:1, 1:3000}).fit(XTrain_good, yTrain_good)
pred_prob_sub = model_sub.predict_proba_single(XValid)
y_pred_sub = threshold_predict(pred_prob_sub, yValid, fpr = FPR_THRESHOLD)
tpr_sub = true_positive_rate(yValid, y_pred_sub)


# Test on unseen data
pred_prob_full = model_full.predict_proba_single(XTest)
y_pred_full = threshold_predict(pred_prob_full, yTest, fpr = FPR_THRESHOLD)
tpr_full_test = true_positive_rate(yTest, y_pred_full)


pred_prob_sub = model_sub.predict_proba_single(XTest)
y_pred_sub = threshold_predict(pred_prob_sub, yTest, fpr = FPR_THRESHOLD)
tpr_sub_test = true_positive_rate(yTest, y_pred_sub)


(tpr_full_test != tpr_sub_test).sum()


print("TPR on valid. full: {}. sub: {}".format(tpr_full, tpr_sub))
print("TPR on valid. full: {}. sub: {}".format(tpr_full_test, tpr_sub_test))



evaluator = Evaluation.Evaluation(yTest, pred_prob_full)
# Save ROC plot
_ = evaluator.roc_plot(plot = True, title = MODEL_NAME)

evaluator = Evaluation.Evaluation(yTest, pred_prob_sub)
# Save ROC plot
_ = evaluator.roc_plot(plot = True, title = MODEL_NAME)






# ----------------------------------------------------
# ========= 2.b. Built-in Shapley =========
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from ED_support_module.Shapley import ShapNN
from ED_support_module.DShap import DShap
from ED_support_module.shap_utils import *

problem, model = 'classification', 'logistic'
hidden_units = [] # Empty list in the case of logistic regression.
train_size = 100


d, difficulty = 50, 1
num_classes = 2
tol = 0.03
target_accuracy = 0.7
important_dims = 5
clf = return_model(model, solver='liblinear', hidden_units=tuple(hidden_units))
_param = 1.0
for _ in range(100):
    X_raw = np.random.multivariate_normal(mean=np.zeros(d), cov = np.eye(d), 
                                          size=train_size + 5000)
    _, y_raw, _, _ = label_generator(
        problem, X_raw, param = _param,  difficulty = difficulty, important=important_dims)
    clf.fit(X_raw[:train_size], y_raw[:train_size])
    test_acc = clf.score(X_raw[train_size:], y_raw[train_size:])
    if test_acc>target_accuracy:
        break
    _param *= 1.1


print('Performance using the whole training set = {0:.2f}'.format(test_acc))


# Runing
X, y = X_raw[:train_size], y_raw[:train_size]
X_test, y_test = X_raw[train_size:], y_raw[train_size:]
model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
sources = None
dshap = DShap(X, y, X_test, y_test, num_test, sources=sources, model_family=model, metric='accuracy',
              directory=directory, seed=0)
dshap.run(100, 0.1)

X, y = X_raw[:100], y_raw[:100]
X_test, y_test = X_raw[100:], y_raw[100:]
model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
dshap = DShap(X, y, X_test, y_test, num_test, model_family=model, metric='accuracy',
              directory=directory, seed=1)
dshap.run(100, 0.1)

X, y = X_raw[:100], y_raw[:100]
X_test, y_test = X_raw[100:], y_raw[100:]
model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
dshap = DShap(X, y, X_test, y_test, num_test, model_family=model, metric='accuracy',
              directory=directory, seed=2)
dshap.run(100, 0.1)


# Merge results
dshap.merge_results()

convergence_plots(dshap.marginals_tmc)

convergence_plots(dshap.marginals_g)

dshap.performance_plots([dshap.vals_tmc, dshap.vals_g, dshap.vals_loo], num_plot_markers=20,
                       sources=dshap.sources)