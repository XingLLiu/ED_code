from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module.NeuralNet import NeuralNet
from ED_support_module import LogisticRegression
from scipy.special import comb


# ----------------------------------------------------
# ========= 0.i. Supporting functions and classes =========
def shapley_exact(model_class, train_dict, test_data, fpr_threshold,
                convergence_tol, performance_tol, max_iter, benchmark_score,
                model_name, num_epochs, batch_size, optimizer, criterion, device):
    groups = list( train_dict.keys() )
    power_set = list_powerset(groups)
    power_set.remove([])
    # Initialize shapley
    shapley_vec = pd.DataFrame(0, index = range(1), columns = groups)
    # Separate test data
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    for current_gp in groups:
        print("Computing Shapley for {}".format(current_gp))
        shapley = 0
        for subgp in power_set:
            if current_gp not in subgp:
                input_size = x_test.shape[1]
                DROP_PROB = 0.4
                HIDDEN_SIZE = 500
                BATCH_SIZE = 128
                NUM_EPOCHS = 100
                LEARNING_RATE = 1e-3
                CLASS_WEIGHT = 3000
                model_class = NeuralNet(device = device,
                                        input_size = input_size,
                                        drop_prob = DROP_PROB,
                                        hidden_size = HIDDEN_SIZE).to(device)
                criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, CLASS_WEIGHT])).to(device)
                optimizer = torch.optim.SGD(model_class.parameters(), lr = LEARNING_RATE)
                summand = shapley_summand(model_class,
                                            subgp = subgp,
                                            current_gp = current_gp,
                                            train_dict = train_dict,
                                            x_test = x_test,
                                            y_test = y_test,
                                            fpr_threshold = fpr_threshold,
                                            model_name = model_name,
                                            num_epochs = num_epochs,
                                            batch_size = batch_size,
                                            optimizer = optimizer,
                                            criterion = criterion,
                                            device = device)
                shapley += summand
        shapley_vec[current_gp] = shapley
    return shapley_vec



def shapley_summand(model_class, subgp, current_gp, train_dict, x_test, y_test, fpr_threshold,
                    model_name, num_epochs, batch_size, optimizer, criterion, device):
    '''
    Compute the summand.
    '''
    # Retrieve train data upto pi(j) as in the paper
    train_data = train_dict[subgp[0]]
    if len(subgp) > 1:
        for name in subgp[1:]:
            train_data = pd.concat( [ train_data, train_dict[name] ], axis = 0 )
    # S union i
    train_data_large = pd.concat( [ train_data, train_dict[current_gp] ], axis = 0 )
    # Evaluate metrics
    scores = [0, 0]
    for k, data in enumerate([train_data, train_data_large]):
        # Get design matrix
        x_train = data.iloc[:, :-1]
        # Get labels
        y_train = data.iloc[:, -1]
        # Fit model and evaluate metric
        if model_name == "logistic":
            model = model_class.fit(x_train, y_train)
            pred_prob = model.predict_proba(x_test)[:, 1]
        elif model_name == "nn":
            model, _ = model_class.fit(x_data = x_train,
                                    y_data = y_train,
                                    num_epochs = num_epochs,
                                    batch_size = batch_size,
                                    optimizer = optimizer,
                                    criterion = criterion)
            pred_prob = model.predict_proba_single(x_test)
        # y_pred = pred_prob > 0.5
        # scores[k] = sk.metrics.accuracy_score(y_test, y_pred)
        y_pred = threshold_predict(pred_prob, y_test, fpr_threshold)
        scores[k] = true_positive_rate(y_test, y_pred)
    
    return (scores[1] - scores[0]) / comb( len( train_dict ) - 1, len( subgp ) )



def list_powerset(lst):
    # the power set of the empty set has one element, the empty set
    result = [[]]
    for x in lst:
        result.extend([subset + [x] for subset in result])
    return result
 
   

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






# Path set-up
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
j = 2
time = time_span[j]
FPR_THRESHOLD = 0.1


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
            .format( yTrain.shape, yTest.shape, yTrain.sum(), yTest.sum() ) )


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


# shapley_val = shapley_exact(model_class = sk.linear_model.LogisticRegression(solver = "liblinear", penalty = "l1",
#                                                                 class_weight = {0:1, 1:3000}),
#                      train_dict = train_dict,
#                      test_data = pd.concat([XValid, yValid], axis = 1),
#                      fpr_threshold = FPR_THRESHOLD,
#                      convergence_tol = 0.01,
#                      performance_tol = 0.01,
#                      max_iter = 50,
#                      benchmark_score = None)



# NN model
input_size = XTrain.shape[1]
DROP_PROB = 0.4
HIDDEN_SIZE = 500
BATCH_SIZE = 128
NUM_EPOCHS = 1000


model = NeuralNet(device = device,
                          input_size = input_size,
                          drop_prob = DROP_PROB,
                          hidden_size = HIDDEN_SIZE).to(device)

criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, CLASS_WEIGHT])).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)

shapley_val = shapley_exact(model_class = model,
                     train_dict = train_dict,
                     test_data = pd.concat([XValid, yValid], axis = 1),
                     fpr_threshold = FPR_THRESHOLD,
                     convergence_tol = 0.01,
                     performance_tol = 0.01,
                     max_iter = 50,
                     benchmark_score = None,
                     model_name = "nn",
                     batch_size = BATCH_SIZE,
                     num_epochs = NUM_EPOCHS,
                     optimizer = optimizer,
                     criterion = criterion,
                     device = device)



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
data_good = pd.concat([train_dict["201808"],
                        train_dict["201809"]], axis = 0)
yTrain_good = data_good["Primary.Dx"]
XTrain_good = data_good.drop(["Primary.Dx"], axis = 1)


# Fit with whole data
XTrain = XTrain.drop(["Arrived"], axis = 1)
XTest = XTest.drop(["Arrived"], axis = 1)

# Pred with full data
model_full = NeuralNet(device = device,
                          input_size = XTrain.shape[1],
                          drop_prob = DROP_PROB,
                          hidden_size = HIDDEN_SIZE).to(device)

criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, CLASS_WEIGHT])).to(device)
optimizer = torch.optim.Adam(model_full.parameters(), lr = LEARNING_RATE)

model_full, loss = model_full.fit(XTrain, yTrain, NUM_EPOCHS, BATCH_SIZE, optimizer, criterion)
pred_prob_full = model_full.predict_proba_single(XTest)
y_pred_full = threshold_predict(pred_prob_full, yTest, fpr = FPR_THRESHOLD)
tpr_full = true_positive_rate(yTest, y_pred_full)



# Fit with subset
model_sub = NeuralNet(device = device,
                          input_size = XTrain_good.shape[1],
                          drop_prob = DROP_PROB,
                          hidden_size = HIDDEN_SIZE).to(device)

criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, CLASS_WEIGHT])).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

model_sub, loss = model_sub.fit(XTrain_good, yTrain_good, NUM_EPOCHS, BATCH_SIZE, optimizer, criterion)
pred_prob_sub = model_sub.predict_proba_single(XTest)
y_pred_sub = threshold_predict(pred_prob_sub, yTest, fpr = FPR_THRESHOLD)
tpr_sub = true_positive_rate(yTest, y_pred_sub)

# Summary
print("TPR on valid. full: {}. sub: {}".format(tpr_full, tpr_sub))



evaluator = Evaluation.Evaluation(yTest, pred_prob_full)
# Save ROC plot
_ = evaluator.roc_plot(plot = True, title = MODEL_NAME)

evaluator = Evaluation.Evaluation(yTest, pred_prob_sub)
# Save ROC plot
_ = evaluator.roc_plot(plot = True, title = MODEL_NAME)












# ========= 2.b. Refit without the group with the lowest shapley value =========
# Logistic regression
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








