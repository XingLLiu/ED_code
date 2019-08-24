from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module.NeuralNet import NeuralNet

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
NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
# SAMPLE_WEIGHT = 15
DROP_PROB = 0.4
HIDDEN_SIZE = 500





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


# for j, time in enumerate(time_span[2:-1]):

j = 0
time = time_span[j + 2]

# ========= 2.a. Setup =========
# Month to be predicted
time_pred = time_span[j + 3]

# Create folder if not already exist
DYNAMIC_PATH = FIG_PATH + "dynamic/" + f"{time_pred}/"
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
print("Training for data up to {} ...".format(time))
print( "Train size: {}. Test size: {}. Validation size: {}. Sepsis cases in [train, test, valid]: [{}, {}, {}]."
            .format( yTrain.shape, yTest.shape, len(yValid), yTrain.sum(), yTest.sum(), yValid.sum() ) )



# ========= 2.a.i. Model =========
# Initialize the model at the first iteration

# Neural net model
input_size = XTrain.shape[1]
model = NeuralNet(device = device,
                    input_size = input_size,
                    drop_prob = DROP_PROB,
                    hidden_size = HIDDEN_SIZE).to(device)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, CLASS_WEIGHT])).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# Initialize loss vector
loss_vec = np.zeros(NUM_EPOCHS)
loss_vec_valid = np.zeros(NUM_EPOCHS)

# Construct data loaders
train_loader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTrain, yTrain], axis = 1)),
                                            batch_size = BATCH_SIZE,
                                            shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = np.array(XTest),
                                            batch_size = len(yTest),
                                            shuffle = False)
valid_loader = torch.utils.data.DataLoader(dataset = np.array(XValid),
                                            batch_size = len(yValid),
                                            shuffle = False)
# Otherwise only update the model on data from the previous month
# else:
# train_loader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTrainOld, yTrainOld], axis = 1)),
#                                             batch_size = BATCH_SIZE,
#                                             shuffle = True)
# test_loader = torch.utils.data.DataLoader(dataset = np.array(XTest),
#                                             batch_size = len(yTest),
#                                             shuffle = False)

# Train the model
transformation = nn.Sigmoid()
for epoch in trange(NUM_EPOCHS):
    loss = model.train_model(train_loader,
                            criterion = criterion,
                            optimizer = optimizer)
    # Validation
    output_valid = model.eval_model(test_loader = valid_loader,
                        transformation = transformation)
    y_pred = threshold_predict(pred_prob = output_valid[:, 1], y_data = yValid, fpr = FPR_THRESHOLD)
    with torch.no_grad():
        y_pred = np.array(y_pred)
        yValid = np.array(yValid)
        loss_valid = criterion(torch.Tensor(output_valid), torch.Tensor(yValid).long())
    # Save loss
    loss_vec[epoch] = loss.item()
    loss_vec_valid[epoch] = loss_valid.item()

# Plot and save losses
_ = sns.scatterplot(range(len(loss_vec)), loss_vec, label = "train")
_ = sns.scatterplot(range(len(loss_vec)), loss_vec_valid, label = "valid")
plt.savefig(DYNAMIC_PATH + f"losses_{time_pred}.png")
plt.close()



    # Prediction on train
    # test_loader = torch.utils.data.DataLoader(dataset = np.array(XTrain),
    #                                             batch_size = len(yTrain),
    #                                             shuffle = False)

    transformation = nn.Sigmoid()
    pred = model.eval_model(test_loader = test_loader,
                            transformation = transformation)[:, 1]

    evaluator = Evaluation.Evaluation(yTrain, pred)

    # Save ROC plot
    _ = evaluator.roc_plot(plot = False, title = MODEL_NAME, save_path = DYNAMIC_PATH + f"roc_{time_pred}")
    


    # Save data of this month as train set for the next iteration
    XTrainOld = XTest
    yTrainOld = yTest

    # # Save model
    # model_to_save = model.module if hasattr(model, "module") else model
    # torch.save(model_to_save.state_dict(), DYNAMIC_PATH + f"model_{time_pred}.pkl")


#     # ========= 2.a.ii. Feature importance by permutation test =========
#     # Permutation test
#     imp_means, imp_vars = feature_importance_permutation(
#                             predict_method = model.predict_proba_single,
#                             X = np.array(XTest),
#                             y = np.array(yTest),
#                             metric = true_positive_rate,
#                             fpr_threshold = FPR_THRESHOLD,
#                             num_rounds = 5,
#                             seed = RANDOM_SEED)
#     # Save feature importance plot
#     fi_evaluator = Evaluation.FeatureImportance(imp_means, imp_vars, XTest.columns, MODEL_NAME)
#     fi_evaluator.FI_plot(save_path = DYNAMIC_PATH, y_fontsize = 4, eps = True)


#     # ========= 2.b. Evaluation =========
#     evaluator = Evaluation.Evaluation(yTest, pred)

#     # Save ROC plot
#     _ = evaluator.roc_plot(plot = False, title = MODEL_NAME, save_path = DYNAMIC_PATH + f"roc_{time_pred}")

#     # Save summary
#     summary_data = evaluator.summary()
#     summary_data.to_csv(DYNAMIC_PATH + f"summary_{time_pred}.csv", index = False)


#     # ========= 2.c. Save predicted results =========
#     pred = pd.DataFrame(pred, columns = ["pred_prob"])
#     pred.to_csv(DYNAMIC_PATH + f"predicted_result_{time_pred}.csv", index = False)


#     # ========= End of iteration =========
#     print("Completed evaluation for {}.\n".format(time_pred))



# # ========= 2.c. Summary plots =========
# print("Saving summary plots ...")

# SUMMARY_PLOT_PATH = FIG_PATH + "dynamic/"
# # Subplots of ROCs
# evaluator.roc_subplot(SUMMARY_PLOT_PATH, time_span, dim = [3, 3], eps = True)
# # Aggregate ROC
# aggregate_summary = evaluator.roc_aggregate(SUMMARY_PLOT_PATH, time_span)
# # Save aggregate summary
# aggregate_summary.to_csv(SUMMARY_PLOT_PATH + "aggregate_summary.csv", index = False)

# print("Summary plots saved at {}".format(SUMMARY_PLOT_PATH))
# print("====================================")
