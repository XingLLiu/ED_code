import sys
sys.path.append("../")
from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module.NeuralNet import NeuralNet


# ----------------------------------------------------
# ========= 0. Preliminary seetings =========
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = "NN"
RANDOM_SEED = 27
CLASS_WEIGHT = 300000
NORMAL_CLASS_WEIGHT = 100
MODE = "a"
FPR_THRESHOLD = 0.1

NUM_CLASS = 2
NUM_EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
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
FIG_PATH = "../../../results/stacked_model/"
DATA_PATH = "../../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"
FIG_ROOT_PATH = FIG_PATH + f"dynamic_{MODE}/"
RAW_SAVE_DIR = FIG_PATH + "Raw_Notes/"



# ----------------------------------------------------
# ========= 1.Read data =========
time_span = pickle.load(open(RAW_SAVE_DIR + "time_span", "rb"))


# ========= 2.a. Setup =========
j = 0
time = time_span[j]
for j, time in enumerate(time_span):
    # Month to be predicted
    time_pred = time_span[j + 3]

    # Create folder if not already exist
    DYNAMIC_PATH = FIG_ROOT_PATH + f"{time_pred}/"
    RESULTS_PATH = FIG_ROOT_PATH + f"{time_pred}/nn_results/"
    NUMERICS_DATA_PATH = DYNAMIC_PATH + "numerical_data/"
    for path in [DYNAMIC_PATH, RESULTS_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)


    # Prepare train/test sets
    XTrain = pd.read_csv(NUMERICS_DATA_PATH + "x_train.csv")
    yTrain = pd.read_csv(NUMERICS_DATA_PATH + "y_train.csv")
    XTest = pd.read_csv(NUMERICS_DATA_PATH + "x_test.csv")
    yTest = pd.read_csv(NUMERICS_DATA_PATH + "y_test.csv")



    # Neural net model
    input_size = XTrain.shape[1]
    model = NeuralNet(device = device,
                        input_size = input_size,
                        drop_prob = DROP_PROB,
                        hidden_size = HIDDEN_SIZE).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, CLASS_WEIGHT])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # Train the model
    model, loss_vec = model.fit(x_data = XTrain,
                                y_data = yTrain,
                                num_epochs = NUM_EPOCHS,
                                batch_size = BATCH_SIZE,
                                optimizer = optimizer,
                                criterion = criterion)

    # Prediction
    transformation = nn.Sigmoid().to(device)
    pred = model.predict_proba_single(x_data = XTest,
                                        batch_size = BATCH_SIZE,
                                        transformation = transformation)

    # Save model, optimize and loss function
    model.save_model(save_path = RESULTS_PATH + f"model_{time_pred}.ckpt")
    pickle.dump( optimizer, open( RESULTS_PATH + f"optimizer_{time_pred}.ckpt", "wb" ) )
    pickle.dump( criterion, open( RESULTS_PATH + f"loss_func_{time_pred}.ckpt", "wb" ) )

    # Save results
    pred = pd.Series(pred, name = "pred_prob")
    pred.to_csv(RESULTS_PATH + "pred_prob_nn.csv", header = True, index = False)


# # ========= 2.b. Evaluation =========
# evaluator = Evaluation.Evaluation(yTest, np.array(pred))

# # Save ROC plot
# _ = evaluator.roc_plot(plot = False, title = MODEL_NAME, save_path = RESULTS_PATH + f"roc_{time_pred}")

# # Save summary
# summary_data = evaluator.summary()
# summary_data.to_csv(RESULTS_PATH + f"summary_{time_pred}.csv", index = False)

# # ========= End of iteration =========
# print("Completed evaluation for {}.\n".format(time_pred))



# # ========= 2.c. Summary plots =========
# print("Saving summary plots ...")

# SUMMARY_PLOT_PATH = FIG_ROOT_PATH

# # Subplots of ROCs
# evaluator.roc_subplot(SUMMARY_PLOT_PATH, time_span, dim = [3, 3], eps = True)
# # Aggregate ROC
# aggregate_summary = evaluator.roc_aggregate(SUMMARY_PLOT_PATH, time_span)
# # Save aggregate summary
# aggregate_summary.to_csv(SUMMARY_PLOT_PATH + "aggregate_summary.csv", index = False)

# print("Summary plots saved at {}".format(SUMMARY_PLOT_PATH))
# print("====================================")
