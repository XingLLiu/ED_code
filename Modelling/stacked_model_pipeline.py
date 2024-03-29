# ----------------------------------------------------
# To run:
# 1. customize hyper-parameters and DATA_PATH in Section 0
# 2. in Terminal:
#       python random_forest_pipeline.py
#
# May run into bugs if nn_pipeline.py and run_bert_pipeline.py
# didn't run properly!
# ----------------------------------------------------
from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module.StackedModel import StackedModel


# ----------------------------------------------------
# ========= 0. Preliminary seetings =========
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = "stacked"
RANDOM_SEED = 27
CLASS_WEIGHT1 = 300000
CLASS_WEIGHT0 = 100
MODE = "a"
FPR_THRESHOLD = 0.1

NUM_CLASS = 2
NUM_EPOCHS = 350
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
DROP_PROB = 0.4
HIDDEN_SIZE = 50

# Parameters of NN (for loading results only)
NUM_EPOCHS_NN = 200
HIDDEN_SIZE_NN = 800 
# Parameters of BERT (for loading results only)
TASK_NAME = "epic_task"



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


# ----------------------------------------------------
# Path set-up
FIG_PATH = "../../results/stacked/"
DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"
FIG_ROOT_PATH = FIG_PATH + f"dynamic_{NUM_EPOCHS}epochs/"

# For loading NN and BERT results
NN_ROOT_PATH =  f"../../results/neural_net/dynamic_{MODE}_{NUM_EPOCHS_NN}epochs_{2 * HIDDEN_SIZE_NN}hiddenSize/"
BERT_ROOT_PATH = f"../../results/bert/dynamic/{TASK_NAME}/"


# Create folder if not already exist
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)


# ----------------------------------------------------
# ========= 1. Read in data =========
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
    # ========= 2.a. Setup =========
    # Month to be predicted
    time_pred = time_span[j + 3]

    # Create folder if not already exist
    DYNAMIC_PATH = FIG_ROOT_PATH + f"{time_pred}/"

    # NN results path
    NN_RESULTS_PATH = NN_ROOT_PATH + f"{time_pred}/"
    # BERT results path
    BERT_RESULTS_PATH = BERT_ROOT_PATH + f"{time_pred}/"

    if not os.path.exists(DYNAMIC_PATH):
        os.makedirs(DYNAMIC_PATH)


    # Prepare train set
    nn_results = pd.read_csv(NN_RESULTS_PATH + f"predicted_result_train_{time_pred}.csv")
    bert_results = pd.read_csv(BERT_RESULTS_PATH + f"predicted_result_train_{time_pred}.csv")

    # Combine data, assuming data is sorted by time
    nn_results = nn_results.iloc[-bert_results.shape[0] : ]
    nn_results.reset_index(drop = True, inplace = True)
    XTrain = pd.concat( [ nn_results.iloc[-bert_results.shape[0] : ], bert_results ], axis = 1 )

    if bert_results.shape[0] != nn_results.shape[0]:
        print( "Warning: Numbers of bert and NN results do not agree in train sets: [{}, {}]. Year: {}"
                        .format( bert_results.shape[0], nn_results.shape[0] , time_pred ) )



    # Prepare test set
    nn_results = pd.read_csv(NN_RESULTS_PATH + f"predicted_result_{time_pred}.csv")
    bert_results = pd.read_csv(BERT_RESULTS_PATH + f"predicted_result_{time_pred}.csv")

    if bert_results.shape[0] != nn_results.shape[0]:
        print( "Warning: Numbers of bert and NN results do not agree in test sets: [{}, {}]. Year: {}"
                        .format( bert_results.shape[0], nn_results.shape[0], time_pred ) )


    # Combine data
    XTest = pd.concat( [ nn_results, bert_results ], axis = 1 )


    # Get labels
    if j == 0:
        _, _, yTrain, _ = splitter(EPIC_arrival,
                                    num_cols,
                                    MODE,
                                    time_threshold = time,
                                    test_size = None,
                                    EPIC_CUI = EPIC_CUI,
                                    seed = RANDOM_SEED)
        _, _, _, yTest = splitter(EPIC_arrival,
                                    num_cols,
                                    MODE,
                                    time_threshold = time,
                                    test_size = None,
                                    EPIC_CUI = EPIC_CUI,
                                    seed = RANDOM_SEED)
    else:
        _, _, _, yTrain = splitter(EPIC_arrival,
                                    num_cols,
                                    MODE,
                                    time_threshold = time_span[j + 1],
                                    test_size = None,
                                    EPIC_CUI = EPIC_CUI,
                                    seed = RANDOM_SEED)
        _, _, _, yTest = splitter(EPIC_arrival,
                                    num_cols,
                                    MODE,
                                    time_threshold = time,
                                    test_size = None,
                                    EPIC_CUI = EPIC_CUI,
                                    seed = RANDOM_SEED)


    # Assign the indices back to train data (useful when concatenating dataframes)
    XTrain.index = yTrain.index
    XTest.index = yTest.index

    print("Training for data up to {} ...".format(time_pred))
    print( "Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}]."
                .format( yTrain.shape, yTest.shape, yTrain.sum(), yTest.sum() ) )


    # ========= 2.a.i. Model =========
    input_size = XTrain.shape[1]
    model = StackedModel(device = device,
                        input_size = input_size,
                        drop_prob = DROP_PROB,
                        hidden_size = HIDDEN_SIZE).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([CLASS_WEIGHT0, CLASS_WEIGHT1])).to(device)
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


    # ========= 2.b. Evaluation =========
    evaluator = Evaluation.Evaluation(yTest, pred)

    # Save ROC plot
    _ = evaluator.roc_plot(plot = False, title = MODEL_NAME, save_path = DYNAMIC_PATH + f"roc_{time_pred}")

    # Save summary
    summary_data = evaluator.summary()
    summary_data.to_csv(DYNAMIC_PATH + f"summary_{time_pred}.csv", index = False)


    # ========= 2.c. Save predicted results =========
    # Store probs for test set
    pred = pd.DataFrame(pred, columns = ["pred_prob"])
    pred.to_csv(DYNAMIC_PATH + f"predicted_result_{time_pred}.csv", index = False)



    # ========= End of iteration =========
    print("Completed evaluation for {}.\n".format(time_pred))



# ========= 2.c. Summary plots =========
print("Saving summary plots ...")

SUMMARY_PLOT_PATH = FIG_ROOT_PATH

# Subplots of ROCs
evaluator.roc_subplot(SUMMARY_PLOT_PATH, time_span, dim = [3, 3], eps = True)
# Aggregate ROC
aggregate_summary = evaluator.roc_aggregate(SUMMARY_PLOT_PATH, time_span)
# Save aggregate summary
aggregate_summary.to_csv(SUMMARY_PLOT_PATH + "aggregate_summary.csv", index = False)

print("Summary plots saved at {}".format(SUMMARY_PLOT_PATH))
print("====================================")
