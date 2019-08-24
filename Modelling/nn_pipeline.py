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
CLASS_WEIGHT1 = 300000
CLASS_WEIGHT0 = 100
MODE = "e"
FPR_THRESHOLD = 0.1

NUM_CLASS = 2
NUM_EPOCHS = 25
BATCH_SIZE = 1000
LEARNING_RATE = 1e-4
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
FIG_PATH = "../../results/neural_net/"
DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"
FIG_ROOT_PATH = FIG_PATH + f"dynamic_{NUM_EPOCHS}epochs_{2 * HIDDEN_SIZE}hiddenSize/"


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
    # ========= 2.a. Setup =========
    # Month to be predicted
    time_pred = time_span[j + 3]

    # Create folder if not already exist
    DYNAMIC_PATH = FIG_ROOT_PATH + f"{time_pred}/"
    MONTH_DATA_PATH = DYNAMIC_PATH + "data/"
    for path in [DYNAMIC_PATH, MONTH_DATA_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)


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
                .format( yTrain.shape, yTest.shape, yTrain.sum(), yTest.sum() ) )


    # ========= 2.a.i. Model =========
    # Initialize the model at all iterations
    if j >= 0:
        # Neural net model
        input_size = XTrain.shape[1]
        model = NeuralNet(device = device,
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
    

    # Save data
    XTrain.to_csv(MONTH_DATA_PATH + f"x_train_{time}.csv", index = False)
    yTrain.to_csv(MONTH_DATA_PATH + f"y_train_{time}.csv", index = False, header = True)
    XTest.to_csv(MONTH_DATA_PATH + f"x_test_{time_pred}.csv", index = False)
    yTest.to_csv(MONTH_DATA_PATH + f"y_test_{time_pred}.csv", index = False, header = True)

    # Save model, optimizer and loss function
    model.save_model(save_path = DYNAMIC_PATH + f"model_{time_pred}.ckpt")
    pickle.dump( optimizer, open( DYNAMIC_PATH + f"optimizer_{time_pred}.ckpt", "wb" ) )
    pickle.dump( criterion, open( DYNAMIC_PATH + f"loss_func_{time_pred}.ckpt", "wb" ) )


    # Comput and store the predicted probs for the train set (for stacked model)
    pred_train = model.predict_proba_single(x_data = XTrain,
                                            batch_size = BATCH_SIZE,
                                            transformation = transformation)


    # ========= 2.a.ii. Feature importance by permutation test =========
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
    summary_data.to_csv(DYNAMIC_PATH + f"summary_{time_pred}.csv", index = True)


    # ========= 2.c. Save predicted results =========
    pred = pd.DataFrame(pred, columns = ["pred_prob"])
    pred.to_csv(DYNAMIC_PATH + f"predicted_result_{time_pred}.csv", index = True)

    # Save probs for train set (for stacked model)
    pred_train = pd.DataFrame(pred_train, columns = ["pred_prob"])
    pred_train.to_csv(DYNAMIC_PATH + f"predicted_result_train_{time_pred}.csv", index = True)   


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
