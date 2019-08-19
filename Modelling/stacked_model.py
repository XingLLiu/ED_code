from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation


# ----------------------------------------------------
# ========= 0.i. Supporting functions and classes =========
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NN model
class Ensemble(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(Ensemble, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.fc(x)
    def train_model(self, train_loader, criterion, optimizer, device):
        '''
        Train and back-propagate the neural network model. Note that
        this is different from the built-in method self.train, which
        sets the model to train mode.

        Model will be set to evaluation mode internally.

        Input : train_loader = [DataLoader] training set. The
                               last column must be the response.
                criterion = [Function] tensor function for evaluatin
                            the loss.
                optimizer = [Function] tensor optimizer function.
                device = [object] cuda or cpu
        Output: loss
        '''
        self.train()
        for i, x in enumerate(train_loader):
            x = x.to(device)
            # Retrieve design matrix and labels
            labels = x[:, -1].long()
            x = x[:, :(-1)].float()
            # Forward pass
            outputs = self(x)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss
    def eval_model(self, test_loader, device, transformation=None):
        '''
        Evaluate the neural network model. Only makes sense if
        test_loader contains all test data. Note that this is
        different from the built-in method self.train, which
        sets the model to train mode.
        
        Model will be set to evaluation mode internally.

        Input : train_loader = [DataLoader] training set. The
                               last column must be the response.
                transformation = [Function] function for evaluatin
                                 transforming the output. If not given,
                                 raw output is return.
        Output: outputs = output from the model (after transformation).
        '''
        model.eval()
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(device)
                # Retrieve design matrix
                x = x[:, :(-1)].float()
                # Prediction
                outputs = model(x)
                if transformation is not None:
                    # Probability of belonging to class 1
                    outputs = transformation(outputs).detach()
                if i == 0:
                    outputs_vec = np.array(outputs[:, 1])
                else:
                    outputs_vec = np.append(outputs_vec,
                                            np.array(outputs[:, 1]),
                                            axis = 0)
        return outputs_vec






# ----------------------------------------------------
# ========= 0.ii. Preliminary seetings =========
MODEL_NAME = "Ensemble"
RANDOM_SEED = 27
CLASS_WEIGHT = 3000
MODE = "a"
FPR_THRESHOLD = 0.1

NUM_CLASS = 2
NUM_EPOCHS = 10000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
# SAMPLE_WEIGHT = 15
DROP_PROB = 0.1
HIDDEN_SIZE = 200





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
FIG_PATH = "/".join(os.getcwd().split("/")[:3]) + "/Pictures/ensemble/"
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
num_cols.remove("Will.Return")

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
    # Construct data loaders
    train_loader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTrain, yTrain], axis = 1)),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTest, yTest], axis = 1)),
                                                batch_size = len(yTest),
                                                shuffle = False)

    # Neural net model
    input_size = XTrain.shape[1]
    model = NeuralNet(input_size = input_size, drop_prob = DROP_PROB, hidden_size = HIDDEN_SIZE).to(device)

    # Loss and optimizer
    # nn.CrossEntropyLoss() computes softmax internally
    criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, CLASS_WEIGHT])).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)

    # Initialize loss vector
    loss_vec = np.zeros(NUM_EPOCHS)

    # Train the model
    for epoch in trange(NUM_EPOCHS):
        loss = model.train_model(train_loader,
                                criterion = criterion,
                                optimizer = optimizer,
                                device = device)
        loss_vec[epoch] = loss.item()


    # Prediction
    transformation = nn.Sigmoid()
    pred = model.eval_model(test_loader = test_loader,
                            device = device,
                            transformation = transformation)


    # ========= 2.a.ii. Feature importance by permutation test =========
    # # Add method for feature importance evaluation
    # add_method(y_true = yTest, fpr = FPR_THRESHOLD, device = device, transformation = transformation)

    # # Permutation test
    # imp_means, imp_vars = mlxtend.evaluate.feature_importance_permutation(
    #                         predict_method = model.threshold_predict,
    #                         X = np.array(XTest),
    #                         y = np.array(yTest),
    #                         metric = true_positive_rate,
    #                         num_rounds = 10,
    #                         seed = RANDOM_SEED)

    # # Permutation test
    # imp_means, imp_vars = feature_importance_permutation(
    #                         predict_method = model.predict_proba_single,
    #                         X = np.array(XTest),
    #                         y = np.array(yTest),
    #                         metric = true_positive_rate,
    #                         fpr_threshold = FPR_THRESHOLD,
    #                         num_rounds = 5,
    #                         seed = RANDOM_SEED)

    # # Save feature importance plot
    # fi_evaluator = Evaluation.FeatureImportance(imp_means, imp_vars, XTest.columns, MODEL_NAME)
    # fi_evaluator.FI_plot(save_path = DYNAMIC_PATH, y_fontsize = 4, eps = True)


    # ========= 2.b. Evaluation =========
    evaluator = Evaluation.Evaluation(yTest, pred)

    # Save ROC plot
    _ = evaluator.roc_plot(plot = False, title = MODEL_NAME, save_path = DYNAMIC_PATH + f"roc_{time_pred}")

    # Save summary
    summary_data = evaluator.summary()
    summary_data.to_csv(DYNAMIC_PATH + f"summary_{time_pred}.csv", index = False)
    quit()


    # ========= 2.c. Save predicted results =========
    pred.to_csv(DYNAMIC_PATH + f"pedicted_result_{time_pred}.csv", index = False)


    # ========= End of iteration =========
    print("Completed evaluation for {}.\n".format(time_pred))



# ========= 2.c. Summary plots =========
print("Saving summary plots ...")

SUMMARY_PLOT_PATH = FIG_PATH + "dynamic/"
# Subplots of ROCs
evaluator.roc_subplot(SUMMARY_PLOT_PATH, time_span, dim = [3, 3], eps = True)
# Aggregate ROC
aggregate_summary = evaluator.roc_aggregate(SUMMARY_PLOT_PATH, time_span)
# Save aggregate summary
aggregate_summary.to_csv(SUMMARY_PLOT_PATH + "aggregate_summary.csv", index = False)

print("Summary plots saved at {}".format(SUMMARY_PLOT_PATH))
print("====================================")
