from ED_support_module import *
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module import OneClassSVM


# ----------------------------------------------------
# ========= 0. Preliminary seetings =========
MODEL_NAME = "OC-SVM"
RANDOM_SEED = 27
MODE = "a"

KERNEL = "rbf"
NU = 0.1
GAMMA = 0.0001




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
FIG_PATH = "../../results/ocsvm/"
DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"
FIG_ROOT_PATH = FIG_PATH + f"dynamic_{MODE}/"


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
    # Fit model
    model = sk.svm.OneClassSVM(kernel = KERNEL, nu = NU, gamma = GAMMA)
    model = model.fit(XTrain.loc[yTrain == 0, :])

    # Prediction
    pred = model.predict_transform(XTest)

    # ========= 2.a.ii. Feature importance by permutation test =========
    # Permutation test
    imp_means, imp_vars = mlxtend.evaluate.feature_importance_permutation(
                            predict_method = model.predict_transform,
                            X = np.array(XTest),
                            y = np.array(yTest),
                            metric = sk.metrics.f1_score,
                            num_rounds = 10,
                            seed = RANDOM_SEED)

    # Save feature importance plot
    fi_evaluator = Evaluation.FeatureImportance(imp_means, imp_vars, XTest.columns, MODEL_NAME)
    fi_evaluator.FI_plot(save_path = DYNAMIC_PATH, y_fontsize = 4, eps = True)


    # ========= 2.b. Evaluation =========
    # 'Evaluation' class does not make sense for non-score predicted values
    _= roc_plot(yTest = yTest,
                pred = pred,
                plot = False,
                show_results = False,
                save_path = DYNAMIC_PATH + f"roc_{time_pred}.eps",
                eps = True)


    # ========= End of iteration =========
    print("Completed evaluation for {}.\n".format(time_pred))



print("====================================")
