# ----------------------------------------------------
# Up to two inputs:
# 1. if clean note
# 2. mode: train, test, train_test
# ----------------------------------------------------
from __future__ import absolute_import, division, print_function
from ED_support_module import *
sys.path.append("../ClinicalNotePreProcessing")
from extract_dates_script import findDates
import csv
import logging
from tqdm import tqdm, trange
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module.BertForSepsis import *


# ----------------------------------------------------
# Directories for saving files
# Path to save figures
FIG_PATH = "../../results/bert/"
DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"
TEXT_DATA_PATH = "../../data/EPIC_DATA/EPIC.csv"
RAW_SAVE_DIR = FIG_PATH + "Raw_Notes/"


# ----------------------------------------------------
# Arguments
def setup_parser():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--clean_notes",
                        default=False,
                        help="True if clean the notes. False if load cleaned notes.")
    parser.add_argument("--mode",
                        default="test",
                        type=str,
                        help="train for fine-tuning. test for evaluation. train/test for both.")
    parser.add_argument("--bert_model",
                        default="clinical_bert",
                        type=str,
                        help="Which BERT model to run.")
    parser.add_argument("--task_name",
                        default="epic_task",
                        type=str,
                        required=True,
                        help="Customized task name.")
    parser.add_argument("--path",
                        type=str,
                        default=None,
                        required=True,
                        help="Directory of the raw EPIC data.")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="Maximum length of a text sequence.")
    parser.add_argument("--weight",
                        default=1500,
                        type=int,
                        help="Weight for the minority class."
                        )
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Batch size during fine-tuning.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Batch size during testing.")
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="Learning rate during fine-tuning.")
    parser.add_argument
    return parser


# ----------------------------------------------------
# Preliminary settings

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# Parser arguements
# parser = setup_parser()
# args = parser.parse_args()

# Bert pre-trained model selected in the list: bert-base-uncased, 
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'clinical_bert'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'epic_task'

# # The output directory where the fine-tuned model and checkpoints will be written.
# OUTPUT_DIR = SAVE_DIR + f'Saved_Checkpoints/{TASK_NAME}/'

# # The directory where the evaluation reports will be written to.
# REPORTS_DIR = SAVE_DIR + f'Reports/{TASK_NAME}_evaluation_report/'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = '../../ClinicalBert/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 512

# Other model hyper-parameters
MODE = "a"
WEIGHT = 500
WEIGHT2 = 16
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 40
LEARNING_RATE = 1e-3
NUM_TRAIN_EPOCHS = 6
RANDOM_SEED = 27
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "bert_config.json"
WEIGHTS_NAME = "pytorch_model.bin"
PREDICTION_HEAD_NAME = "prediction_head.bin"

MODEL_NAME = "bert"   # For saving the results

# Use GPU if exists otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPU = torch.cuda.device_count()



CLEAN_NOTES = False



# ----------------------------------------------------
# Create folder to save raw text data if not exist
if not os.path.exists(RAW_SAVE_DIR):
    os.makedirs(RAW_SAVE_DIR)


# ----------------------------------------------------
# Prepare train and test sets
# Load file
EPIC_original = pd.read_csv(TEXT_DATA_PATH, encoding = 'ISO-8859-1')
preprocessor = EPICPreprocess.Preprocess(path = TEXT_DATA_PATH)
EPIC_original = preprocessor.BinarizeSepsis(EPIC_original)


# Only keep text columns and label
notesCols = ['Note.Data_ED.Triage.Notes']
EPIC = EPIC_original[['Primary.Dx'] + notesCols]


# ----------------------------------------------------
# ========= 1. Further preprocessing =========

if CLEAN_NOTES:
    # Loop over each file and write to a csv
    print("\nStart cleaning notes ...")
    # Clean text
    for col in notesCols:
        print("Cleaning {}".format(col))
        EPIC.loc[:, col] = list(map(clean_text, EPIC[col]))
    # Save data
    EPIC.to_csv(RAW_SAVE_DIR + 'EPIC_triage.csv', index=False)
    # Load data nonetheless to convert empty notes "" to nan
    EPIC = pd.read_csv(RAW_SAVE_DIR + 'EPIC_triage.csv')
    # Fill in missing vals
    EPIC = fill_missing_text(EPIC, EPIC_original, notesCols)
    # Save imputed text
    EPIC.to_csv(RAW_SAVE_DIR + 'EPIC_triage.csv', index=False)
    # Further preprocessing
    preprocessor = EPICPreprocess.Preprocess(DATA_PATH)
    _, _, _, EPIC_arrival = preprocessor.streamline()
    # Remove the obvious outliers
    EPIC = EPIC.loc[EPIC_arrival.index, :]
    # Add time variable
    EPIC = pd.concat([EPIC, EPIC_arrival["Arrived"].astype(int)], axis = 1)
    # Get time span
    time_span = EPIC['Arrived'].unique().tolist()
    # Save data
    EPIC.to_csv(RAW_SAVE_DIR + 'EPIC.csv', index=False)
    pickle.dump(time_span, open( RAW_SAVE_DIR + "time_span", "wb") )
else:
    # Load data
    EPIC = pd.read_csv(RAW_SAVE_DIR + "EPIC.csv")
    time_span = pickle.load( open( RAW_SAVE_DIR + "time_span", "rb" ) )



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
    OUTPUT_DIR = DYNAMIC_PATH + f'Saved_Checkpoints/{TASK_NAME}/'
    REPORTS_DIR = DYNAMIC_PATH + f'Reports/{TASK_NAME}_evaluation_report/'
    for path in [DYNAMIC_PATH, OUTPUT_DIR, REPORTS_DIR]:
        if not os.path.exists(path):
            os.makedirs(path)


    # Prepare train/test sets
    XTrain, XTest, yTrain, yTest= time_split(data = EPIC, threshold = time)

    print("Training for data up to {} ...".format(time))
    print( "Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}]."
                .format( len(yTrain), len(yTest), yTrain.sum(), yTest.sum() ) )



    # ========= 2.a.i. Model =========
    if j == 0:
        # Convert to the appropriate format and save
        train_bert = create_bert_data(x_data = XTrain["Note.Data_ED.Triage.Notes"],
                                        y_data = yTrain,
                                        save_path = OUTPUT_DIR + "train.tsv")
        test_bert = create_bert_data(x_data = XTest["Note.Data_ED.Triage.Notes"],
                                        y_data = yTest,
                                        save_path = OUTPUT_DIR + "dev.tsv")
        # Load data. Necessary for feeding in BERT
        processor = BinaryClassificationProcessor()
        train_data = processor.get_train_examples(OUTPUT_DIR)
        label_list = processor.get_labels()
        # Optimization step
        num_train_optimization_steps = int(
            len(train_data) / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
        # Load pretrained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained(CACHE_DIR, do_lower_case=False)
        # Load model
        model = BertModel.from_pretrained(CACHE_DIR, cache_dir=CACHE_DIR).to(device)
        # Prediction model
        prediction_model = BertForSepsis(bert_model = model,
                                        device = device,
                                        hidden_size = model.config.hidden_size).to(device)
        # Optimizer
        param_optimizer = list(prediction_model.named_parameters())
        no_decay = []
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=LEARNING_RATE,
                            warmup=WARMUP_PROPORTION,
                            t_total=num_train_optimization_steps)

        # Loss
        criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, WEIGHT])).to(device)
    else:
        # Convert to the appropriate format and save
        train_bert = create_bert_data(x_data = XTrainOld["Note.Data_ED.Triage.Notes"],
                                        y_data = yTrainOld,
                                        save_path = OUTPUT_DIR + "train.tsv")
        test_bert = create_bert_data(x_data = XTest["Note.Data_ED.Triage.Notes"],
                                        y_data = yTest,
                                        save_path = OUTPUT_DIR + "dev.tsv")
        # Load data. Necessary for feeding in BERT
        processor = BinaryClassificationProcessor()
        train_data = processor.get_train_examples(OUTPUT_DIR)
        label_list = processor.get_labels()
        # Optimization step
        num_train_optimization_steps = int(
            len(train_data) / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
        # Load pretrained tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR_OLD + 'vocab.txt', do_lower_case=False)
        model = pickle.load(open(OUTPUT_DIR_OLD + "bert_model.pkl", "rb"))
        # model = BertModel.from_pretrained(CACHE_DIR, cache_dir=CACHE_DIR).to(device)
        # model.load_state_dict( torch.load( OUTPUT_DIR_OLD + WEIGHTS_NAME ) )
        # Load entrie model
        prediction_model = pickle.load(open(OUTPUT_DIR_OLD + "entire_model.pkl", "rb"))
        # prediction_model = BertForSepsis(bert_model = model,
        #                                 device = device,
        #                                 hidden_size = model.config.hidden_size).to(device)
        # prediction_model = prediction_model.load_state_dict( torch.load( OUTPUT_DIR_OLD + "entire_model.pkl" ) )



    # Convert tokens to features.
    print("\nConverting examples to features ...")
    train_features = convert_examples_to_features(train_data, label_list, MAX_SEQ_LENGTH, tokenizer)

    # Save converted features
    pickle.dump(train_features, open(OUTPUT_DIR + "train_features.pkl", 'wb'))
    print("Complete and saved to {}".format(OUTPUT_DIR))

    # Set weights of all embedding layers trainable
    for p in model.parameters():
        p.requires_grad = True


    # Set up data loaders
    train_loader = feature_to_loader(train_features, TRAIN_BATCH_SIZE)

    # Train the model
    loss_vec = np.zeros(1)
    for epoch in range(NUM_TRAIN_EPOCHS):
        loss = prediction_model.train_model(train_loader,
                                            criterion = criterion,
                                            optimizer = optimizer)
        if epoch == 0:
            loss_vec = loss
        else:
            loss_vec = np.append(loss_vec, loss)


    # Save model
    pickle.dump(model, open(OUTPUT_DIR + "bert_model.pkl", "wb"))
    pickle.dump(prediction_model, open(OUTPUT_DIR + "entire_model.pkl", "wb"))
    print("Models saved at {} \n".format(OUTPUT_DIR))
    # save_bert(prediction_model = prediction_model,
    #             bert_model = model,
    #             tokenizer = tokenizer,
    #             OUTPUT_DIR = OUTPUT_DIR,
    #             WEIGHTS_NAME = WEIGHTS_NAME,
    #             CONFIG_NAME = CONFIG_NAME)


    # Prediction
    # Get tokenizer
    tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)
    # Get test data
    eval_examples = processor.get_dev_examples(OUTPUT_DIR)
    eval_features = convert_examples_to_features(eval_examples,
                                                label_list, MAX_SEQ_LENGTH, tokenizer)
    test_loader = feature_to_loader(eval_features, EVAL_BATCH_SIZE)
    


    transformation = nn.Sigmoid()
    pred = prediction_model.eval_model(test_loader = test_loader,
                                        batch_size = EVAL_BATCH_SIZE,
                                        transformation = transformation)


    # Save predicted probabilities
    pred = pd.DataFrame(pred, columns = ["pred_prob"])
    pred.to_csv(REPORTS_DIR + f"predicted_result_{time_pred}.csv", index = False)

    # Save data of this month as train set for the next iteration
    XTrainOld = XTest
    yTrainOld = yTest
    # Save trained model of this month as train set for the next iteration
    OUTPUT_DIR_OLD = OUTPUT_DIR


    # ========= 2.b. Evaluation =========
    evaluator = Evaluation.Evaluation(yTest, pred)

    # Save ROC plot
    _ = evaluator.roc_plot(plot = False, title = MODEL_NAME, save_path = REPORTS_DIR + f"roc_{time_pred}")

    # Save summary
    summary_data = evaluator.summary()
    summary_data.to_csv(REPORTS_DIR+ f"summary_{time_pred}.csv", index = False)


    # ========= 2.c. Save predicted results =========
    pred = pd.DataFrame(pred, columns = ["pred_prob"])
    pred.to_csv(REPORTS_DIR + f"predicted_result_{time_pred}.csv", index = False)


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
