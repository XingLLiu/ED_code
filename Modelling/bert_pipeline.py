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
import subprocess
from tqdm import tqdm, trange
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from ED_support_module import EPICPreprocess
from ED_support_module import Evaluation
from ED_support_module.BertForSepsis import *


# ----------------------------------------------------
# Directories for saving files
# Path set-up
FIG_PATH = "../../results/bert/"
DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"
RAW_TEXT_PATH = "../../data/EPIC_DATA/EPIC.csv"
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
    parser.add_argument("--start_time",
                        type=int,
                        required=True,
                        help="Index of the month to start with. Must be 3 - 10.")
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
    return parser





# ----------------------------------------------------
# Preliminary settings

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# Parser arguements
parser = setup_parser()
args = parser.parse_args()

# Bert pre-trained model selected in the list: bert-base-uncased, 
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'clinical_bert'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'epic_task'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = '../../ClinicalBert/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 512

# Other model hyper-parameters
WEIGHT1 = 300000
WEIGHT0 = 100
TRAIN_BATCH_SIZE = 6
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 1e-6
NUM_TRAIN_EPOCHS = 1
RANDOM_SEED = 27
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.15
OUTPUT_MODE = 'classification'

CONFIG_NAME = "bert_config.json"
WEIGHTS_NAME = "pytorch_model.bin"
PREDICTION_HEAD_NAME = "prediction_head.bin"

MODEL_NAME = "bert"   # For saving the results

# Use GPU if exists otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPU = torch.cuda.device_count()


# ----------------------------------------------------
# Create folder to save raw text data if not exist
if not os.path.exists(RAW_SAVE_DIR):
    os.makedirs(RAW_SAVE_DIR)


# ----------------------------------------------------
# Prepare train and test sets
# Load file
EPIC_original = pd.read_csv(RAW_TEXT_PATH, encoding = 'ISO-8859-1')
preprocessor = EPICPreprocess.Preprocess(path = RAW_TEXT_PATH)
EPIC_original = preprocessor.BinarizeSepsis(EPIC_original)


# Only keep text columns and label
notes_cols = ['Note.Data_ED.Triage.Notes']
EPIC = EPIC_original[['Primary.Dx'] + notes_cols]


# ----------------------------------------------------
# ========= 1. Further preprocessing =========
# Clean the file if not already done
if not os.path.exists(RAW_SAVE_DIR + "EPIC.csv"):
    _ = clean_epic_notes(EPIC = EPIC,
                        EPIC_cc = EPIC_original,
                        notes_cols = notes_cols,
                        data_path = DATA_PATH,
                        save_path = RAW_SAVE_DIR)

# Load data
EPIC = pd.read_csv(RAW_SAVE_DIR + "EPIC.csv")
time_span = pickle.load( open( RAW_SAVE_DIR + "time_span", "rb" ) )



# ----------------------------------------------------
# ========= 2.a. One-month ahead prediction =========
print("====================================")
print("Dynamically evaluate the model ...\n")



for j, time in enumerate(time_span[args.start_time : args.start_time + 1]):
    # ========= 2.a. Setup =========
    # Month to be predicted
    j = args.start_time
    time = time_span[j]
    time_pred = time_span[j + 1]


    # Create folder if not already exist
    FIG_ROOT_PATH = FIG_PATH + f"dynamic/{TASK_NAME}/"
    DYNAMIC_PATH = FIG_ROOT_PATH + f"{time_pred}/"
    OUTPUT_DIR = DYNAMIC_PATH + f'Saved_Checkpoints/'
    REPORTS_DIR = DYNAMIC_PATH
    PROCESSED_NOTES_DIR = DYNAMIC_PATH + f"Processed_Texts/"
    for path in [DYNAMIC_PATH, OUTPUT_DIR, REPORTS_DIR, PROCESSED_NOTES_DIR]:
        if not os.path.exists(path):
            os.makedirs(path)


    # Prepare train/test sets
    XTrain, XTest, yTrain, yTest= time_split(data = EPIC, threshold = time)

    print("Training for data up to {} ...".format(time))
    print( "Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}]."
                .format( yTrain.shape, yTest.shape, yTrain.sum(), yTest.sum() ) )



    # ========= 2.a.i. Model =========
    if j == 2:
        # Convert data to the appropriate format and save
        train_bert = create_bert_data(x_data = XTrain["Note.Data_ED.Triage.Notes"],
                                        y_data = yTrain,
                                        save_path = PROCESSED_NOTES_DIR + "train.tsv")
        test_bert = create_bert_data(x_data = XTest["Note.Data_ED.Triage.Notes"],
                                        y_data = yTest,
                                        save_path = PROCESSED_NOTES_DIR + "dev.tsv")
        # Load data. Necessary for feeding in BERT
        processor = BinaryClassificationProcessor()
        train_data = processor.get_train_examples(PROCESSED_NOTES_DIR)
        label_list = processor.get_labels()
        # Optimization step
        # num_train_optimization_steps = int(
        #     len(XTrain) / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS // 10
        num_train_optimization_steps = 500
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
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-6},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=LEARNING_RATE,
                            warmup=WARMUP_PROPORTION,
                            t_total=num_train_optimization_steps)
        # optimizer = torch.optim.Adam(prediction_model.parameters(), lr = LEARNING_RATE)
        # Loss
        criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([WEIGHT0, WEIGHT1])).to(device)
    else:
        # Get train set (= test set from the last month)
        _, XTrainOld, _, yTrainOld= time_split(data = EPIC, threshold = time_span[j - 1])
        # Get directory of the previous model
        OUTPUT_DIR_OLD = FIG_PATH + "dynamic/" + f"{TASK_NAME}/{time}/" + f"Saved_Checkpoints/"
        # Convert data to the appropriate format and save
        train_bert = create_bert_data(x_data = XTrainOld["Note.Data_ED.Triage.Notes"],
                                        y_data = yTrainOld,
                                        save_path = PROCESSED_NOTES_DIR + "train.tsv")
        test_bert = create_bert_data(x_data = XTest["Note.Data_ED.Triage.Notes"],
                                        y_data = yTest,
                                        save_path = PROCESSED_NOTES_DIR + "dev.tsv")
        # Load data. Necessary for feeding in BERT
        processor = BinaryClassificationProcessor()
        train_data = processor.get_train_examples(PROCESSED_NOTES_DIR)
        label_list = processor.get_labels()
        # Optimization step
        # num_train_optimization_steps = int(
        #     len(train_data) / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS // 20
        num_train_optimization_steps = 500
        # Load pretrained tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR_OLD + 'vocab.txt', do_lower_case=False)
        model = BertModel.from_pretrained(OUTPUT_DIR_OLD + f"{TASK_NAME}.tar.gz",cache_dir=OUTPUT_DIR)
        # Load entire model
        prediction_model = torch.load(OUTPUT_DIR_OLD + "entire_model.ckpt")
        # Load optimizer
        optimizer = pickle.load(open(OUTPUT_DIR_OLD + "optimizer.ckpt", "rb"))
        # Load loss
        criterion = pickle.load(open(OUTPUT_DIR_OLD + "loss.ckpt", "rb"))



    # Convert tokens to features and save
    print("\nConverting examples to features ...")
    train_features = convert_examples_to_features(train_data, label_list, MAX_SEQ_LENGTH, tokenizer,
                                                    save_path = OUTPUT_DIR + "train_features.pkl")
    print("Complete and saved to {}".format(OUTPUT_DIR))

    # Set weights of all embedding layers trainable
    for p in model.parameters():
        p.requires_grad = True


    # ========= 2.a.ii. Fine-tuning =========
    # if j <= 5:
    # Train on the first 5 months to prevent overfitting
    prediction_model, loss_vec = prediction_model.fit(train_features = train_features,
                                                        num_epochs = NUM_TRAIN_EPOCHS,
                                                        batch_size = TRAIN_BATCH_SIZE,
                                                        optimizer = optimizer,
                                                        criterion = criterion)


    # Save bert model
    save_bert(model, tokenizer, OUTPUT_DIR)
    # Save entire model
    torch.save(prediction_model, OUTPUT_DIR + "entire_model.ckpt")
    # Save optimizer and loss function
    pickle.dump(optimizer, open( OUTPUT_DIR + "optimizer.ckpt", "wb" ) )
    pickle.dump(criterion, open( OUTPUT_DIR + "loss.ckpt", "wb" ) )
    # Save loss vector
    loss_vec = pd.Series(loss_vec, name = "loss")
    loss_vec.to_csv(OUTPUT_DIR + "loss_vec.csv", index = False, header = True)

    # Tar files
    P = subprocess.check_call(["./tar_bert_models.sh", OUTPUT_DIR, TASK_NAME, CONFIG_NAME, WEIGHTS_NAME])
    print("Models saved at {} \n".format(OUTPUT_DIR))


    # ========= 2.a.iii. Prediction =========
    # Get test data
    eval_examples = processor.get_dev_examples(PROCESSED_NOTES_DIR)
    eval_features = convert_examples_to_features(eval_examples,
                                                label_list, MAX_SEQ_LENGTH, tokenizer)
    # Transformation function
    transformation = nn.Sigmoid()

    pred = prediction_model.predict_proba_single(eval_features = eval_features,
                                                batch_size = EVAL_BATCH_SIZE,
                                                transformation = transformation)


    # Save predicted probabilities
    pred = pd.DataFrame(pred, columns = ["pred_prob"])
    pred.to_csv(REPORTS_DIR + f"predicted_result_{time_pred}.csv", index = False, header = True)

    # Save probs for train set (for stacked model)
    pred_train = prediction_model.predict_proba_single(eval_features = train_features,
                                                        batch_size = EVAL_BATCH_SIZE,
                                                        transformation = transformation)
    pred_train = pd.DataFrame(pred_train, columns = ["pred_prob"])
    pred_train.to_csv(DYNAMIC_PATH + f"predicted_result_train_{time_pred}.csv", index = False)   


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
if args.start_time == 10:
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
