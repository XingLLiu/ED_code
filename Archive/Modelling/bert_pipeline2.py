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
from ED_support_module.BertForSepsis import create_bert_data



# ----------------------------------------------------
# Directories for saving files
FIG_PATH = "../../results/bert/"
DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"
TEXT_DATA_PATH = "../../data/EPIC_DATA/EPIC.csv"
RAW_SAVE_DIR = FIG_PATH + "Raw_Notes/"


# ----------------------------------------------------
# Arguments


# ----------------------------------------------------
# Preliminary settings
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# # Parser arguements
# parser = setup_parser()
# args = parser.parse_args()

# Bert pre-trained model
BERT_MODEL = 'clinical_bert'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'epic_task'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = '../../ClinicalBert/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 512

# Other model hyper-parameters
MODE = "a"
WEIGHT = 500
WEIGHT2 = 16
TRAIN_BATCH_SIZE = 6
EVAL_BATCH_SIZE = 8
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


j = 0
time = time_span[j]
time_pred = time_span[j + 1]



# Create folder if not already exist
FIG_ROOT_PATH = FIG_PATH + "dynamic/"
DYNAMIC_PATH = FIG_ROOT_PATH + f"{time_pred}/"
OUTPUT_DIR = DYNAMIC_PATH + f'Saved_Checkpoints/{TASK_NAME}/'
REPORTS_DIR = DYNAMIC_PATH + f'Reports/{TASK_NAME}_evaluation_report/'
for path in [DYNAMIC_PATH, OUTPUT_DIR, REPORTS_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)


# Prepare train/test sets
XTrain, XTest, yTrain, yTest= time_split(data = EPIC, threshold = time)

print("Training for data up to {} ...".format(time))
print( "Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}]."
            .format( yTrain.shape, yTest.shape, yTrain.sum(), yTest.sum() ) )



# Convert to the appropriate format and save
train_bert = create_bert_data(x_data = XTrain["Note.Data_ED.Triage.Notes"],
                                y_data = yTrain,
                                save_path = OUTPUT_DIR + "train.tsv")
test_bert = create_bert_data(x_data = XTest["Note.Data_ED.Triage.Notes"],
                                y_data = yTest,
                                save_path = OUTPUT_DIR + "dev.tsv")


# ----------------------------------------------------
# Text to feature classes
class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        # Make object iterable
        def __iter__(self):
            return self
        def __next__(self):
            return self.guid + 1


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell) for cell in line)
                lines.append(line)
            return lines


class BinaryClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    def get_labels(self):
        """See base class."""
        return ["0", "1"]
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
    

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    max_len = 0
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            seq_len = len(tokens_a) + len(tokens_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            seq_len = len(tokens_a)
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        if seq_len > max_len:
            max_len = seq_len
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        # tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens = tokens_a
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # # Segment id
        # counter=0
        # for i, id in enumerate(input_ids):
        #     if id == 102:
        #         #'[SEP]'==102
        #         counter+=1
        #     segment_ids[i] = counter
        # print(segment_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label_id = label_map[example.label]
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        features.append(
                InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_id))
    print('Max Sequence Length: %d' %max_len)
    return features


class NoteClassificationHead(nn.Module):
    """
    Head layer for prediction.
    """
    def __init__(self, hidden_size, dropout_prob=0.4, num_labels=2):
        super(NoteClassificationHead, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)
    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits





# ----------------------------------------------------
# Main method
# ----------------------------------------------------

# Prepare for fine-tuning
# Load data
processor = BinaryClassificationProcessor()
trainData = processor.get_train_examples(OUTPUT_DIR)
labelList = processor.get_labels()

num_train_optimization_steps = int(
    len(trainData) / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS * 10

# Load pretrained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(CACHE_DIR, do_lower_case=False)

# Convert tokens to features. Load model if exists
if "train_features.pkl" not in os.listdir(OUTPUT_DIR):
    print("No previously converted features. Converting now ...")
    trainFeatures = convert_examples_to_features(trainData, labelList, MAX_SEQ_LENGTH, tokenizer)
    # Save converted features
    pickle.dump(trainFeatures, open(OUTPUT_DIR + "train_features.pkl", 'wb'))
    print("Complete and saved to {}".format(OUTPUT_DIR))
else:
    print("Loading converted features ...")
    trainFeatures = pickle.load(open(OUTPUT_DIR + "train_features.pkl", 'rb'))
    print("Complete")



# ----------------------------------------------------
# Fine-tuning

# Set optimizer and data loader
# Load model
model = BertModel.from_pretrained(CACHE_DIR, cache_dir=CACHE_DIR).to(device)
# Update weights of all embedding layers
for p in model.parameters():
    p.requires_grad = True



print("***** Running training *****")
print("  Num examples = {}".format( len(trainData) ) )
print("  Batch size = {}".format( TRAIN_BATCH_SIZE ) )
print("  Num steps = {}".format( num_train_optimization_steps ) )

all_input_ids = torch.tensor([f.input_ids for f in trainFeatures], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in trainFeatures], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in trainFeatures], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in trainFeatures], dtype=torch.long)

# Set up data loaders
train_dataloader = torch.utils.data.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_dataloader = torch.utils.data.DataLoader(train_dataloader, batch_size=TRAIN_BATCH_SIZE)


# ----------------------------------------------------
# Fine tuning
loss_vec = np.zeros(NUM_TRAIN_EPOCHS * (len(train_dataloader) // 10))

prediction_head = NoteClassificationHead(hidden_size=model.config.hidden_size)
_ = prediction_head.to(device)


# Optimizers
param_optimizer = list(model.named_parameters())
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
no_decay = []
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


optimizer_bert = BertAdam(optimizer_grouped_parameters,
                        lr=LEARNING_RATE,
                        warmup=WARMUP_PROPORTION,
                        t_total=num_train_optimization_steps)
optimizer_head = torch.optim.Adam( list( prediction_head.parameters() ), lr=LEARNING_RATE )

loss_func = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, WEIGHT])).to(device)
_ = model.train()
_ = prediction_head.train()

print("Start fine-tuning ...")
for epoch in range(NUM_TRAIN_EPOCHS):
    for i, batch in enumerate(tqdm(train_dataloader)):
        # Get batch
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        _, pooled_output = model(input_ids, segment_ids, input_mask)
        logits = prediction_head(pooled_output.to(device)).to(device)
        # Compute loss
        loss = loss_func(logits, label_ids)
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer_bert.zero_grad()
            optimizer_head.zero_grad()
        if NUM_GPU > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer_bert.step()
            optimizer_head.step()
        if (i + 1) % 10 == 0:
            loss_vec[epoch * (len(train_dataloader) // 10) + i // 10] = loss.item()






# ----------------------------------------------------
# Save model
model_to_save = model.module if hasattr(model, "module") else model
layer_to_save = prediction_head.module if hasattr(prediction_head, "module") else prediction_head

# Save using the predefined names so that one can load using `from_pretrained`
output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)
output_classification_file = os.path.join(OUTPUT_DIR, PREDICTION_HEAD_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(OUTPUT_DIR)

# save weights of the final classification layer
torch.save(layer_to_save.state_dict(), output_classification_file)

# Save loss vector
pickle.dump(loss_vec, open(REPORTS_DIR + "loss.pkl", 'wb'))

# Save loss plot
_ = sns.scatterplot(x=range(len(loss_vec)), y=loss_vec)
_ = plt.title("Clinical BERT Train Loss")
plt.savefig(REPORTS_DIR + "train_loss.eps", format="eps", dpi=1000)


# Tar model files
# import tarfile
# BERT_MODEL = OUTPUT_DIR + f"{TASK_NAME}.tar.gz"
# tf = tarfile.open(BERT_MODEL, mode = "w:gz")
# tf.add(output_config_file)
# tf.add(output_model_file)
# tf.close()

import subprocess
P = subprocess.check_call(["./tar_bert_models.sh", OUTPUT_DIR, TASK_NAME, CONFIG_NAME, WEIGHTS_NAME])


# ----------------------------------------------------
# ----------------------------------------------------

# Set path
BERT_MODEL = OUTPUT_DIR + f"{TASK_NAME}.tar.gz"
# Load fine-tuned model
tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)
processor = BinaryClassificationProcessor()

# Testing
# Set test set loaders
eval_examples = processor.get_dev_examples(OUTPUT_DIR)
eval_features = convert_examples_to_features(eval_examples,
                    labelList, MAX_SEQ_LENGTH, tokenizer)
# logger.info("***** Running evaluation *****")
# logger.info("  Num examples = %d", len(eval_examples))
# logger.info("  Batch size = %d", EVAL_BATCH_SIZE)

# Integrate data into required format
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = torch.utils.data.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

# Run prediction for full data
eval_sampler = torch.utils.data.SequentialSampler(eval_data)
eval_dataloader = torch.utils.data.DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

# Load fine-tuned model (weights)
model = BertModel.from_pretrained(OUTPUT_DIR + f"{TASK_NAME}.tar.gz",cache_dir=OUTPUT_DIR)
# model.load_state_dict( torch.load(OUTPUT_DIR + WEIGHTS_NAME) )
_ = model.to(device)
_ = model.eval()


prediction_head = NoteClassificationHead(hidden_size=model.config.hidden_size)
prediction_head.load_state_dict( torch.load(OUTPUT_DIR + PREDICTION_HEAD_NAME) )
_ = prediction_head.eval()
_ = prediction_head.to(device)
print("fine-tuned pred_head:", prediction_head.state_dict())

# Final setup
sigmoid_func = nn.Sigmoid()
prob = np.zeros(len(eval_data))

for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
    input_ids, input_mask, segment_ids, label_ids = batch
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)
    # Evaluate loss
    with torch.no_grad():
        _, pooled_output = model(input_ids, segment_ids, input_mask)
        pooled_output = pooled_output.to(device)
        logits = prediction_head(pooled_output)
        # Evaluation metric
        logits = logits.detach().cpu().numpy()
        logits = torch.from_numpy(logits[:, 1])
        pred_prob = sigmoid_func(logits)
        label_ids = label_ids.to('cpu').numpy()
    # Store predicted probabilities
    begin_ind = EVAL_BATCH_SIZE * i
    end_ind = np.min( [ EVAL_BATCH_SIZE * (i + 1), len(eval_data) ] )
    prob[begin_ind:end_ind] = pred_prob


# Save predicted probabilities
pickle.dump(prob, open(REPORTS_DIR + "predicted_probs.pkl", 'wb'))
prob = pd.DataFrame(prob, columns = ["pred_prob"])
prob.to_csv(REPORTS_DIR + "predicted_probs.csv", index = False)

roc = lr_roc_plot(yTest, prob, save_path = REPORTS_DIR + f'roc.eps', plot = True)


