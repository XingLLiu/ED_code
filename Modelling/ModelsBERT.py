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

# from EDA import EPIC, EPIC_enc, EPIC_CUI, numCols, catCols 


# ----------------------------------------------------
# Directories for saving files
path = '/home/xingliu/Documents/ED/data/EPIC_DATA/EPIC.csv'
SAVE_DIR = '/'.join(path.split('/')[:-1]) + '/EPIC_with_Bert/'
RAW_SAVE_DIR = SAVE_DIR + 'Raw_Notes/'
PROCESSED_SAVE_DIR = SAVE_DIR + 'Processed_Notes/'


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
parser = setup_parser()
args = parser.parse_args()

# Bert pre-trained model selected in the list: bert-base-uncased, 
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'clinical_bert'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'epic_task'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = SAVE_DIR + f'Saved_Checkpoints/{TASK_NAME}/'

# The directory where the evaluation reports will be written to.
REPORTS_DIR = SAVE_DIR + f'Reports/{TASK_NAME}_evaluation_report/'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = '/'.join(path.split('/')[:-3]) + '/ClinicalBert/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 512

# Other model hyper-parameters
WEIGHT = 500
WEIGHT2 = 16
TRAIN_BATCH_SIZE = 40
EVAL_BATCH_SIZE = 40
LEARNING_RATE = 1e-3
NUM_TRAIN_EPOCHS = 2
RANDOM_SEED = 27
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "bert_config.json"
WEIGHTS_NAME = "pytorch_model.bin"
PREDICTION_HEAD_NAME = "prediction_head.bin"

# Use GPU if exists otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPU = torch.cuda.device_count()


# ----------------------------------------------------
# Create folder to save evaluation reports if not exist
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)


# Create folder to save fine-tuned model if not exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ----------------------------------------------------
# Prepare train and test sets
# Load file
EPIC_original = pd.read_csv(path, encoding = 'ISO-8859-1')

# Assign labels
# Check if Primary.Dx contains Sepsis or related classes
ifSepsis1 = EPIC_original['Primary.Dx'].str.contains('epsis')
# Check if Diagnosis contains Sepsis or related classes
ifSepsis2 = EPIC_original['Diagnosis'].str.contains('epsis')
# Check if Diagnoses contains Sepsis or related classes
ifSepsis3 = EPIC_original['Diagnoses'].str.contains('epsis')
# Lable as sepsis if any of the above contains Sepsis
ifSepsis = ifSepsis1 | ifSepsis2 | ifSepsis3
EPIC_original['Primary.Dx'] = ifSepsis.astype('int')

# Only keep text columns and ID's
# notesCols = ['Note.Data_ED.Notes', 'Note.Data_ED.Procedure.Note', 'Note.Data_ED.Provider.Notes', 'Note.Data_ED.Triage.Notes']
notesCols = ['Note.Data_ED.Triage.Notes']
EPIC = EPIC_original[['Primary.Dx'] + notesCols]


# ----------------------------------------------------
# Clean notes 

# Copied and modified from cleaning_script.py
#a list of common abbreviations that do not have other potential meanings
abbrevs = {'hrs':'hours', 'mins':'minutes',
           'S&S':'signs and symptoms', 
           'bc':'because', 'b/c':'because', 
           'wo':'without', 'w/o':'without', 
           'yo':'year old', 'y.o':'year old', 'wk':'weeks',
           'm.o':'month old', 'mo':'months', 'mos':'months', 
           'b4':'before', 'pt':'patient',
           'ro':'rule out', 'w/':'with', 
           'o/n':'overnight', 'f/u':'follow up',
           'M':'male', 'F':'female'}


# Function that cleans the text
def clean_text(text):
    if ((text == 'nan') | (text != text)):
        return ''
    #date extraction and replacement
    # dates = findDates(text)[0] # USE ME PLEASE!
    text = findDates(text)[1]
    #note structure
    text = text.replace ("," , " ")
    text = re.sub (" *<<STARTNOTE.*<NOTETEXT", "", text)
    text = text.replace("NOTETEXT>ENDNOTE>>", " ")
    text = re.sub (" *<CRLF>", ". ", text)
    #erroneous UTF symbols
    text = re.sub ("[•â€¢Ã]+", "", text)
    #abbreviations
    for abb in abbrevs:
        if " " + abb + "." in text:
            text = text.replace (" " + abb + ".", " " + abbrevs[abb] + ".")
        elif  " " + abb + " " in text:
            text = text.replace (" " + abb + " ", " " + abbrevs[abb] + " ")
    #numeric ranges
    grp = re.findall ("(?<![0-9]-)([0-9]+) *- *([0-9]+)(?!-[0-9])", text)
    for g in grp:
        text = re.sub ("(?<![0-9]-)" + g[0]+" *- *" + g[1] + "(?!-[0-9])", g[0] + " to " + g[1], text)
    #dealing with x[0-9]+ d
    grp = re.findall("x *([0-9]+) *d([ .,]+)", text)
    for g in grp:
        text = re.sub ("x *" + g[0] + " *d"+g[1], "for " + g[0] + " days" + g[1], text)
    grp = re.findall("x *([0-9]+)/d([ .,]+)", text)
    for g in grp:
        text = re.sub ("x *" + g[0] + "/d"+g[1], g[0] + " times per day" + g[1], text)
    grp = re.findall("x *([0-9]+)/day([ .,]+)", text)
    for g in grp:
        text = re.sub ("x *" + g[0] + "/day" + g[1], g[0] + " times per day" + g[1], text)       
    #dealing with multiple plus signs
    grp = re.findall ("([a-zA-Z0-9]*) *\+{2,3}", text)
    for g in grp:
        text = re.sub (g + " *\+{2,3}", "significant " + g, text)
    #switching symbols for equivalent words
    text = text.replace ("%", " percent ")
    text = text.replace ("=" , " is ")
    text = text.replace ("\$", " dollars ")
    text = text.replace (">", " greater than ")
    text = text.replace ("<", " less than ")
    text = text.replace ("?", " possible ")
    text = text.replace ("~", " approximately ")
    text = text.replace ("(!)", " abnormal ")
    text = text.replace ("@", "at")
    #switching abbreviations: pt or Pt for patient
    text = re.sub ("(\spt\s)|(Pt\s)", " patient ", text)
    #numeric ratios
    grp = re.findall ("(\d{1,1}) *\: *(\d{1,2}) *[^ap][^m][^a-zA-Z0-9]", text)
    for g in grp:
        text = re.sub (g[0] + " *: *" + g[1], g[0] + " to " + g[1] + " ratio", text)
	#symbol removal
    text = text.replace ("["," ")
    text = text.replace ("]"," ")
    text = text.replace ("{"," ")
    text = text.replace ("}"," ")
    text = text.replace ("\\"," ")
    text = text.replace ("|"," ")
    text = text.replace ("-"," ")
    text = text.replace ("_"," ")
    # BERT special tokens
    text = text.replace ('.', ' [SEP] ')
    text = '[CLS] '+ text
    if '[SEP]' not in text[-10:]:
        text = text +' [SEP]'
    text = re.sub("\[SEP\]\s+\[SEP\]", " [SEP] ", text)
    #extra spaces
    text = re.sub (" +", " ", text)
	#extra periods
    text = re.sub ("\. *\.[ .]+", ". ", text)	
    return text


if args.clean_notes == True:
    # Loop over each file and write to a csv
    print("Start cleaning notes ...")
    # Clean text
    for col in notesCols:
        print("Cleaning {}".format(col))
        EPIC.loc[:, col] = list(map(clean_text, EPIC[col]))
    # Save data
    if len(notesCols) == 1:
        EPIC.to_csv(RAW_SAVE_DIR + 'EPIC_triage.csv', index=False)
    else:
        EPIC.to_csv(RAW_SAVE_DIR + 'EPIC_all_notes.csv', index=False)
    # Load data nonetheless to convert empty notes "" to nan
    EPIC = pd.read_csv(RAW_SAVE_DIR + 'EPIC_triage.csv')
else:
    # Load data
    EPIC = pd.read_csv(RAW_SAVE_DIR + 'EPIC_triage.csv')


# Fill in missing notes by CC
for col in notesCols:
    ifNull = EPIC[col].isnull()
    print('Column:', col, '\nNo. of empty entries:', ifNull.sum())
    EPIC.loc[ifNull, col] = "[CLS] " + EPIC_original['CC'][ifNull] + " [SEP]"
    print('No. of empty entries after imputing by CC: {}'
          .format(EPIC[col].isnull().sum()))
    # Impute the remaining missing notes by 'none'
    if EPIC[col].isnull().sum() > 0:
        print('Impute the remaining missing notes by \'None.\' ')
        EPIC.loc[EPIC[col].isnull(), col] = '[CLS] None [SEP]'


# ----------------------------------------------------
# Train-test-validation split
y = EPIC['Primary.Dx']
X = EPIC.drop('Primary.Dx', axis = 1)
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25,
                               random_state=RANDOM_SEED, stratify=y)
# XTrain, XValid, yTrain, yValid = sk.model_selection.train_test_split(XTrain, yTrain, test_size=0.15,
#                                 random_state=RANDOM_SEED, stratify=yTrain)



# Change the format of train/test sets
trainBert = pd.DataFrame({
             'id': range(XTrain.shape[0]),
             'label': yTrain,
             'alpha': ['a'] * XTrain.shape[0],
             'text': XTrain['Note.Data_ED.Triage.Notes']
})

devBert = pd.DataFrame({
            'id': range(XTest.shape[0]),
            'label': yTest,
            'alpha': ['a'] * XTest.shape[0],
            'text': XTest['Note.Data_ED.Triage.Notes']
})

# validBert = pd.DataFrame({
#             'id': range(XValid.shape[0]),
#             'label': yValid,
#             'alpha': ['a'] * XValid.shape[0],
#             'text': XValid['Note.Data_ED.Triage.Notes']
# })

# Save data
trainBert.to_csv(PROCESSED_SAVE_DIR + 'train.tsv', sep='\t', index=False, header=False)
devBert.to_csv(PROCESSED_SAVE_DIR + 'dev.tsv', sep='\t', index=False, header=False)
# validBert.to_csv(PROCESSED_SAVE_DIR + 'valid.tsv', sep='\t', index=False, header=False)


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
        if (ex_index % 10000) == 0:
            print('Iteration:', ex_index)
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


# class NoteClassificationHead(nn.Module):
#     """
#     Head layer for prediction.
#     """
#     def __init__(self, device, model, hidden_size, dropout_prob=0.4, num_labels=2):
#         super(NoteClassificationHead, self).__init__()
#         self.device = device
#         self.model = model
#         self.num_labels = num_labels
#         self.dropout = nn.Dropout(dropout_prob)
#         self.classifier = nn.Linear(hidden_size, num_labels)
#         # nn.init.xavier_normal_(self.classifier.weight)
#     def forward(self, input_ids, segment_ids, input_mask):
#         _, pooled_output = model(input_ids, segment_ids, input_mask)
#         pooled_output = self.dropout(pooled_output.to(self.device))
#         logits = self.classifier(pooled_output)
#         return logits




# ----------------------------------------------------
# Main method
# ----------------------------------------------------
# Prepare for fine-tuning
# Load data
processor = BinaryClassificationProcessor()
trainData = processor.get_train_examples(PROCESSED_SAVE_DIR)
labelList = processor.get_labels()

num_train_optimization_steps = int(
    len(trainData) / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS

# Load pretrained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(CACHE_DIR, do_lower_case=False)

# labelMap = {label: i for i, label in enumerate(labelList)}
# trainForProcessing = [(example, labelMap, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in trainData]

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
if args.mode == "train" or args.mode == "train_test":
    # Fine-tuning

    # Set optimizer and data loader
    # Load model
    model = BertModel.from_pretrained(CACHE_DIR, cache_dir=CACHE_DIR).to(device)
    # Update weights of all embedding layers
    # for p in model.embeddings.parameters():
    #     p.requires_grad = True

    for p in model.parameters():
        p.requires_grad = True


    # Optimizers
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                     lr=LEARNING_RATE,
    #                     warmup=WARMUP_PROPORTION,
    #                     t_total=num_train_optimization_steps)


    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", train_examples_len)
    # logger.info("  Batch size = %d", TRAIN_BATCH_SIZE)
    # logger.info("  Num steps = %d", num_train_optimization_steps)
    print("***** Running training *****")
    print("  Num examples = {}".format( len(trainData) ) )
    print("  Batch size = {}".format( TRAIN_BATCH_SIZE ) )
    print("  Num steps = {}".format( num_train_optimization_steps ) )

    all_input_ids = torch.tensor([f.input_ids for f in trainFeatures], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in trainFeatures], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in trainFeatures], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in trainFeatures], dtype=torch.long)

    ## Set up weight vector
    train_weights = np.array(WEIGHT2 * yTrain + 1 - yTrain)
    # train_sampler = torch.utils.data.sampler.WeightedRandomSampler( train_weights, len(XTrain) )

    # Set up data loaders
    train_dataloader = torch.utils.data.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # train_dataloader = torch.utils.data.DataLoader(train_dataloader, batch_size=TRAIN_BATCH_SIZE,
    #                                                 sampler=train_sampler)
    train_dataloader = torch.utils.data.DataLoader(train_dataloader, batch_size=TRAIN_BATCH_SIZE)


    # ----------------------------------------------------
    # Fine tuning
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    loss_vec = np.zeros(NUM_TRAIN_EPOCHS * (len(train_dataloader) // 10))

    prediction_head = NoteClassificationHead(hidden_size=model.config.hidden_size)
    _ = prediction_head.to(device)

    # optimizer = torch.optim.Adam(list(model.parameters()) + list(prediction_head.parameters()), lr=LEARNING_RATE)

    # Optimizers
    param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    no_decay = []
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
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

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
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



        ## Save at checkpoints
        # if epoch % 10 == 0:
        #     # ----------------------------------------------------
        #     # Save model
        #     model_to_save = model.module if hasattr(model, "module") else model
        #     layer_to_save = prediction_head.module if hasattr(prediction_head, "module") else prediction_head

        #     # Save using the predefined names so that one can load using `from_pretrained`
        #     output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME + str(epoch))
        #     output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME + str(epoch))
        #     output_classification_file = os.path.join(OUTPUT_DIR, PREDICTION_HEAD_NAME + str(epoch))

        #     torch.save(model_to_save.state_dict(), output_model_file)
        #     model_to_save.config.to_json_file(output_config_file)
        #     tokenizer.save_vocabulary(OUTPUT_DIR)

        #     # save weights of the final classification layer
        #     torch.save(layer_to_save.state_dict(), output_classification_file)

        #     # Save loss vector
        #     pickle.dump(loss_vec, open(REPORTS_DIR + f"loss{epoch}.pkl", 'wb'))

        #     # Save loss plot
        #     _ = sns.scatterplot(x=range(len(loss_vec)), y=loss_vec)
        #     _ = plt.title("Clinical BERT Train Loss")
        #     plt.savefig(REPORTS_DIR + f"train_loss{epoch}.eps", format="eps", dpi=1000)
        #     print("Checkpoint epoch: {}".format(epoch))





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

    print(model_to_save.state_dict(), layer_to_save.state_dict())



# ----------------------------------------------------
# ----------------------------------------------------
if args.mode == "test" or args.mode == "train_test":
    # Set path
    BERT_MODEL = OUTPUT_DIR + f"{TASK_NAME}.tar.gz"
    # Load fine-tuned model
    tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)
    processor = BinaryClassificationProcessor()

    # Testing
    # Set test set loaders
    eval_examples = processor.get_dev_examples(PROCESSED_SAVE_DIR)
    eval_features = convert_examples_to_features(eval_examples,
                        labelList, MAX_SEQ_LENGTH, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", EVAL_BATCH_SIZE)

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
    model.load_state_dict( torch.load(OUTPUT_DIR + WEIGHTS_NAME) )
    _ = model.to(device)
    _ = model.eval()
    print("fine-tuned:", model.state_dict())

    prediction_head = NoteClassificationHead(hidden_size=model.config.hidden_size)
    prediction_head.load_state_dict( torch.load(OUTPUT_DIR + PREDICTION_HEAD_NAME) )
    _ = prediction_head.eval()
    _ = prediction_head.to(device)
    print("fine-tuned pred_head:", prediction_head.state_dict())

    # Final setup
    sigmoid_func = nn.Sigmoid()
    prob = np.zeros(len(eval_data))
    # eval_loss, eval_accuracy = 0, 0
    # nb_eval_steps, nb_eval_examples = 0, 0

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
    print("Complete and saved to {}".format(REPORTS_DIR))

    roc = lr_roc_plot(yTest, prob, save_path = REPORTS_DIR + f'roc.eps', plot = True)


