from __future__ import absolute_import, division, print_function
from ED_support_module import *
sys.path.append("../../ClinicalNotePreProcessing")
from extract_dates_script import findDates
import csv
import logging
from tqdm import tqdm, trange
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule


# ----------------------------------------------------
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



# ----------------------------------------------------
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


# ----------------------------------------------------
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


class BertForSepsis(nn.Module):
    """
    Bert model with prediction layer.
    """
    def __init__(self, bert_model, device, hidden_size, dropout_prob=0.4, num_labels=2):
        super(BertForSepsis, self).__init__()
        self.bert = bert_model
        self.device = device
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)
    def forward(self, input_ids, segment_ids, input_mask):
        _, pooled_output = self.bert(input_ids, segment_ids, input_mask)
        pooled_output = self.dropout(pooled_output.to(self.device))
        logits = self.classifier(pooled_output)
        return logits
    def train_model(self, train_loader, criterion, optimizer,
                    gradient_accumulation_steps=1,
                    NUM_GPU=1):
        self.train()
        # Initialize loss vector
        loss_vec = np.zeros( len( train_loader ) // 10 )
        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Get batch
            batch = tuple( t.to( self.device ) for t in batch )
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = self(input_ids, segment_ids, input_mask).to(self.device)
            # Compute loss
            loss = criterion(logits, label_ids)
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            # Adapt for GPU
            if NUM_GPU > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            # Accumulate loss
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            # Back propagate
            loss.backward()
            # Update optimizer
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
            # Store loss
            if (i + 1) % 10 == 0:
                loss_vec[i // 10] = loss.item()
        return loss_vec
    def eval_model(self, test_loader, batch_size, transformation=None):
        self.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Get batch
                batch = tuple( t.to( self.device ) for t in batch )
                input_ids, input_mask, segment_ids, label_ids = batch
                # Evaluate loss
                with torch.no_grad():
                    logits = self(input_ids, segment_ids, input_mask).to(self.device)
                    # Evaluation metric
                    logits = logits.detach().cpu().numpy()
                    logits = torch.from_numpy(logits[:, 1])
                    label_ids = label_ids.to('cpu').numpy()
                # Store predicted probabilities
                if i == 0:
                    output = logits
                else:
                    output = np.append(output, logits, axis = 0)
        return output


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


# Function that fills in missing text by CC
def fill_missing_text(EPIC, EPIC_original, notesCols):
    '''
    Fill in missing notes by CC
    '''
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
    return EPIC


def create_bert_data(x_data, y_data, save_path=None):
    '''
    Generate data in the format required by BERT.
    Input :
            x_data = [DataFrame or array] text data
            y_data = [DataFrame or array] lables
            save_path = [str] if not None, data is saved to save_path. Must
                        end with .tsv.
    Output: 
            data = [DataFrame] data in the BERT format.
    '''
    data = pd.DataFrame({
             'id': range(x_data.shape[0]),
             'label': y_data,
             'alpha': ['a'] * x_data.shape[0],
             'text': x_data
    })
    if save_path is not None:
        data.to_csv(save_path, sep='\t', index=False, header=False)
    return data


def feature_to_loader(train_features, batch_size):
    '''
    Takes in the train features from BERT model and prepares a DataLoader from it.
    
    Input :
            train_features = train features returned from conver_example_to_features.
            batch_size = [int] batch size of the data loader.
    Output: 
            [object] data loader.
    '''
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype = torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype = torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype = torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype = torch.long)
    # Create data loader
    train_dataloader = torch.utils.data.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_dataloader = torch.utils.data.DataLoader(train_dataloader, batch_size = batch_size)
    return train_dataloader




def save_bert(prediction_model, bert_model, tokenizer, OUTPUT_DIR, WEIGHTS_NAME, CONFIG_NAME,
                entire_model_name="entire_model.bin"):
    '''
    Save BERT with prediction head layer.
    '''
    model_to_save = bert_model.module if hasattr(bert_model, "module") else bert_model
    entire_model_to_save = prediction_model.module if hasattr(prediction_model, "module") else prediction_model
    # Save using the predefined names so that one can load using `from_pretrained`
    output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
    output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)
    output_entire_file = os.path.join(OUTPUT_DIR, entire_model_name)
    # Save
    torch.save(entire_model_to_save.state_dict(), output_entire_file)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(OUTPUT_DIR)
