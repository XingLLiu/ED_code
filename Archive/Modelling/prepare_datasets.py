from __future__ import absolute_import, division, print_function
import sys
sys.path.append("../ClinicalNotePreProcessing")
from ED_support_module import *
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


def prepare_data(data_path, mode, random_seed, validation_size = 0, clean_notes=True):
    RANDOM_SEED = random_seed
    MODE = mode
    # MODE = "a"
    VALID_SIZE = validation_size
    # Path set-up
    # FIG_PATH = "../../../results/stacked_model/"
    FIG_PATH = data_path
    RAW_TEXT_PATH = "../../data/EPIC_DATA/EPIC.csv"
    DATA_PATH = "../../data/EPIC_DATA/preprocessed_EPIC_with_dates_and_notes.csv"
    FIG_ROOT_PATH = FIG_PATH + f"dynamic_{MODE}/"
    RAW_SAVE_DIR = FIG_PATH + "Raw_Notes/"
    CLEAN_NOTES = clean_notes
    # Create folder if not already exist
    if not os.path.exists(FIG_PATH):
        os.makedirs(FIG_PATH)
    # ----------------------------------------------------
    # ========= 1.i. Further preprocessing =========
    # Create folder if not already exist
    if not os.path.exists(RAW_SAVE_DIR):
        os.makedirs(RAW_SAVE_DIR)
    preprocessor = EPICPreprocess.Preprocess(DATA_PATH)
    EPIC, EPIC_enc, EPIC_CUI, EPIC_arrival = preprocessor.streamline()
    # Get numerical columns (for later transformation)
    num_cols = preprocessor.which_numerical(EPIC)
    num_cols.remove("Primary.Dx")
    # Get time span
    time_span = EPIC_arrival['Arrived'].unique().tolist()
    # ========= 1.ii. Clean text data =========
    # Text data
    EPIC_original = pd.read_csv(RAW_TEXT_PATH, encoding = 'ISO-8859-1')
    preprocessor = EPICPreprocess.Preprocess(path = RAW_TEXT_PATH)
    EPIC_original = preprocessor.BinarizeSepsis(EPIC_original)
    # Only keep text columns and label
    notes_cols = ['Note.Data_ED.Triage.Notes']
    EPIC = EPIC_original[['Primary.Dx'] + notes_cols]
    if CLEAN_NOTES:
        # Clean texts
        EPIC_text = clean_epic_notes(EPIC = EPIC,
                                    EPIC_cc = EPIC_original,
                                    notes_cols = notes_cols,
                                    data_path = DATA_PATH,
                                    save_path = RAW_SAVE_DIR,
                                    save_index = True)
        print("Cleaned text saved to {}".format(RAW_SAVE_DIR))
    else:
        # Load data
        print("Loading cleaned text from {}".format(RAW_SAVE_DIR))
        EPIC_text = pd.read_csv(RAW_SAVE_DIR + "EPIC.csv")
        # Assign index back
        EPIC_text.index = EPIC_text.iloc[:, 0]
        EPIC_text = EPIC_text.drop( EPIC_text.columns[0], axis = 1 )
        time_span = pickle.load( open( RAW_SAVE_DIR + "time_span", "rb" ) )
    discrepancy = (EPIC_enc.index != EPIC_text.index).sum()
    if discrepancy != 0:
        raise Warning("EPIC numerics and text data do not match! Number of unmatched cases: {}"
                        .format(discrepancy))
    # ========= 1.iii. Prepare train/test/validation sets =========
    # Splitting data by month
    for j, time in enumerate(time_span[2:-1]):
        # Month to be predicted
        time_pred = time_span[j + 3]
        # Create folder if not already exist
        DYNAMIC_PATH = FIG_ROOT_PATH + f"{time_pred}/"
        NUMERICS_DATA_PATH = DYNAMIC_PATH + "numerical_data/"
        # Create BERT folder if not already exist
        PROCESSED_NOTES_DIR = DYNAMIC_PATH + "Processed_Texts/"
        for path in [DYNAMIC_PATH, NUMERICS_DATA_PATH, PROCESSED_NOTES_DIR]:
            if not os.path.exists(path):
                os.makedirs(path)
        # Valid set for the first 3 months
        if j == 0:
            # Prepare train/test/valid sets
            # Not prepare validation set if required
            if VALID_SIZE == 0:
                XTrain, XTest, yTrain, yTest= splitter(EPIC_arrival,
                                                        num_cols,
                                                        MODE,
                                                        time_threshold = time,
                                                        test_size = None,
                                                        EPIC_CUI = EPIC_CUI,
                                                        seed = RANDOM_SEED)
                print("Saving data up to {} ...".format(time))
                print( "Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}]."
                            .format( yTrain.shape, yTest.shape, yTrain.sum(), yTest.sum() ) )
            else:
                XTrain, XTest, XValid, yTrain, yTest, yValid= splitter(EPIC_arrival,
                                                                    num_cols,
                                                                    MODE,
                                                                    time_threshold = time,
                                                                    test_size = None,
                                                                    valid_size = VALID_SIZE,
                                                                    EPIC_CUI = EPIC_CUI,
                                                                    seed = RANDOM_SEED)
                print("Saving data up to {} ...".format(time))
                print( "Train size: {}. Test size: {}. Validation size: {}. Sepsis cases in [train, test, valid]: [{}, {}, {}]."
                            .format( yTrain.shape, yTest.shape, len(yValid), yTrain.sum(), yTest.sum(), yValid.sum() ) )
                # Get validation index
                valid_index = XValid.index
                # Get text data
                XValidText = EPIC_text.loc[valid_index, :]
                valid_bert = create_bert_data(x_data = XValidText["Note.Data_ED.Triage.Notes"],
                                            y_data = yValid,
                                            save_path = PROCESSED_NOTES_DIR + "valid.tsv")
                # Save numerics data
                XValid.to_csv(NUMERICS_DATA_PATH + "x_valid.csv", index = False)
                yValid.to_csv(NUMERICS_DATA_PATH + "y_valid.csv", index = False, header = True)
                # Labels for the text set
                yTrainText = yTrain
        else:
            XTrain, XTest, yTrain, yTest= splitter(EPIC_arrival,
                                                    num_cols,
                                                    MODE,
                                                    time_threshold = time,
                                                    test_size = None,
                                                    EPIC_CUI = EPIC_CUI,
                                                    seed = RANDOM_SEED)
            print("Saving data up to {} ...".format(time))
            print( "Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}]."
                        .format( yTrain.shape, yTest.shape, yTrain.sum(), yTest.sum() ) )
            # Set text train data to the previous month
            XTrainText = XTrainTextOld
            yTrainText = yTrainTextOld
        # Save train and test sets
        train_index = XTrain.index
        test_index = XTest.index
        XTrainText = EPIC_text.loc[train_index, :]
        XTestText = EPIC_text.loc[test_index, :]
        # Save text data
        train_bert = create_bert_data(x_data = XTrainText["Note.Data_ED.Triage.Notes"],
                                        y_data = yTrainText,
                                        save_path = PROCESSED_NOTES_DIR + "train.tsv")
        test_bert = create_bert_data(x_data = XTestText["Note.Data_ED.Triage.Notes"],
                                        y_data = yTest,
                                        save_path = PROCESSED_NOTES_DIR + "dev.tsv")
        # Save numerics data
        XTrain.to_csv(NUMERICS_DATA_PATH + "x_train.csv", index = False)
        yTrain.to_csv(NUMERICS_DATA_PATH + "y_train.csv", index = False, header = True)
        XTest.to_csv(NUMERICS_DATA_PATH + "x_test.csv", index = False)
        yTest.to_csv(NUMERICS_DATA_PATH + "y_test.csv", index = False, header = True)
        # Only store text set of the previous month to save time in training the BERT
        XTrainTextOld = XTestText
        yTrainTextOld = yTest




# Prepare datasets
prepare_data(data_path = "../../results/data_by_month/",
                mode = "e",
                random_seed = 27,
                validation_size = 0,
                clean_notes = False)

