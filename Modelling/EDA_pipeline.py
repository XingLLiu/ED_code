from ED_support_module import *

# ----------------------------------------------------
# EDA pipeline (further preprocessing)
class EDA:
    def __init__(self, path):
        self.path = path
    def load_data(self):
        EPIC = pd.read_csv(self.path, encoding = 'ISO-8859-1')
        return(EPIC)
    def separate_data(self, EPIC):
        '''
        Takes in full EPIC data with arrival date and CUIs,
        and output three versions of data:
        EPIC, EPIC_arrival, EPIC_CUI
        '''
        notes = ['Notes', 'Provider.Notes', 'Triage.Notes']
        EPIC_CUI = EPIC[notes]
        EPIC = EPIC.drop(notes, axis = 1)
        # Separate MRN and arrival date
        EPIC_arrival = EPIC[['MRN', 'Arrived']]
        EPIC = EPIC.drop(['MRN', 'Arrived'], axis = 1)
        # Separate diagnoses columns
        diagnosis = EPIC['Diagnosis']
        diagnoses = EPIC['Diagnoses']
        EPIC = EPIC.drop(['Diagnosis', 'Diagnoses'], axis = 1)