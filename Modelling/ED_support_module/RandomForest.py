from ED_support_module import *


def predict_proba_single(self, x):
    '''
    Output the predicted probability of being of class 1
    only, as opposed to 2 columns for being of class 0 and class 1.
    '''
    return self.predict_proba(x)[:, 1]
    

sk.ensemble.RandomForestClassifier.predict_proba_single = predict_proba_single

