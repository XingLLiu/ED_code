from ED_support_module import *

# ----------------------------------------------------
# EDA pipeline (further preprocessing)
class Preprocess:
    '''
    Preparing datasets for modelling using the preprocessed EPIC dataset.
    '''
    def __init__(self, path, drop_cols='default', after_triage='default'):
        self.path = path
        self.drop_cols = drop_cols
        self.after_triage = after_triage


    def SeparateData(self, EPIC):
        '''
        Takes in full EPIC data with arrival date and CUIs,
        and output three versions of data:
        EPIC, EPIC_CUI, EPIC_arrival
        '''
        notes = ['Notes', 'Provider.Notes', 'Triage.Notes']
        EPIC_CUI = EPIC[notes]
        EPIC = EPIC.drop(notes, axis = 1)
        # Separate MRN and arrival date
        EPIC_arrival = EPIC[['MRN', 'Arrived']]
        EPIC = EPIC.drop(['MRN', 'Arrived'], axis = 1)
        # Convert three cols of notes to list
        for col in notes:
            noteLst = pd.Series( map( lambda note: note[2:-2].split('\', \''), EPIC_CUI[col] ) )
            EPIC_CUI.loc[:, col] = noteLst.values
        # Change 'Disch.Date.Time' and 'Roomed' to categorical
        EPIC['Disch.Date.Time'] = EPIC['Disch.Date.Time'].astype('object')
        EPIC['Roomed'] = EPIC['Roomed'].astype('object')
        # Change 'Will.Return' to binary if present
        if 'Will.Return' in EPIC.columns:
            EPIC['WillReturn'] = EPIC['WillReturn'].astype('object')
        return EPIC, EPIC_CUI, EPIC_arrival


    def CleanColNames(self, EPIC):
        '''
        Takes in EPIC and the same dataset with the
        format of its column names unified.
        '''
        colNames = list(EPIC.columns)
        # Special cases
        indexVec = np.linspace(0, len(colNames) - 1, len(colNames), dtype = 'int')
        ifCC = int(indexVec[EPIC.columns == 'CC'])
        ifFSA = int(indexVec[EPIC.columns == 'FSA'])
        for i in range(len(colNames)):
            name = colNames[i]
            # Change names of the form 'XX XXX'
            if len(name.split(' ')) > 1:
                name = name.replace(' ', '.')
            # Change names of the form 'XxXxx'
            elif len(name.split('.')) == 1:
                nameList = re.findall('[A-Z][a-z]+', name)
                name = '.'.join(nameList)
            colNames[i] = name
        # Assign special cases back
        colNames[ifCC] = 'CC'
        colNames[ifFSA] = 'FSA'
        # Re-assign col names
        EPIC.columns = colNames
        # Print error warning if there is empty colname remaining
        for name in colNames:
            if len(colNames) == 0:
                print('Empty column name warning! Column name assignment may be wrong!')
        return EPIC


    def BinarizeSepsis(self, EPIC):
        '''
        Binarize Primary.Dx for is(1)/not(0) sepsis.
        Mark as sepsis if any of the following shows the
        keyword sepsis/Sepsis:
        Primary.Dx, Diagnosis, Diagnoses.
        Return EPIC with Diagnosis and Diagnoses removed.
        '''
        # Separate diagnoses columns
        diagnosis = EPIC['Diagnosis']
        diagnoses = EPIC['Diagnoses']
        EPIC = EPIC.drop(['Diagnosis', 'Diagnoses'], axis = 1)
        # Check if Primary.Dx contains Sepsis or related classes
        ifSepsis1 = EPIC['Primary.Dx'].str.contains('epsis')
        # Check if Diagnosis contains Sepsis or related classes
        ifSepsis2 = diagnosis.str.contains('epsis')
        # Check if Diagnoses contains Sepsis or related classes
        ifSepsis3 = diagnoses.str.contains('epsis')
        # Lable as sepsis if any of the above contains Sepsis
        ifSepsis = ifSepsis1 | ifSepsis2 | ifSepsis3
        # Convert into binary class
        EPIC.loc[-ifSepsis, 'Primary.Dx'] = 0
        EPIC.loc[ifSepsis, 'Primary.Dx'] = 1
        EPIC['Primary.Dx'] = EPIC['Primary.Dx'].astype('int')
        return EPIC


    def RemoveCols(self, EPIC):
        '''
        Remove features after triage and other
        feature selection.
        Return dataset with the given feature names in
        drop_cols and after_triage removed.
        '''
        # Discard the following features in modelling
        if self.drop_cols == 'default':
            self.drop_cols = ['First.ED.Provider', 'Last.ED.Provider', 'ED.Longest.Attending.ED.Provider',
                'Day.of.Arrival', 'Arrival.Month', 'FSA', 'Name.Of.Walkin', 'Name.Of.Hospital',
                'Admitting.Provider', 'Disch.Date.Time', 'Discharge.Admit.Time',
                'Distance.To.Sick.Kids', 'Distance.To.Walkin', 'Distance.To.Hospital', 'Systolic',
                'Pulse']

        # (Some) features obtained after triage
        if self.after_triage == 'default':
            self.after_triage = ['Lab.Status', 'Rad.Status', 'ED.PIA.Threshold', 'Same.First.And.Last',
                                 'Dispo', 'Size.Of.Treatment.Team', 'Number.Of.Prescriptions',
                                 'Length.Of.Stay.In.Minutes', 'Arrival.to.Room', 'Roomed', 'Will.Return']
        colRem = self.drop_cols + self.after_triage
        EPIC =  EPIC.drop(colRem, axis = 1)
        return EPIC


    def GroupClasses(self, EPIC):
        '''
        Takes in EPIC and group some classes as 
        'Others' in certain features.
        '''
        # Pref.Language: Keep top 4 languages + others
        topLangs = EPIC['Pref.Language'].value_counts().index[:4]
        ifTopLangs = [not language in topLangs for language in EPIC['Pref.Language'].values]
        EPIC['Pref.Language'].loc[ ifTopLangs ] = 'Others'
        # CC: Keep top 19 + others
        topCC = EPIC['CC'].value_counts().index[:49]
        ifTopCC = [not complaint in topCC for complaint in EPIC['CC'].values]
        EPIC['CC'].loc[ ifTopCC ] = 'Others'
        # Arrival method: combine 'Unknown' and 'Other' and keep top 9 + others
        EPIC.loc[EPIC['Arrival.Method'] == 'Unknown', 'Arrival.Method'] = 'Others'
        topMethods = EPIC['Arrival.Method'].value_counts().index[:14]
        ifTopMethods = [not method in topMethods for method in EPIC['Arrival.Method'].values]
        EPIC['Arrival.Method'].loc[ ifTopMethods ] = 'Others'
        return EPIC


    def RemoveOutliers(self, EPIC):
        '''
        Takes in EPIC and output a boolean pd.Series
        indicating which instance is an obvious outlier.
        1 for outliers, 0 for inliers.
        '''
        # Abnormal cases. Suspected to be typos
        # Old patients. All non-Sepsis 
        cond1 = EPIC['Age.at.Visit'] > 40
        # Temperature > 50 or < 30
        cond2 = (EPIC['Temp'] > 50) | (EPIC['Temp'] < 30)
        # Blood pressure > 240
        cond3 = (EPIC['Diastolic'] > 240)
        # # Resp > 300
        cond4 = EPIC['Resp'] > 300
        # Remove these outlisers
        cond = cond1 | cond2 | cond3 | cond4
        return cond

    
    def SimpleInpute(self, EPIC):
        '''
        Takes in EPIC and perform mean-imputation on the
        numerical features, and most-frequent-class imputation
        on the categorical features.
        Output the imputed dataset, categorical column names and
        numerical column names.
        '''
        y = EPIC['Primary.Dx']
        X = EPIC.drop('Primary.Dx', axis = 1)
        numCols = self.which_numerical(X)
        # numCols = X.select_dtypes(include = [np.number]).columns.tolist()
        catCols = [col for col in X.columns if col not in numCols]
        XCat = X.drop(numCols, axis = 1)
        XNum = X.drop(catCols, axis = 1)
        # Simple imputation
        meanImp = sk.impute.SimpleImputer(strategy = 'mean')
        freqImp = sk.impute.SimpleImputer(strategy = 'most_frequent')
        # Impute categorical features with the most frequent class
        freqImp.fit(XCat)
        XCatImp = freqImp.transform(XCat)
        # Impute numerical features with mean
        meanImp.fit(XNum)
        XNumImp = meanImp.transform(XNum)
        EPIC[catCols] = XCatImp
        EPIC[numCols] = XNumImp
        return EPIC, catCols, numCols
    
    
    def DateFillNA(self, EPIC_arrival, how="both"):
        '''
        Fill in the missing values in Arrived with
        the adjacent values.
        how = 1. before: fill in the missing value with
                 the immediately previous one.
              2. after: fill in the missing value with
                 the immediately succeeding one.
              3. both: fill in the missing value with
                 the immediately previous then the
                 succeeding.
        '''
        null_date = EPIC_arrival.index[EPIC_arrival['Arrived'].isnull()]
        null_date = null_date.values
        # Check if no previous or existing instance
        indicator1 = null_date == 0
        # Impute by the proceeding instance if it is the first one
        if (indicator1).any():
            null_date[indicator1] = 2
        EPIC_arrival['Arrived'][null_date] = EPIC_arrival.loc[null_date - 1, 'Arrived']   
        # Impute with the proceeding value if still missing
        null_date = EPIC_arrival.index[EPIC_arrival['Arrived'].isnull()]
        if len(null_date) > 0:
            null_date = null_date.values
            indicator2 = null_date == EPIC_arrival.shape[0]
            # Impute by the previous instance if it is the last one
            if (indicator2).any():
                null_date[indicator2] = EPIC_arrival.shape[0] - 2
            EPIC_arrival['Arrived'][null_date] = EPIC_arrival.loc[null_date + 1, 'Arrived']
        # Print warning if still missing
        if len(EPIC_arrival.index[EPIC_arrival['Arrived'].isnull()]) > 0:
            raise ValueError("Arrival date ('Arrived') still missing after imputation.")
        return EPIC_arrival
    
    
    def TFIDF(self, cui, EPIC):
        '''
        Compute the TF-IDF of the CUIs in cui and append to EPIC.
        Input : cui: dataframe of the triage notes CUIs. Must contain column Triage.Notes
                EPIC: dataframe to which the TF-IDFs are appended. Must have Primary.Dx.
        Output: EPIC: dataframe with the TF-IDFs appended.
                cuiCols: Column names of the CUIs
        '''
        # Find all Sepsis
        ifSepsis = EPIC['Primary.Dx'] == 1
        CUISepsis = EPIC.iloc[ifSepsis.values]
        # Get all unique CUIs
        triageNotes = {}
        for i in CUISepsis.index:
            cuiLst = [cui for cui in CUISepsis.loc[i, 'Triage.Notes']]
            for cui in cuiLst:
                if cui not in triageNotes.keys():
                    triageNotes[cui] = 0
        # For each unique CUI, count the number of documents that contains it
        for notes in EPIC['Triage.Notes']:
            for cui in triageNotes.keys():
                if cui in notes:
                    triageNotes[cui] += 1
        # Create TF-IDF dataframe
        triageDf = pd.DataFrame(index = range(len(EPIC)),
                                columns = range(len(triageNotes)),
                                dtype = 'float')
        triageDf.iloc[:, :] = 0
        triageDf.columns = triageNotes.keys()
        triageDf.index = EPIC.index
        # Compute TF and IDF
        corpusLen = len(EPIC)
        for i in triageDf.index:
            notes = EPIC.loc[i, 'Triage.Notes']
            for cui in notes:
                # Compute TF-IDF if cui is in vocab
                if cui in triageNotes.keys():
                    # TF 
                    tf = sum([term == cui for term in notes]) / len(notes)
                    # IDF 
                    idf = np.log( corpusLen / triageNotes[cui] )
                    # Store TF-IDF
                    triageDf.loc[i, cui] = tf * idf
        # Append to EPIC
        cuiCols = triageDf.columns
        EPIC = pd.concat([EPIC, triageDf], axis = 1, sort = False)
        return EPIC, cuiCols


    def streamline(self):
        # Load data (specifying encoding for stability)
        EPIC = pd.read_csv(self.path, encoding = 'ISO-8859-1')
        # Separate data
        EPIC, EPIC_CUI, EPIC_arrival = self.SeparateData(EPIC)
        # Clean column names
        EPIC = self.CleanColNames(EPIC)
        # Binarize Primary.Dx
        EPIC = self.BinarizeSepsis(EPIC)
        # Feature selection and remove data after triage
        EPIC =self.RemoveCols(EPIC)
        # Group classes in some features
        EPIC = self.GroupClasses(EPIC)
        # Remove obvious outliers
        outliers = self.RemoveOutliers(EPIC)
        EPIC = EPIC.loc[~outliers]
        EPIC_CUI = EPIC_CUI.loc[~outliers]
        EPIC_arrival = EPIC_arrival.loc[~outliers]
        # Remove cases wtih missing arrival date and append
        null_date = self.missing_index(EPIC_arrival, "Arrived")
        EPIC = EPIC.drop(labels = null_date, axis = 0)
        EPIC_CUI = EPIC_CUI.drop(labels = null_date, axis = 0)
        EPIC_arrival = EPIC_arrival.drop(labels = null_date, axis = 0)
        # Simple imputation
        EPIC, catCols, _ = self.SimpleInpute(EPIC)
        # One-hot encode the categorical variables
        EPIC_enc = pd.get_dummies(EPIC, columns = catCols, drop_first = True).copy()
        # # Fill in missing arrival date and append
        # EPIC_arrival = self.DateFillNA(EPIC_arrival)
        # Append arrival date
        EPIC_arrival = pd.concat([EPIC_enc, EPIC_arrival['Arrived'].astype('int')], axis = 1)
        return EPIC, EPIC_enc, EPIC_CUI, EPIC_arrival


    def which_numerical(self, data):
        '''
        Get names of numerical columns.
        Input : data = [DataFrame]
        Output: num_cols = [list] col names of numerical features
        '''
        num_cols = data.select_dtypes(include = [np.number]).columns.tolist()
        # Disch.Date.Time should have been categorical
        if "Disch.Date.Time" in num_cols:
            num_cols.remove("Disch.Date.Time")
        return num_cols


    def missing_index(self, data, col_name):
        '''
        Remove cases with missing values in col_name.
        Input :
                data = [DataFrame] dataset with arrival date in
                        pd.datetime format.
                col_name = [str] name of column in which missing
                            values are identified.
        Output:
                index = [pd.Index] dataset with missing dates removed.
        '''
        # Get dates
        data_col = data[col_name]
        # Get index of missing dates
        return data_col.isnull().loc[data_col.isnull()].index


