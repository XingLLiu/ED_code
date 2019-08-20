from ED_support_module import *
from EDA_pipeline import EPICPreprocess

# ----------------------------------------------------
# Prepare train/test dataloaders
class PrepareDataLoaders(EPICPreprocess):
    def __init__(self,
                 path,
                 drop_cols='default',
                 after_triage='default',
                 mode):
        super().__init__(path, drop_cols, after_triage)
    

    def append_TFIDF(self, EPIC_CUI, EPIC):
        '''
        Append the TFIDF calculated from the CUIs of
        all septic patients. Triage.Notes must be
        present in EPIC_CUI
        Return EPIC with the TFIDF appended.
        '''
        # Find all Sepsis
        ifSepsis = EPIC['Primary.Dx'] == 1
        CUISepsis = EPIC_CUI.iloc[ifSepsis.values]
        # Get all unique CUIs
        triageNotes = {}
        for i in CUISepsis.index:
            cuiLst = [cui for cui in CUISepsis.loc[i, 'Triage.Notes']]
            for cui in cuiLst:
                if cui not in triageNotes.keys():
                    triageNotes[cui] = 0
        # For each unique CUI, count the number of documents that contains it
        for notes in EPIC_CUI['Triage.Notes']:
            for cui in triageNotes.keys():
                if cui in notes:
                    triageNotes[cui] += 1
        # Create TF-IDF dataframe
        triageDf = pd.DataFrame(index = range(len(EPIC_CUI)),
                                columns = range(len(triageNotes)),
                                dtype = 'float')
        triageDf.iloc[:, :] = 0
        triageDf.columns = triageNotes.keys()
        triageDf.index = EPIC.index
        # Compute TF and IDF
        corpusLen = len(EPIC_CUI)
        for i in triageDf.index:
            notes = EPIC_CUI.loc[i, 'Triage.Notes']
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
        print('Complete')
        cuiCols = triageDf.columns
        EPIC = pd.concat([EPIC, triageDf], axis = 1, sort = False)
        return EPIC, cuiCols

    





def prepare_EPIC_loaders():
    # Construct train/test data
    if mode not in ['a', 'b']:
        EPIC_enc, cuiCols = TFIDF(EPIC_CUI, EPIC_enc)
        EPIC_arrival = pd.concat([EPIC_enc, EPIC_arrival['Arrived']], axis = 1)

    XTrain, XTest, yTrain, yTest = time_split(EPIC_arrival, threshold = month, dynamic = True)

    print('Training for data before {} ...'.format(month))
    print('Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}].'
            .format( len(yTrain), len(yTest), yTrain.sum(), yTest.sum() ))

    # Separate the numerical and categorical features
    if mode in ['c', 'e', 'f']:
        numCols = numCols + list(cuiCols)
    XTrainNum = XTrain[numCols]
    XTestNum = XTest[numCols]
    # PCA on the numerical entries
    if mode in ['b', 'd', 'e', 'f']:
        if mode in ['f']:
            # Sparse PCA
            pca = sk.decomposition.SparsePCA(int(np.ceil(XTrainNum.shape[1]/2))).fit(XTrainNum)
        elif mode in ['b', 'd', 'e']:
            pca = sk.decomposition.PCA(0.95).fit(XTrainNum)

        # Transfered numerical columns
        XTrainNum = pd.DataFrame(pca.transform(XTrainNum))
        XTestNum = pd.DataFrame(pca.transform(XTestNum))
        XTrainNum.index, XTestNum.index = XTrain.index, XTest.index
    # Form transferred data
    XTrain[numCols] = XTrainNum
    XTest[numCols] = XTestNum

    # Construct data loaders
    trainLoader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTrain, yTrain], axis = 1)),
                                                batch_size = batch_size,
                                                shuffle = False)
    testLoader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTest, yTest], axis = 1)),
                                                batch_size = len(yTest),
                                                shuffle = False)





# ----------------------------------------------------
# Dynamic prediction pipeline
class DynamicPred():
    def __init__(self, time_span, data):
        self.time_span = time_span
        self.data = data

    def 
        # Construct train/test data
        if mode not in ['a', 'b']:
            EPIC_enc, cuiCols = TFIDF(EPIC_CUI, EPIC_enc)
            EPIC_arrival = pd.concat([EPIC_enc, EPIC_arrival['Arrived']], axis = 1)

        XTrain, XTest, yTrain, yTest = time_split(EPIC_arrival, threshold = month, dynamic = True)

        print('Training for data before {} ...'.format(month))
        print('Train size: {}. Test size: {}. Sepsis cases in [train, test]: [{}, {}].'
                .format( len(yTrain), len(yTest), yTrain.sum(), yTest.sum() ))

        # Separate the numerical and categorical features
        if mode in ['c', 'e', 'f']:
            numCols = numCols + list(cuiCols)
        XTrainNum = XTrain[numCols]
        XTestNum = XTest[numCols]
        # PCA on the numerical entries
        if mode in ['b', 'd', 'e', 'f']:
            if mode in ['f']:
                # Sparse PCA
                pca = sk.decomposition.SparsePCA(int(np.ceil(XTrainNum.shape[1]/2))).fit(XTrainNum)
            elif mode in ['b', 'd', 'e']:
                pca = sk.decomposition.PCA(0.95).fit(XTrainNum)

            # Transfered numerical columns
            XTrainNum = pd.DataFrame(pca.transform(XTrainNum))
            XTestNum = pd.DataFrame(pca.transform(XTestNum))
            XTrainNum.index, XTestNum.index = XTrain.index, XTest.index

        # Construct data loaders
        trainLoader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTrain, yTrain], axis = 1)),
                                                  batch_size = batch_size,
                                                  shuffle = False)
        testLoader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTest, yTest], axis = 1)),
                                                 batch_size = len(yTest),
                                                 shuffle = False)
 

# ----------------------------------------------------
# Build a customized model
class NeuralNet(nn.Module):
    def __init__(self, input_size=61, num_classes=2, drop_prob=0):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(input_size, num_classes)
        self.dp_layer = nn.Dropout(drop_prob)
    def forward(self, x):
        h = self.dp_layer(x)
        h = self.fc1(h)
        h = self.ac1(h)
        return self.fc2(h)


# ----------------------------------------------------
# Prediction with stratified train/test split
# # Loss and optimizer
# criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, weight]))
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# model = NeuralNet(input_size = input_size, drop_prob = drop_prob).to(device)
def Train(model, criterion, optimizer, train_loader,
          num_epochs, plot_path):
    '''
    Input : model = instantiated model class.
            criterion = instantiated criterion class
            optimizer = instantiated optimizer class
            train_loader = Dataloader of the trainset. The last column must
                           be the labels.
            plot_path = path to save the ROC plot.
    '''
    # Feed to GPUs if exists
    device = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
    n_gpu = torch.cuda.device_count()
    _ = model.to(device)
    if n_gpu > 1:

    # Train the model
    lossVec = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        for i, x in enumerate(train_loader):
            # Retrieve design matrix and labels
            labels = x[:, -1].long()
            x = x[:, :(-1)].float()
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lossVec[epoch] = loss
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    # Plot losses
    _ = sns.scatterplot(x = range(len(lossVec)), y = lossVec)
    plt.savefig(plot_path + 'losses.eps', format='eps', dpi=1000)
    plt.show()




