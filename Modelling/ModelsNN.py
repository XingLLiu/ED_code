# ----------------------------------------------------
# Up to seven inputs:
# 1. mode
# 2. random seed
# 3. dynamic train-test splitting (True/False)
# 4. no. of epochs
# 5. batch size
# 6. learning rate
# 7. class weight
# 8. dropout probability
#
# To run: python ModelsNN.py b 27 "True" 4000 128 1e-3 3000 1 0.1
#         python ModelsNN.py b 27 "" 50000 256 1e-3 1500 15 0.1
# ----------------------------------------------------
# Command arguments: mode, no. of epochs, batch size, learning rate
from ED_support_module import *
from EDA import EPIC, EPIC_enc, EPIC_CUI, EPIC_arrival, numCols, catCols


# ----------------------------------------------------
# Set arguments
def setup_parser():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--mode",
                        default="a",
                        type=str,
                        required=True,
                        help="The mode to be used.")
    return parser
    

# ----------------------------------------------------
# Choose preprocessing pipeline
'''
Choose mode:
a -- No PCA, no TF-IDF
b -- PCA, no TF-IDF
c -- No PCA, TF-IDF
d -- PCA, but not on TF-IDF
e -- PCA, TF-IDF
f -- Sparse PCA, TF-IDF
05/08 c is the best
'''
try:
    mode = sys.argv[1]
except:
    mode = 'a'


if mode == 'a':
    suffix = 'NoPCA_noTF-IDF'
elif mode == 'b':
    suffix = 'PCA_noTF-IDF'
elif mode == 'c':
    suffix = 'NoPCA_TF-IDF'
elif mode == 'd':
    suffix = 'PCA_but_not_on_TF-IDF'
elif mode == 'e':
    suffix = 'PCA_TF-IDF'
elif mode == 'f':
    suffix = 'Sparse_PCA_TF-IDF'
else:
    print('Invalid mode')
    quit()


# Random seed
try:
    seed = int(sys.argv[2])
except:
    seed = 27


# If split dataset by arrival time
try:
    useTime = bool(sys.argv[3])
except:
    useTime = False


# ----------------------------------------------------
# Hyper-parameters
num_classes = 2
hyper_params = sys.argv[4:]

if len(hyper_params) == 6:
    num_epochs = int(hyper_params[0])
    batch_size = int(hyper_params[1])
    learning_rate = float(hyper_params[2])
    weight = int(hyper_params[3])
    sample_weight = int(hyper_params[4])
    drop_prob = float(hyper_params[5])
else:
    num_epochs = 500
    batch_size = 128
    learning_rate = 1e-3
    weight = 1000
    sample_weight = 1
    drop_prob = 0


# ----------------------------------------------------
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
results_path = 'saved_results/neural_net/' + mode
plot_path = '/'.join(os.getcwd().split('/')[:3]) + '/Pictures/neural_net/'
dynamic_plot_path = plot_path + 'dynamic/'

if not os.path.exists(results_path):
    os.makedirs(results_path)


if not os.path.exists(plot_path):
    os.makedirs(plot_path)


if not os.path.exists(dynamic_plot_path):
    os.makedirs(dynamic_plot_path)


# ----------------------------------------------------
# NN model
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
# Prepare taining set
if mode not in ['a', 'b']:
    EPIC_enc, cuiCols = TFIDF(EPIC_CUI, EPIC_enc)


# Split using stratified sampling or arrival time
# if not useTime:

# Stratified splitting
# Separate input features and target
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25,
                                random_state=seed, stratify=y)
XTrain, XValid, yTrain, yValid = sk.model_selection.train_test_split(XTrain, yTrain, test_size=0.15,
                                random_state=seed, stratify=yTrain)

# else:
#     XTrain, XTest, yTrain, yTest = time_split(EPIC_arrival, threshold = 201904)
#     print("Train size: {}. Test size: {}".format(len(yTrain), len(yTest)))


# Separate the numerical and categorical features
if mode in ['c', 'e', 'f']:
    numCols = numCols + list(cuiCols)


XTrainNum = XTrain[numCols]
XTestNum = XTest[numCols]

# PCA on the numerical entries   # 27, 11  # Without PCA: 20, 18
if mode in ['b', 'd', 'e', 'f']:
    if mode in ['f']:
        # Sparse PCA 
        pca = sk.decomposition.SparsePCA(int(np.ceil(XTrainNum.shape[1]/2))).fit(XTrainNum)
    elif mode in ['b', 'd', 'e']:
        pca = sk.decomposition.PCA(0.95).fit(XTrainNum)
    XTrainNum = pd.DataFrame(pca.transform(XTrainNum))
    XTestNum = pd.DataFrame(pca.transform(XTestNum))
    XTrainNum.index, XTestNum.index = XTrain.index, XTest.index


# ----------------------------------------------------
if not useTime:
    print("Train and test with stratified sampling...")

    # Construct weight vectors
    train_weights = np.array(sample_weight * yTrain + 1 - yTrain)
    trainSampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, batch_size)

    test_weights = np.array(weight * yTest + 1 - yTest)
    testSampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, batch_size)

    valid_weights = np.array(weight * yValid + 1 - yValid)
    validSampler = torch.utils.data.sampler.WeightedRandomSampler(valid_weights, batch_size)

    # Construct data loaders
    trainLoader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTrain, yTrain], axis = 1)),
                                                batch_size = batch_size,
                                                shuffle = False,
                                                sampler = trainSampler)
    testLoader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTest, yTest], axis = 1)),
                                                batch_size = len(yTest),
                                                shuffle = False)
    validLoader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XValid, yValid], axis = 1)),
                                                batch_size = len(yValid),
                                                shuffle = False)

    # Neural net model
    input_size = XTrain.shape[1]
    model = NeuralNet(input_size = input_size, drop_prob = drop_prob).to(device)

    # Loss and optimizer
    # nn.CrossEntropyLoss() computes softmax internally
    criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, weight]))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    # total_step = len(trainLoader)
    # lossVec = np.zeros( num_epochs * (total_step//100) )
    trainLossVec = np.zeros(num_epochs)
    validLossVec = np.zeros(num_epochs)
    for epoch in trange(num_epochs):
        model.train()
        for i, x in enumerate(trainLoader):
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

        model.eval()
        transform = nn.Sigmoid()
        with torch.no_grad():
            correct = 0
            total = 0
            for x in validLoader:
                # Retrieve design matrix and labels
                labels = x[:, -1].long()
                x = x[:, :(-1)].float()
                # Prediction
                outputs = model(x)
                loss_valid = criterion(outputs, labels)
                # Probability of belonging to class 1
                prob = transform(outputs)[:, 1]
                _, yPred = torch.max(outputs.data, 1)

        trainLossVec[epoch] = loss.item()
        validLossVec[epoch] = loss_valid.item()
        # print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


    # Plot losses
    _ = sns.scatterplot(x = range(len(trainLossVec)), y = trainLossVec, label = 'Train')
    _ = sns.scatterplot(x = range(len(validLossVec)), y = validLossVec, label = 'Validation')
    _ = plt.legend()
    plt.savefig(plot_path + 'train_valid_loss.eps', format='eps', dpi=1000)
    plt.show()


    # ----------------------------------------------------
    # Test the model
    model.eval()
    transform = nn.Sigmoid()
    with torch.no_grad():
        correct = 0
        total = 0
        for x in testLoader:
            # Retrieve design matrix
            x = x[:, :(-1)].float()
            # Prediction
            outputs = model(x)
            # Probability of belonging to class 1
            prob = transform(outputs)[:, 1]
            _, yPred = torch.max(outputs.data, 1)


    # Plot results        
    roc_plot(yTest, yPred, save_path = plot_path + 'roc1.eps')
    nnRoc = lr_roc_plot(yTest, prob, save_path = plot_path + 'roc2.eps')

    nnTpr = nnRoc['TPR']
    nnFpr = nnRoc['FPR']
    print( '\nWith TNR:{}, TPR:{}'.format( round( 1 - nnFpr[5], 4), round(nnTpr[5], 4) ) )


# ----------------------------------------------------
# Train and test the model dynamically
else:
    print('Dynamically evaluate the model.')

    # Time span (3 months of data to up-to-date month - 1)
    timeSpan = [201807, 201808, 201809, 201810, 201811, 201812, 201901, 201902,
                201903, 201904, 201905, 201906]
    for j, month in enumerate(timeSpan[2:]):
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
        # Neural net model
        input_size = XTrain.shape[1]
        model = NeuralNet(input_size = input_size, drop_prob = drop_prob).to(device)
        # Loss and optimizer
        # nn.CrossEntropyLoss() computes softmax internally
        criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, weight]))
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # Train the model
        lossVec = np.zeros(num_epochs)
        for epoch in trange(num_epochs):
            for i, x in enumerate(trainLoader):
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
        print("Training for data upto {} completed.".format(month))
        # ----------------------------------------------------
        # Test the model
        model.eval()
        transform = nn.Sigmoid()
        with torch.no_grad():
            correct = 0
            total = 0
            for x in testLoader:
                # Retrieve design matrix and labels
                labels = x[:, -1].long()
                x = x[:, :(-1)].float()
                # Prediction
                outputs = model(x)
                # Probability of belonging to class 1
                prob = transform(outputs)[:, 1].detach()

        # Save results
        month_pred = timeSpan[j + 3]
        nnRoc = lr_roc_plot(yTest, prob, save_path = dynamic_plot_path + f'roc2_{month_pred}.eps', plot = False)
        summary = dynamic_summary(pd.DataFrame(nnRoc), yTest.sum(), len(yTest) - yTest.sum())
        summary.to_csv(dynamic_plot_path + f'summary_{month_pred}.csv', index=False)
        print('Completed prediction for {} \n'.format(month_pred))


