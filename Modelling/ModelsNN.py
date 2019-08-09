# ----------------------------------------------------
# Up to seven inputs:
# 1. mode
# 2. random seed
# 3. train-test splitting method (True/False)
# 4. no. of epochs
# 5. batch size
# 6. learning rate
# 7. class weight
# ----------------------------------------------------
# Command arguments: mode, no. of epochs, batch size, learning rate
from ED_support_module import *                                
from EDA import EPIC, EPIC_enc, EPIC_CUI, EPIC_arrival, numCols, catCols 


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

if len(hyper_params) == 5:
    num_epochs = int(hyper_params[0])
    batch_size = int(hyper_params[1])
    learning_rate = float(hyper_params[2])
    weight = int(hyper_params[3])
    drop_prob = float(hyper_params[4])
else:
    num_epochs = 500
    batch_size = 128
    learning_rate = 1e-3
    weight = 1000
    drop_prob = 0


# ----------------------------------------------------
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
results_path = 'saved_results/neural_net/' + mode
plot_path = '/'.join(os.getcwd().split('/')[:3]) + '/Pictures/neural_net/'
if not os.path.exists(results_path):
    os.makedirs(results_path)


if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    

# ----------------------------------------------------
# NN model
class NeuralNet(nn.Module):
    def __init__(self, input_size=61, num_classes=2, drop_prob=0):
        super(NeuralNet, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.dp_layer = nn.Dropout(drop_prob)
    def forward(self, x):
        h = self.dp_layer(x)
        return self.fc(h)
    

# ----------------------------------------------------
# Prepare taining set
if mode not in ['a', 'b']:
    EPIC_enc, cuiCols = TFIDF(EPIC_CUI, EPIC_enc)


# Split using stratified sampling or arrival time
if not useTime:
    # Stratified splitting
    # Separate input features and target
    y = EPIC_enc['Primary.Dx']
    X = EPIC_enc.drop('Primary.Dx', axis = 1)
    XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25,
                                   random_state=seed, stratify=y)
else:
    XTrain, XTest, yTrain, yTest = time_split(EPIC_arrival, threshold = 201904)


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


# Construct data loaders
trainLoader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTrain, yTrain], axis = 1)),
                                           batch_size = batch_size,
                                           shuffle = False)
testLoader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([XTest, yTest], axis = 1)),
                                           batch_size = len(yTest),
                                           shuffle = False)


# ----------------------------------------------------
# Neural net model
input_size = XTrain.shape[1]
model = NeuralNet(input_size = input_size, drop_prob = drop_prob).to(device)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1, weight]))
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(trainLoader)
# lossVec = np.zeros( num_epochs * (total_step//100) )
lossVec = np.zeros(num_epochs)
for epoch in range(num_epochs):
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
        # if (i+1) % 100 == 0:
        #     # Store losses
        #     ind = epoch * (total_step//100) + (i + 1)//100 - 1
        #     lossVec[ind] = loss
        #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
        #            .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    lossVec[epoch] = loss
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


# Plot losses
_ = sns.scatterplot(x = range(len(lossVec)), y = lossVec)
plt.savefig(plot_path + 'losses.eps', format='eps', dpi=1000)
plt.show()


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
        prob = transform(outputs)[:, 1]
        _, yPred = torch.max(outputs.data, 1)


# Plot results        
roc_plot(yTest, yPred, save_path = plot_path + 'roc1.eps')
nnRoc = lr_roc_plot(yTest, prob, save_path = plot_path + 'roc2.eps')

nnTpr = nnRoc['tpr']
nnFpr = nnRoc['fpr']
print( '\nWith TNR:{}, TPR:{}'.format( round( 1 - nnFpr[5], 4), round(nnTpr[5], 4) ) )



