# ----------------------------------------------------
from ED_support_module import *                                # Source required modules and functions
from EDA import EPIC, EPIC_enc, EPIC_CUI, numCols, catCols     # Source datasets from EDA.py


# ----------------------------------------------------
# Variational autoencoder

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
results_dir = 'saved_results/vae'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# Hyper-parameters
feature_size = EPIC_enc.shape[1] - 1
h_dim = 15
z_dim = 10
num_epochs = 200
batch_size = 128
learning_rate = 1e-3

# Prepare taining set
EPIC_enc, cuiCols = TFIDF(EPIC_CUI, EPIC_enc)
y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)
XTrainNormal = XTrain.loc[yTrain == 0]

# Separate the numerical and categorical features
numCols = numCols + list(cuiCols)
XTrainNum = XTrainNormal[numCols]
XTestNum = XTest[numCols]

# PCA on the numerical entries   # 27, 11  # Without PCA: 20, 18
pca = sk.decomposition.SparsePCA(0.95).fit(XTrainNum)
# pca = sk.decomposition.PCA(0.95).fit(XTrainNum)
XTrainNum = pd.DataFrame(pca.transform(XTrainNum))
XTestNum = pd.DataFrame(pca.transform(XTestNum))
XTrainNum.index, XTestNum.index = XTrainNormal.index, XTest.index

# Transform the train set
scaler = sk.preprocessing.MinMaxScaler()
XTrainNum = scaler.fit_transform(XTrainNum)
XTrainNum = pd.DataFrame(XTrainNum)
XTrainNum.index = XTrainNormal.index

# Transform the test set
XTestNum = scaler.transform(XTestNum)
XTestNum = pd.DataFrame(XTestNum)
XTestNum.index = XTest.index

# Construct scaled datasets
XTrainNormal = pd.concat([pd.DataFrame(XTrainNum), pd.DataFrame(XTrainNormal.drop(numCols, axis = 1))],
                         axis = 1, sort = False)
XTrainNormal = np.array(XTrainNormal)
XTest = pd.concat([pd.DataFrame(XTestNum), pd.DataFrame(XTest.drop(numCols, axis = 1))],
                  axis = 1, sort = False)
XTest = np.array(XTest)

trainLoader = torch.utils.data.DataLoader(dataset = torch.from_numpy(XTrainNormal),
                                          batch_size = batch_size,
                                          shuffle = True)


# VAE model
class VAE(nn.Module):
    def __init__(self, feature_size=61, h_dim=45, z_dim=30):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(feature_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, feature_size)
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h = torch.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


model = VAE(feature_size=XTrainNormal.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize vectors to store the loss
recLossVec = np.zeros(num_epochs * len(trainLoader))
klDivVec = np.zeros(num_epochs * len(trainLoader))

# Start training
for epoch in range(num_epochs):
    for i, x in enumerate(trainLoader):
        # Forward pass
        # x = x.to(device).view(-1, feature_size)
        x = x.float()
        x_reconst, mu, log_var = model(x)
        # Compute reconstruction loss and kl divergence
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Store losses
        ind = epoch * len(trainLoader) + i
        recLossVec[ind] = reconst_loss.item()
        klDivVec[ind] = kl_div.item()
        if (i+1) % 10 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, len(trainLoader),
                   round(reconst_loss.item(), 4), round(kl_div.item(), 4)))


# Save results
saveModel(recLossVec, './saved_results/vae/recLossVec')
saveModel(klDivVec, './saved_results/vae/klDivVec')
saveModel(model, './saved_results/vae/vaeModel')


# Plot losses
sns.scatterplot(x = range(len(recLossVec)), y = recLossVec)
plt.show()

sns.scatterplot(x = range(len(klDivVec)), y = klDivVec)
plt.show()


# Prediction
recLossVec = loadModel('./saved_results/vae/recLossVec')
klDivVec = loadModel('./saved_results/vae/klDivVec')
model = loadModel('./saved_results/vae/vaeModel')

testLoader = torch.utils.data.DataLoader(dataset = torch.from_numpy(XTest),
                                          batch_size = 1,
                                          shuffle = False)


testRL = np.zeros(len(XTest))
testKL = np.zeros(len(XTest))
with torch.no_grad():
    for i, data in enumerate(testLoader):
        data = data.float()
        x_reconst, mu, log_var = model(data)
        # Reconstruction loss
        reconst_loss = F.binary_cross_entropy(x_reconst, data, reduction = 'sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        testRL[i] = reconst_loss
        testKL[i] = kl_div
        if (i + 1) % 1000 == 0:
            print("Prediction step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                   .format(i+1, len(testLoader),
                   round(reconst_loss.item(), 4), round(kl_div.item(), 4)))


xVec = np.linspace(1, len(testRL), len(testRL))
_ = sns.scatterplot(x = xVec, y = testRL)
_ = sns.scatterplot(x = xVec[yTest == 1], y = testRL[yTest == 1], color = 'red')
plt.show()

_ = sns.scatterplot(x = xVec, y = testKL)
_ = sns.scatterplot(x = xVec[yTest == 1], y = testKL[yTest == 1], color = 'red')
plt.show()

_ = sns.scatterplot(x = xVec, y = testKL + testRL)
_ = sns.scatterplot(x = xVec[yTest == 1], y = testKL[yTest == 1] + testRL[yTest == 1], color = 'red')
plt.show()


# Set threshold to be the mean of 
def vaePredict(loss_train = None, loss_test = None, batch_size = None, sample_size = 1000, k = 1, percent = 0.1):
    '''
    Make prediction based on the train loss and the test loss.
    Threshold is set to be mu + k * std, where mu and std are computed
    from the last sample_size batches.
    Input : loss_train = loss of the train set (np.array or pd.Series)
            loss_test = loss of the test set (np.array or pd.Series)
            batch_size = batch size used in the training
            sample_size = size of the sample used to set the threshold
            k = parameter for setting the threshold
    '''
    # Set threshold
    # mu = (loss_train[-sample_size:] / batch_size).mean()
    # std = np.sqrt( ( loss_train[-sample_size:] / batch_size ).var() )
    # threshold = mu + k * std
    ## Outlier if loss > threshold
    # yPred = loss_test > threshold
    testLossSorted = np.sort(loss_test)
    threshold = testLossSorted[-int( np.ceil( percent * len(loss_test) ) )]
    yPred = loss_test > threshold
    return(yPred, threshold)


# Plot summary statistics
vaePred, threshold = vaePredict(recLossVec + klDivVec, testKL + testRL, batch_size, k = 0, percent = 0.1)
roc_plot(yTest, vaePred)

vaePred, threshold = vaePredict(klDivVec, testRL, batch_size, k = 0, percent = 0.1)
roc_plot(yTest, vaePred)

