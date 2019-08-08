# ----------------------------------------------------
# Command arguments: mode, no. of epochs, batch size, learning rate
from ED_support_module import *                                
from EDA import EPIC, EPIC_enc, EPIC_CUI, numCols, catCols 


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
mode = sys.argv[1]
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


# ----------------------------------------------------
# Hyper-parameters
h_dim = 15
z_dim = 10
num_classes = 2
hyper_params = sys.argv[2:]

if len(hyper_params) == 3:
    num_epochs = int(hyper_params[0])
    batch_size = int(hyper_params[1])
    learning_rate = float(hyper_params[2])
else:
    num_epochs = 200
    batch_size = 128
    learning_rate = 1e-3


# ----------------------------------------------------
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
results_dir = 'saved_results/random_forest/' + mode
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# ----------------------------------------------------
# Prepare taining set
if mode not in ['a', 'b']:
    EPIC_enc, cuiCols = TFIDF(EPIC_CUI, EPIC_enc)


y = EPIC_enc['Primary.Dx']
X = EPIC_enc.drop('Primary.Dx', axis = 1)
XTrain, XTest, yTrain, yTest = sk.model_selection.train_test_split(X, y, test_size=0.25, random_state=27, stratify = y)

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
                                           batch_size = batch_size,
                                           shuffle = False)

                                           
# ----------------------------------------------------
# Logistic regression model
input_size = XTrain.shape[1]
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(trainLoader)
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
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for x in XTestLoader:
        # Retrieve labels
        labels = next(iter(yTestLoader))
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))





# ----------------------------------------------------
# # Hyper-parameters 
# input_size = 784
# num_classes = 10
# num_epochs = 5
# batch_size = 100
# learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

testLoader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(trainLoader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainLoader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testLoader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
