from ED_support_module import *

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




