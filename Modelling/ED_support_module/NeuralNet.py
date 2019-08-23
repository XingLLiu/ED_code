from ED_support_module import *

class NeuralNet(nn.Module):
    def __init__(self, device, input_size=61, hidden_size=500, num_classes=2, drop_prob=0):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.ac1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.ac2 = nn.LeakyReLU()
        self.classification = nn.Linear(hidden_size, num_classes)
        self.dp_layer1 = nn.Dropout(drop_prob)
        self.dp_layer2 = nn.Dropout(drop_prob)
        self.device = device
        nn.init.xavier_normal_(self.fc1.weight)
    def forward(self, x):
        h = self.dp_layer1(x)
        h = self.fc1(h)
        h = self.ac1(h)
        h = self.fc2(h)
        h = self.ac2(h)
        h = self.dp_layer2(h)
        return self.classification(h)
    def train_model(self, train_loader, criterion, optimizer):
        '''
        Train and back-propagate the neural network model. Note that
        this is different from the built-in method self.train, which
        sets the model to train mode.

        Model will be set to evaluation mode internally.

        Input : train_loader = [DataLoader] training set. The
                               last column must be the response.
                criterion = [Function] tensor function for evaluatin
                            the loss.
                optimizer = [Function] tensor optimizer function.
                device = [object] cuda or cpu
        Output: loss
        '''
        self.train()
        for i, x in enumerate(train_loader):
            x = x.to(self.device)
            # Retrieve design matrix and labels
            labels = x[:, -1].long()
            x = x[:, :(-1)].float()
            # Forward pass
            outputs = self(x)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss
    def eval_model(self, test_loader, transformation=None):
        '''
        Evaluate the neural network model. Only makes sense if
        test_loader contains all test data. Note that this is
        different from the built-in method self.eval, which
        sets the model to train mode.
        
        Model will be set to evaluation mode internally.

        Input :
                train_loader = [DataLoader] training set. Must not
                                contain the response.
                transformation = [Function] function for evaluatin
                                 transforming the output. If not given,
                                 raw output is return.
        Output: 
                outputs = output from the model (after transformation).
        '''
        self.eval()
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(self.device)
                # Retrieve design matrix
                x = x.float()
                # Prediction
                outputs = self(x)
                if transformation is not None:
                    # Probability of belonging to class 1
                    outputs = transformation(outputs).detach()
                # Append probabilities
                if i == 0:
                    outputs_vec = np.array(outputs.cpu())
                else:
                    outputs_vec = np.append(outputs_vec,
                                            np.array(outputs),
                                            axis = 0)
        return outputs_vec
    def predict_proba_single(self, x_data):
        '''
        Transform x_data into dataloader and return predicted scores
        for being of class 1.
        Input :
                x_data = [DataFrame or array] train set. Must not contain
                         the response.
        Output:
                pred_prob = [array] predicted probability for being
                            of class 1.
        '''
        test_loader = torch.utils.data.DataLoader(dataset = np.array(x_data),
                                                batch_size = len(x_data),
                                                shuffle = False)
        return self.eval_model(test_loader, transformation=None)[:, 1]
        
    def fit(self, x_data, y_data, num_epochs, batch_size, optimizer, criterion):
        '''
        Fit the model on x_data and y_data
        Input :
                x_data = [DataFrame] x data
                y_data = [Series] labels
                num_epochs = [int] number of epochs
                batch_size = [int] batch size
                optimizer = [pytorch function] optimizer for training
                criterion = [pytorch function] loss function
        Output: trained model, train loss
        '''
        train_loader = torch.utils.data.DataLoader(dataset = np.array(pd.concat([x_data, y_data], axis = 1)),
                                                    batch_size = batch_size,
                                                    shuffle = True)
        # Train the model
        loss_vec = np.zeros(num_epochs)
        for epoch in trange(num_epochs):
            loss = self.train_model(train_loader,
                                    criterion = criterion,
                                    optimizer = optimizer)
            loss_vec[epoch] = loss.item()
        return self, loss_vec

