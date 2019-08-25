from ED_support_module import *

class NeuralNet(nn.Module):
    def __init__(self, device, input_size=61, hidden_size=500, num_classes=2, drop_prob=0):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.ac1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.ac2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.ac3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size)
        self.ac4 = nn.LeakyReLU()
        self.classification = nn.Linear(hidden_size, num_classes)
        self.dp_layer1 = nn.Dropout(drop_prob)
        self.dp_layer2 = nn.Dropout(drop_prob)
        self.dp_layer3 = nn.Dropout(drop_prob)
        self.dp_layer4 = nn.Dropout(drop_prob)
        self.device = device
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    def forward(self, x):
        h = self.fc1(x)
        h = self.ac1(h)
        h = self.dp_layer1(h)
        h = self.fc2(h)
        h = self.ac2(h)
        h = self.dp_layer2(h)
        h = self.fc3(h)
        h = self.ac3(h)
        h = self.dp_layer3(h)
        h = self.fc4(h)
        h = self.ac4(h)
        h = self.dp_layer4(h)
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
        loss_sum = 0
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
            # Add loss
            loss_sum += loss.item()
        return loss_sum
    def eval_model(self, test_loader, transformation=None, criterion=None,
                    if_y_data=False):
        '''
        Evaluate the neural network model. Only makes sense if
        test_loader contains all test data. Note that this is
        different from the built-in method self.eval, which
        sets the model to train mode.
        
        Model will be set to evaluation mode internally.

        Input :
                train_loader = [DataLoader] training set. Must contain
                                the response in the last column.
                transformation = [Function] function for evaluatin
                                 transforming the output. If not given,
                                 raw output is return.
                criterion = [Function] if given, the accumulative loss
                            is computed.
        Output:
                outputs = output from the model (after transformation).
                loss = accumulative loss.
        '''
        self.eval()
        loss_sum = 0
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(self.device)
                # Retrieve design matrix and labels
                if if_y_data:
                    labels = x[:, -1].long()
                    x = x[:, :(-1)].float()
                else:
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
                                            np.array(outputs.cpu()),
                                            axis = 0)
                # Compute loss
                if criterion is not None and not if_y_data:
                    loss = criterion(outputs, labels)
                    loss_sum += loss.item()
        if criterion is not None:
            return outputs_vec, loss_sum
        else:
            return outputs_vec
    def predict_proba_single(self, x_data, batch_size=None, transformation=None):
        '''
        Transform x_data into dataloader and return predicted scores
        for being of class 1.
        Input :
                x_data = [DataFrame] train set.
                y_data = [DataFrame] labels. If not given, criterion must be
                        None.
                batch_size = [int] batch size used to construct data loader.
                transformation = [Function] function for evaluatin
                                 transforming the output. If not given,
                                 raw output is returned.
                criterion = [Function] if given, the accumulative loss
                            is computed.
        Output:
                pred_prob = [array] predicted probability for being
                            of class 1.
        '''
        data = x_data
        if batch_size is None:
            batch_size = x_data.shape[0]
        test_loader = torch.utils.data.DataLoader(dataset = np.array(data),
                                                batch_size = batch_size,
                                                shuffle = False)
        if transformation is None:
            transformation = nn.Sigmoid().to(self.device)
        return self.eval_model(test_loader, transformation, criterion = None, if_y_data = False)[:, 1]
    def validate(self, x_data, y_data=None, batch_size=None,
                transformation=None, criterion=None):
        '''
        Transform x_data into dataloader and return predicted scores
        for being of class 1.
        Input :
                x_data = [DataFrame] train set.
                y_data = [DataFrame] labels. If not given, criterion must be
                        None.
                batch_size = [int] batch size used to construct data loader.
                transformation = [Function] function for evaluatin
                                 transforming the output. If not given,
                                 raw output is returned.
                criterion = [Function] if given, the accumulative loss
                            is computed.
        Output:
                pred_prob = [array] predicted probability for being
                            of class 1.
        '''
        if y_data is None and criterion is not None:
            raise ValueError("Criterion must be None if y_data is None")
        if y_data is not None:
            data = pd.concat([x_data, y_data], axis = 1)
            if_y_data = True
        else:
            data = x_data
            if_y_data = False
        if batch_size is None:
            batch_size = x_data.shape[0]
        test_loader = torch.utils.data.DataLoader(dataset = np.array(data),
                                                batch_size = batch_size,
                                                shuffle = False)
        if criterion is not None:
            pred_prob, loss = self.eval_model(test_loader, transformation, criterion, if_y_data)
            return pred_prob[:, 1], loss
        else:
            return self.eval_model(test_loader, transformation, criterion, if_y_data)[:, 1]
    def fit(self, x_data, y_data, num_epochs, batch_size, optimizer, criterion):
        '''
        Fit the model on x_data and y_data.
        
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
            loss_sum = self.train_model(train_loader,
                                        criterion = criterion,
                                        optimizer = optimizer)
            loss_vec[epoch] = loss_sum
        return self, loss_vec
    def save_model(self, save_path):
        '''
        Save the model to save_path.
        Input :
                save_path = [str] path to save the model parameters.
        Output:
                None
        '''
        # Save model
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save, save_path)


