from ED_support_module import *
from ED_support_module.NeuralNet import NeuralNet

class StackedModel(NeuralNet):
    def __init__(self, device, input_size=2, hidden_size=100, num_classes=2, drop_prob=0):
        super(StackedModel, self).__init__(device, input_size, hidden_size, num_classes, drop_prob)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.classification = nn.Linear(self.hidden_size, num_classes)
        self.ac1 = nn.LeakyReLU()
        self.dp_layer = nn.Dropout(drop_prob)
        self.device = device
    def forward(self, x):
        h = self.fc1(x)
        h = self.ac1(h)
        h = self.dp_layer1(h)
        return self.classification(h)