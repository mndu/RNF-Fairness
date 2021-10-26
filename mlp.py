import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


# The whole model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        representation = self.fc1_drop(out)
        out = self.fc2(representation)
        out = self.relu(out)
        out = self.fc2_drop(out)
        logit = self.fc3(out)
        return logit, representation


# The classification head
class FC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FC, self).__init__()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc2(x)
        out = self.relu(out)
        out = self.fc2_drop(out)
        logit = self.fc3(out)
        return logit