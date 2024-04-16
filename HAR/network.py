import torch
import torch.nn.functional as F

class LinearNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)  # hidden layer
        self.out = torch.nn.Linear(n_hidden1, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = self.out(x)
        return x


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


class MLP(torch.nn.Module):
    def __init__(self,):
        super(MLP, self).__init__()
        self.hidden1 = torch.nn.Linear(561, 100)
        self.act1 = torch.nn.ReLU()
        self.classification_layer = torch.nn.Linear(100, 6)
        self.act3 = torch.nn.LogSoftmax(dim=1)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.classification_layer(X)
        X = self.act3(X)
        return X