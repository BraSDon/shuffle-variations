import torch.nn as nn


class DummyModel(nn.Module):
    """
    A dummy model for testing purposes.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18, 300)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(300, 300)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(300, 300)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(300, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.fc3(x)
        x = self.tanh3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x


class DeeperANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18, 500)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(500, 500)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(500, 500)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(500, 500)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(500, 500)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(500, 500)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(500, 500)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(500, 500)
        self.relu8 = nn.ReLU()
        self.fc9 = nn.Linear(500, 500)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(500, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc8(x)
        x = self.relu8(x)
        x = self.fc9(x)
        x = self.relu9(x)
        x = self.fc10(x)
        x = self.softmax(x)
        return x
