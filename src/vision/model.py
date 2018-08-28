import torch.nn as nn
import torch.nn.functional as F


class LeNet_standard(nn.Module):
    """
    Standard LeNet architecture model.
    """
    def __init__(self):
        super(LeNet_standard, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Make a forward pass through the network.

        :param x: Observation or batch for which the network should be passed over.
        :return: Prediction as a softmax distribution over a single image or an image in the batch.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class LeNet_dropout(nn.Module):
    """
    Customised LeNet architecture with dropout incorporated into each layer of the network. This dropout is kept on at test time, thus ensuring the convergence to a GP.
    """
    def __init__(self):
        super(LeNet_dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Make a forward pass through the network.

        :param x: Observation or batch for which the network should be passed over.
        :return: Prediction as a softmax distribution over a single image or an image in the batch.
        """
        x = F.relu(F.max_pool2d(F.dropout(self.conv1(x), training=True), 2))
        x = F.relu(F.max_pool2d(F.dropout(self.conv2(x), training=True), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = self.fc2(x)
        return x
