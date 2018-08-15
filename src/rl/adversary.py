import torch.nn as nn
import torch

class Adversary:
    def __init__(self, model, epsilon, limits = (-1, 1)):
        self.net = model
        self.net.eval()
        self.eps = epsilon
        self.lim = limits
        self.cost = nn.CrossEntropyLoss()

    def fgsm(self, x, y):
        # Make prediction
        pred = self.net(x)

        # Calculate loss value
        loss = self.cost(pred, y)

        # Get gradient
        loss.backward()

        # Perturb image
        x_adv = self.eps*torch.sign(x.data)
        return x_adv