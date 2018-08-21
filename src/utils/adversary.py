import torch
from torch import nn as nn

from src.utils import utility_funcs as uf


class Adversary:
    def __init__(self, model, epsilon, limits = (-1, 1)):
        self.net = model
        self.eps = epsilon
        self.lim = limits
        self.cost = nn.CrossEntropyLoss()
        self.counter = 0
        uf.box_print('Creating Adversaries with Epsilon = {}'.format(self.eps))

    def fgsm(self, x, y):
        # Initalise adversary
        adv = x.clone()
        adv = torch.tensor(adv.data, requires_grad=True)

        # Make initial prediction
        pred = self.net(adv)

        # Calculate loss value
        loss = self.cost(pred, y)

        # Reset gradients
        self.net.zero_grad()
        if adv.grad is not None:
            adv.grad.data.fill_(0)


        # Get gradient
        loss.backward()

        # Get sign
        adv.grad.sign_()

        # Calculate perturbation
        eta = self.eps*adv.grad

        # Perturb image
        eta = self.eps*torch.sign(x.data)
        adv = adv - eta
        adv = torch.clamp(adv, self.lim[0], self.lim[1])

        # New prediction
        original_logit = self.net(x)
        adv_pred_logit = self.net(adv)
        _, original = original_logit.max(-1)
        _, adv_pred = adv_pred_logit.max(-1)

        if adv_pred != original:
            # print('{}\nOriginal: {}\nAdversary:{}'.format('-'*80, original.item(), adv_pred.item()))
            self.counter += 1
        return adv