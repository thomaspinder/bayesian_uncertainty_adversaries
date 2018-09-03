import torch
from torch import nn as nn
import matplotlib.pyplot as plt
from src.utils import utility_funcs as uf


class Adversary:
    """
    A PyTorch implementation of adversarial attacks.
    """
    def __init__(self, model, epsilon, limits = (-1, 1)):
        """
        Initial parameters for the adversary.

        :param model: The neural network being attacks
        :type model: PyTorch Model
        :param epsilon: The magnitude for which the image should be perturbed
        :type epsilon: float
        :param limits: The l-infinity bound for perturbations
        :type limits: 2-tuple
        """
        self.net = model
        self.eps = epsilon
        self.lim = limits
        self.cost = nn.CrossEntropyLoss()
        self.counter = 0
        uf.box_print('Creating Adversaries with Epsilon = {}'.format(self.eps))

    def fgsm(self, x, y, i=-1):
        """
        An implementation of the Fast Gradient Sign Method, used to carry out attacks at a pixel level on the image, based upon the gradient of the image's cost function.

        :param x: The original image to perturb
        :type x: Tensor
        :param y: True label of the original image
        :type y: int
        :param i: Indexer, only used for plotting and not necessary. Should plotting not be required, just set i<0
        :type i: int
        :return: Perturbed version of the original image
        :rtype: Tensor
        """
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
        adv_sign = torch.sign(adv.grad.data)

        # Calculate perturbation
        # eta = self.eps*adv.grad
        eta = self.eps*adv_sign

        # Perturb image
        adv = x.data + eta

        # Plot
        if i == 5:
            f, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].imshow(x.numpy().squeeze(), cmap='gray')
            ax[0].set_title('Original Image')
            ax[1].imshow(eta.numpy().squeeze(), cmap='gray')
            ax[1].set_title('Epsilon = {}'.format(self.eps))
            ax[2].imshow(adv.numpy().squeeze(), cmap='gray')
            ax[2].set_title('Perturbed Image')
            plt.savefig('results/plots/mnist_advs/MNIST_noise_{}.png'.format(self.eps))
            print(eta.numpy().squeeze())


        # New prediction
        original_logit = self.net(x)
        adv_pred_logit = self.net(adv)
        _, original = original_logit.max(-1)
        _, adv_pred = adv_pred_logit.max(-1)

        if adv_pred != original:
            # print('{}\nOriginal: {}\nAdversary:{}'.format('-'*80, original.item(), adv_pred.item()))
            self.counter += 1
        return adv