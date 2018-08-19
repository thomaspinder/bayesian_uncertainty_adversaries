# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from src.utils import utility_funcs as uf
from src.vision.model import LeNet_standard, LeNet_dropout

torch.manual_seed(123)


def build_parser():
    """
    Setup parser.
    """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--mode', type=int, default=0, metavar='N',
                        help='train mode (0) test mode (1)'
                        'uncertainty test mode (2) (default: 0)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval of logging training status')
    parser.add_argument('-f', '--fgsmeps', default=0.1, type=float)
    parser.add_argument('--model', default='cnn', choices=['cnn', 'bcnn'], type=str)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def action_args(args):
    """
    Make GPU specific changes based upon the system's setup and the user's arguments.
    :param args: Argparser containing desired arguments.
    :return: Set of kwargs.
    """
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    return kwargs


def load_data(args, kwargs):
    """
    Load in the MNIST dataset are setup batching.

    :param args: Argparser object
    :param kwargs: GPU specific kwargs
    :return: Train and Test datasets
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def train(model, opt, epoch, args, train_loader):
    """
    Train a model.

    :param model: The model to be trained
    :type model: Torch Model
    :param opt: The optimiser to be used during training
    :param epoch: Number of epochs to be used in training. Note, there is no early stopping in place.
    :type epoch: int
    :param args: Argparser containing several user defined arguments.
    :param train_loader: Training data
    :return: Trained model
    """
    model.train()
    lr = args.lr * (0.1 ** (epoch // 10))
    opt.param_groups[0]['lr'] = lr
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Batching Training Data')):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        opt.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, 0), target)
        loss.backward()
        opt.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] lr: {}\tLoss: {:.6f}'
                  .format(epoch, batch_idx * len(data),
                          len(train_loader.dataset),
                          100. * batch_idx / len(train_loader),
                          lr, loss.data[0]))


def test(model, args, test_loader):
    """
    Test a CNN performance

    :param model: A trained BCNN
    :type model: Torch Model
    :param args: Arguments object
    :param test_loader: Testing dataset
    """
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        # Data and target are a single pair of images and labels.
        for data, target in tqdm(test_loader, desc='Batching Test Data'):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            pred, tloss = make_prediction(data, target)
            test_loss += tloss
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        uf.box_print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def mcdropout_test(model, args, test_loader, stochastic_passes=100):
    """
    Carry out basic tests on the BCNN.

    :param model: A trained BCNN
    :type model: Torch Model
    :param args: Arguments object
    :param test_loader: Testing dataset
    :param stochastic_passes: Number of stochastic passes to maker per image
    :type stochastic_passes: int
    """
    with torch.no_grad():
        model.train()
        test_loss = 0
        correct = 0
        for data, target in tqdm(test_loader, desc='Bacthing Test Data'):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output_list = []
            for i in range(stochastic_passes):
                output_list.append(torch.unsqueeze(model(data), 0))
            output_mean = torch.cat(output_list, 0).mean(0)
            test_loss += F.nll_loss(F.log_softmax(output_mean, 0), target, reduction="sum").item()  # sum up batch loss
            pred = output_mean.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        uf.box_print('MC Dropout Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def uncertainty_test(model, args, test_loader, stochastic_passes=100):
    """
    Measure the uncertainty values calculated by the BCNN as the images of the MNIST dataset are rotated through 180 degrees.

    :param model: A trained BCNN
    :type model: Torch Model
    :param args: Arguments object
    :param test_loader: Testing dataset
    :param stochastic_passes: Number of stochastic passes to maker per image
    :type stochastic_passes: int
    """
    with torch.no_grad():
        model.train()
        rotation_list = range(0, 180, 10)
        for data, target in tqdm(test_loader, desc='Batching Test Data'):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output_list = []
            image_list = []
            unct_list = []
            for r in rotation_list:
                rotation_matrix = Variable(
                    torch.Tensor([[[math.cos(r / 360.0 * 2 * math.pi), -math.sin(r / 360.0 * 2 * math.pi), 0],
                                   [math.sin(r / 360.0 * 2 * math.pi), math.cos(r / 360.0 * 2 * math.pi), 0]]]))
                grid = F.affine_grid(rotation_matrix, data.size())
                data_rotate = F.grid_sample(data, grid)
                image_list.append(data_rotate)

                for i in range(stochastic_passes):
                    output_list.append(torch.unsqueeze(F.softmax(model(data_rotate)), 0))
                output_mean = torch.cat(output_list, 0).mean(0)
                output_variance = torch.cat(output_list, 0).var(0).mean().item()
                confidence = output_mean.data.cpu().numpy().max()
                predict = output_mean.data.cpu().numpy().argmax()
                unct_list.append(output_variance)
                print('rotation degree', str(r).ljust(3),
                      'Uncertainty : {:.4f} Predict : {} Softmax : {:.2f}'.format(output_variance, predict, confidence))

            plt.figure()
            for i in range(len(rotation_list)):
                ax = plt.subplot(3, len(rotation_list)/3, i + 1)
                plt.text(0.5, -0.5, "{}\n{}".format(np.round(unct_list[i], 3), str(rotation_list[i])  + u'\xb0'),
                         size=12, ha="center", transform=ax.transAxes)
                plt.axis('off')
                # plt.gca().set_title(str(rotation_list[i]) + u'\xb0')
                plt.imshow(image_list[i][0, 0, :, :].data.cpu().numpy())
            plt.show()
            print()


def fgsm_test(model, adversary, args, test_loader, data_name='MNIST', model_name ='cnn'):
    """
    Evaluate a standard neural network's performance when the images being evaluated have adversarial attacks inflicted upon them.

    :param model: A trained CNN
    :type model: Torch Model
    :param adversary: An adversarial object for which attacks can be crafted
    :param args: Arguments object
    :param test_loader: Testing dataset
    :param data_name: The name of dataset being experimented upon. Not critical, only used for file naming
    :type data_name: str
    """
    model.eval()
    test_loss = 0
    correct = 0
    # Data and target are a single pair of images and labels.
    results = []
    for data, target in tqdm(test_loader, desc='Batching Test Data'):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # Original Prediction
        original_pred, tloss = make_prediction(data, target, model)
        test_loss += tloss

        # Make Adversary Prediction
        adv_data = adversary.fgsm(data, target)
        adv_pred, _ = make_prediction(adv_data, target, model)
        correct += adv_pred.eq(target.data.view_as(adv_pred)).cpu().sum()
        results.append([original_pred.item(), adv_pred.item(), target.item()])
    results_df = pd.DataFrame(results, columns=['Original Prediction', 'Adversary Prediction', 'Truth'])
    results_df['epsilon'] = adversary.eps
    results_df.to_csv('results/experiment3/{}/{}_fgsm_{}_{}.csv'.format(model_name, model_name, adversary.eps,
                                                                        data_name), index=False)


def make_prediction(data, target, model):
    data, target = Variable(data), Variable(target)
    output = model(data)
    loss_val = F.nll_loss(F.log_softmax(output, 0), target, reduction="sum").item()  # sum up batch loss
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    return pred, loss_val


def mc_make_prediction(data, target, model, passes):
    output_list = []
    for i in range(passes):
        output_list.append(torch.unsqueeze(F.softmax(model(data)), 0))
    output_mean = torch.cat(output_list, 0).mean(0)
    output_var = torch.cat(output_list, 0).var(0).mean().item()
    confidence = output_mean.data.cpu().numpy().max()
    predict = output_mean.data.cpu().numpy().argmax()
    return predict, output_var

def fgsm_test_mc(model, adversary, args, test_loader, epsilon=1.0, model_name='bcnn', data_name = 'MNIST'):
    """
    Test a BCNN against adversaries. Through the epsilon parameter here (not to be confused with the epsilon parameter used in FGSM), the proportion of images perturbed can be tested so as to compare uncertainty values calculated from original and perturbed images.

    :param model: A BCNN
    :type model: Torch Model
    :param adversary: Adversary object capable of perturbing images
    :type adversary: Adversary Object
    :param args: User defined arguments to control testing parameters
    :type args: Argparser object
    :param test_loader: Set of test data to be experimented upon
    :type test_loader: Torch DataLoader
    :param epsilon: Value between 0 and 1 to control the proportion of images perturbed. Epsilon = 1 implies that every image is perturbed.
    :type epsilon: float
    """
    uf.box_print('Calcualting MC-Dropout Values for Adversarial Images')
    model.train()
    passes = 100
    results = []
    for data, target in tqdm(test_loader, desc='Batching Test Data'):
        if epsilon >= 1.0:
            orig_pred, orig_conf = mc_make_prediction(data, target, model, passes)

            # Perturb image
            data = adversary.fgsm(data, target)

            # Make prediction on perturbed image
            adv_pred, adv_conf = mc_make_prediction(data, target, model, passes)
            results.append([int(orig_pred), orig_conf, int(adv_pred), adv_conf, target.item()])
            results_df = pd.DataFrame(results, columns=['Original Prediction', 'Original Confidence', 'Adversary Prediction','Adversary Confidence', 'Truth'])
            results_df.epsilon = adversary.eps
            results_df.to_csv('results/experiment3/{}/{}_fgsm_{}_{}.csv'.format(model_name, model_name, adversary.eps, data_name))
        else:
            adv = 'No Adversary'
            rand = np.random.rand()
            if rand < epsilon:
                data = adversary.fgsm(data, target)
                adv = 'Adversary'
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output_list = []
            for i in range(passes):
                output_list.append(torch.unsqueeze(F.softmax(model(data)), 0))
            output_mean = torch.cat(output_list, 0).mean(0)
            output_var = torch.cat(output_list, 0).var(0).mean().item()
            confidence = output_mean.data.cpu().numpy().max()
            predict = output_mean.data.cpu().numpy().argmax()
            results.append([predict.item(), confidence.item(), target.item(), adv])
            results_df = pd.DataFrame(results, columns=['prediction', 'confidence', 'truth', 'adv_status'])
            results_df.to_csv('results/fgsm_{}_bnn.csv'.format(epsilon), index=False)


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

def main():
    args = build_parser()
    kwargs = action_args(args)
    # Setup GPU if necessary
    torch.backends.cudnn.benchmark, dtype = uf.gpu_setup(args.cuda)
    torch.set_default_tensor_type(dtype)

    train_loader, test_loader = load_data(args, kwargs)
    model_standard = LeNet_standard()
    model_dropout = LeNet_dropout()
    if args.cuda:
        model_standard.cuda()
        model_dropout.cuda()

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    # Train both models
    if args.mode == 0:
        optimizer_standard = optim.SGD(model_standard.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer_dropout = optim.SGD(model_dropout.parameters(), lr=args.lr, momentum=args.momentum)

        uf.box_print('Train standard Lenet')
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            train(model_standard, optimizer_standard, epoch, args, train_loader)
        end = time.time()-start
        uf.box_print('Training Time for Standard Model: {}'.format(end))
        test(model_standard, args, test_loader)

        uf.box_print('Train Lenet with dropout at all layer')
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            train(model_dropout, optimizer_dropout, epoch, args, train_loader)
        end = time.time()-start
        uf.box_print('BCNN Training Time: {}'.format(end))
        mcdropout_test(model_dropout, args, test_loader)

        uf.box_print('Save checkpoint/'+'LeNet_stadard'+str(epoch)+'.pth.tar')
        state = {'state_dict': model_standard.state_dict()}
        filename = 'src/vision/checkpoint/'+'LeNet_stadard'+str(epoch)+'.pth.tar'
        torch.save(state, filename)

        uf.box_print('Save checkpoint/'+'LeNet_dropout'+str(epoch)+'.pth.tar')
        state = {'state_dict': model_dropout.state_dict()}
        filename = 'src/vision/checkpoint/'+'LeNet_dropout'+str(epoch)+'.pth.tar'
        torch.save(state, filename)

    # Test models on clean MNIST dataset
    elif args.mode == 1:
        ckpt_standard = torch.load('src/vision/checkpoint/LeNet_stadard5.pth.tar')
        model_standard.load_state_dict(ckpt_standard['state_dict'])
        test(model_standard, args, test_loader)

        ckpt_dropout = torch.load('src/vision/checkpoint/LeNet_dropout5.pth.tar')
        model_dropout.load_state_dict(ckpt_dropout['state_dict'])
        mcdropout_test(model_dropout, args, test_loader)

    # Test uncertainty on MNIST images rotated through 180 degrees
    elif args.mode == 2:
        ckpt_dropout = torch.load('src/vision/checkpoint/LeNet_dropout5.pth.tar')
        model_dropout.load_state_dict(ckpt_dropout['state_dict'])
        uncertainty_test(model_dropout, args, test_loader)

    # Test models on adversarial images
    elif args.mode == 3:
        if args.model == 'cnn':
            uf.box_print('Testing Standard CNN')
            ckpt_standard = torch.load('src/vision/checkpoint/LeNet_stadard20.pth.tar')
            model_standard.load_state_dict(ckpt_standard['state_dict'])
            adv = Adversary(model_standard, args.fgsmeps)
            fgsm_test(model_standard, adv, args, test_loader, model_name=args.model)
            print('Total Fooled: {}'.format(adv.counter))
        elif args.model == 'bcnn':
            uf.box_print('Testing Bayesian CNN')
            ckpt_dropout = torch.load('src/vision/checkpoint/LeNet_dropout20.pth.tar')
            model_dropout.load_state_dict(ckpt_dropout['state_dict'])
            adv = Adversary(model_dropout, args.fgsmeps)
            fgsm_test_mc(model_dropout, adv, args, test_loader, epsilon=1.0, model_name=args.model, data_name='MNIST')
            print('Total Fooled: {}'.format(adv.counter))

    elif args.mode == 4:
        ckpt_dropout = torch.load('src/vision/checkpoint/LeNet_dropout5.pth.tar')
        model_dropout.load_state_dict(ckpt_dropout['state_dict'])
        adv = Adversary(model_dropout, args.fgsmeps)
        fgsm_test_mc(model_dropout, adv, args, test_loader, epsilon=0.5)

    else:
        print('--mode argument is invalid \ntrain mode (0) or test mode (1) uncertainty test mode (2)')


if __name__ == '__main__':
    main()
