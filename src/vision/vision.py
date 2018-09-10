# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from src.utils import utility_funcs as uf
from src.utils.adversary import Adversary
from src.utils.utility_funcs import load_data, vision_parser, action_args
from src.vision.mc_tests import fgsm_test_mc, mcdropout_test, uncertainty_test
from src.vision.model import LeNet_standard, LeNet_dropout
from src.vision.tests import fgsm_test, make_prediction

torch.manual_seed(123)


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
    lr = args.lr
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


if __name__ == '__main__':
    args = vision_parser()
    kwargs = action_args(args)
    # Setup GPU if necessary
    torch.backends.cudnn.benchmark, dtype = uf.gpu_setup(args.cuda)
    torch.set_default_tensor_type(dtype)

    train_loader, test_loader = load_data(args, args.small, kwargs)
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
        end = time.time() - start
        uf.box_print('Training Time for Standard Model: {}'.format(end))
        test(model_standard, args, test_loader)

        uf.box_print('Train Lenet with dropout at all layer')
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            train(model_dropout, optimizer_dropout, epoch, args, train_loader)
        end = time.time() - start
        uf.box_print('BCNN Training Time: {}'.format(end))
        mcdropout_test(model_dropout, args, test_loader)

        uf.box_print('Save checkpoint/' + 'LeNet_stadard' + str(epoch) + '.pth.tar')
        state = {'state_dict': model_standard.state_dict()}
        filename = 'src/vision/checkpoint/' + 'LeNet_stadard' + str(epoch) + '.pth.tar'
        torch.save(state, filename)

        uf.box_print('Save checkpoint/' + 'LeNet_dropout' + str(epoch) + '.pth.tar')
        state = {'state_dict': model_dropout.state_dict()}
        filename = 'src/vision/checkpoint/' + 'LeNet_dropout' + str(epoch) + '.pth.tar'
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
