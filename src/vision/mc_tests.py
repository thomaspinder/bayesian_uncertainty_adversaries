import math
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm
from src.utils import utility_funcs as uf


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
    uf.box_print('Calculating MC-Dropout Values for Adversarial Images')
    model.train()
    passes = 100
    results = []
    for i, (data, target) in enumerate(tqdm(test_loader, desc='Batching Test Data')):
        if epsilon >= 1.0:
            orig_pred, orig_conf = mc_make_prediction(data, target, model, passes)

            # Perturb image
            data = adversary.fgsm(data, target, i)

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
        rotation_list = range(0, 180, 15)
        labels = []
        while len(labels) < 10:
            for data, target in tqdm(test_loader, desc='Batching Test Data'):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                if target.item() not in labels:
                    mc_preds = []
                    mc_unc = []
                    cnn_pred = []
                    cnn_soft = []
                    labels.append(target.item())
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

                        logits = torch.unsqueeze(F.softmax(model(data_rotate)), 0)
                        pred = np.argmax(logits.numpy())
                        prob = logits.numpy().tolist()[0][0][pred]
                        for i in range(stochastic_passes):
                            output_list.append(torch.unsqueeze(F.softmax(model(data_rotate)), 0))
                        output_mean = torch.cat(output_list, 0).mean(0)

                        # Prediction Uncertainty
                        output_variance = torch.cat(output_list, 0).var(0).mean().item()
                        confidence = output_mean.data.cpu().numpy().max()

                        # Get MC-Dropout prediction
                        predict = output_mean.data.cpu().numpy().argmax()
                        mc_preds.append(predict)
                        mc_unc.append(output_variance)
                        cnn_pred.append(pred)
                        cnn_soft.append(prob)
                        unct_list.append(output_variance)
                        print('rotation degree', str(r).ljust(3),
                              'Uncertainty: {:.4f} Predict: {} \nSoftmax: {:.2f} Pred: {}'.format(output_variance,
                                                                                                  predict, prob, pred))


                    # f, ax = plt.subplots(nrows=3, ncols=4)
                    # for i, a in enumerate(ax):
                    #     a.imshow(image_list[i][0, 0, :, :].data.cpu().numpy(), cmap='gray')
                    #     a.set_title('Rotation: {}\nMC Pred: {} with uncertainty: {}\nCNN Pred: {} with softmax: {}'.format(rotation_list[i],
                    #                                                                                                            mc_preds[i],
                    #                                                                                                            mc_unc[i],
                    #                                                                                                            cnn_pred[i],
                    #                                                                                                            cnn_soft[i]))
                    # plt.savefig('results/plots/rotations/mnist_{}_rot.png'.format(target.item()))

                    plt.figure(figsize=(14,12))
                    for i in range(len(rotation_list)):
                        ax = plt.subplot(3, len(rotation_list)/3, i + 1)
                        plt.tight_layout()
                        plt.subplots_adjust(hspace=0.5, bottom=0.1)
                        plt.text(0.5, -0.35, "MC-Prediction: {} \nUncertainty: {}\n"
                                            "CNN Prediction: {} "
                                         "\nSoftmax Probability: {}".format(mc_preds[i],
                                                                            np.round(mc_unc[i], 3),
                                                                            cnn_pred[i],
                                                                            np.round(cnn_soft[i]), 5),
                                 size=12, ha="center", transform=ax.transAxes)
                        plt.axis('off')
                        plt.gca().set_title('Rotation: ' + str(rotation_list[i]) + u'\xb0')
                        plt.imshow(image_list[i][0, 0, :, :].data.cpu().numpy(), cmap='gray')
                    plt.savefig('results/plots/rotations/mnist_{}_rot.png'.format(target.item()))
                    print()
                else:
                    pass