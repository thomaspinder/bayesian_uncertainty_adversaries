import pandas as pd
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm


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
    """
    Make a single prediction for a pair of observations and labels.

    :param data: The input features
    :param target: The output labels
    :param model: Model to test
    :return: A prediction and the corresponding loss incurred from the respective prediction
    """
    data, target = Variable(data), Variable(target)
    output = model(data)
    loss_val = F.nll_loss(F.log_softmax(output, 0), target, reduction="sum").item()  # sum up batch loss
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    return pred, loss_val