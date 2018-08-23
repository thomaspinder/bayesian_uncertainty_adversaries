import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import argparse

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.manual_seed(123)


class BayesianNet(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(BayesianNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_sizes[2], 1)

    def forward(self, x):
        out = self.relu(F.dropout(self.fc1(x), p=0.5, training=True))
        out = self.relu(F.dropout(self.fc2(out), p=0.5, training=True))
        out = self.relu(F.dropout(self.fc3(out), p=0.5, training=True))
        out = self.fc4(out)
        return out


class StandardNet(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(StandardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_sizes[2], 1)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out


class Dataset:
    def __init__(self):
        self.data = None
        self.data_list= None

    def load_avocado(self, split, directory='data/avocado.csv'):
        avocados = pd.read_csv(directory)

        # Generate Delta time feature
        avocados['Date'] = pd.to_datetime(avocados['Date'])
        avocados['delta'] = (avocados['Date'] - min(avocados['Date'])).dt.days
        avocados['delta'] = (avocados['delta']-avocados['delta'].mean())/avocados['delta'].std()
        avocados = avocados.dropna()
        self.data = avocados

        # Split data
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(avocados['delta'], avocados['AveragePrice'],
                                                                      random_state=123, test_size=0.2)


def build_parser():
    args = argparse.ArgumentParser()
    args.add_argument('-m', '--model', type=str, default='bnn', choices=['bnn', 'nn'], help='Bayesian Net or a standard Net?')
    return args.parse_args()

if __name__=='__main__':
    Avocado = Dataset()
    Avocado.load_avocado(split=True)
    input_dim = Avocado.data.shape[1]
    parser = build_parser()

    # Specify Hyperparameters
    l_rate = 0.01
    epochs = 60
    batch_size = 512
    if parser.model=='bnn':
        model = BayesianNet(1, hidden_sizes=[32, 16, 32])
    elif parser.model=='nn':
        model = StandardNet(1, hidden_sizes=[32, 16, 32])
    model.cuda().float()
    criterion = nn.MSELoss()  # Mean Squared Loss
    optimiser = torch.optim.SGD(model.parameters(), lr=l_rate)

    train = torch.utils.data.TensorDataset(torch.Tensor(Avocado.X_tr.values).cuda().float(),
                                           torch.Tensor(Avocado.y_tr.values).cuda().float())
    test = torch.utils.data.TensorDataset(torch.Tensor(Avocado.X_te.values).cuda().float(),
                                           torch.Tensor(Avocado.y_te.values).cuda().float())
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, shuffle=False)

    # Train
    for i in range(epochs):
        losses = []
        epoch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = torch.unsqueeze(data, dim=0).permute(1,0)
            target = torch.unsqueeze(target, dim=0).permute(1,0)
            outputs = model(data)

            loss = criterion(outputs, target)
            epoch_loss.append(loss.item())
            optimiser.zero_grad()
            loss.backward()  # back props
            optimiser.step()  # update the parameters
        losses.append(sum(epoch_loss)/len(epoch_loss))
        print('epoch {}, loss {}'.format(i, loss.data[0]))

    # Test
    predictions = []
    passes = 100
    if parser.model=='bnn':
        print('Making Stochastic Predictions')
        for bacth_idx, (data, target) in enumerate(test_loader):
            data = torch.unsqueeze(data, dim=0).permute(1, 0)
            stochastic_predictions = []
            for _ in range(passes):
                predicted = model(data).item()
                stochastic_predictions.append(predicted)
            stochastic_predictions.insert(0, target.item())
            stochastic_predictions.insert(0, data.item())
            predictions.append(stochastic_predictions)
        columns = ['prediction_{}'.format(i+1) for i in range(passes)]
        columns.insert(0, 'truth')
        columns.insert(0, 'obs')
        predictions_df = pd.DataFrame(predictions, columns=columns)

    elif parser.model=='nn':
        print('Standard Prediction')
        for bacth_idx, (data, target) in enumerate(test_loader):
            data = torch.unsqueeze(data, dim=0).permute(1,0)
            predicted = model(data).item()
            predictions.append([data.item(), predicted, target.item()])
        predictions_df = pd.DataFrame(predictions, columns=['obs', 'prediction', 'truth'])
    predictions_df.to_csv('results/motivation_{}_avocados.csv'.format(parser.model), index=False)
