import argparse
import subprocess
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


def box_print(msg):
    """
    Small helper function to print messages to console in a centralised box.

    :param msg: Message to be placed in box
    :type msg: str
    """
    max_len = max(78, len(msg)+10)
    print('{}'.format('-'*(max_len+2)))
    print('|{}|'.format(msg.center(max_len)))
    print('{}'.format('-'*(max_len+2)))


def experiment_parser():
    """
    Build the terminal parser to make running experiments easier.
    """
    parser = argparse.ArgumentParser(description="Run Vision or RL tasks.")

    # Create vision parser
    subparser = parser.add_subparsers(help='Experiment mode (e.g. run Vision or RL Experiments.')

    # Add vision parser arugments
    vision_parser = subparser.add_parser('vision', help='Run vision experiments.')
    vision_parser.add_argument('--task', default=0, type=int)
    vision_parser.add_argument('-m', '--mode', help='Which experiment to run: \nTraining (0) \nCNN vs. BCNN (1)'
                                                    '\nMNIST Rotation (2) \nFGSM Adversary(3)', default=3,
                               type=int, choices=[0, 1, 2, 3], required=True)
    vision_parser.add_argument('-f', '--fgsmepsilon', help='Value of epsilon for crafting adversaries.',
                               type=fgsm_float, required=False, default=0.1)
    vision_parser.add_argument('-e', '--epochs', help='Number of epochs to train the CNN and BCNN for.', type=int,
                               required=False, default=20)
    vision_parser.add_argument('-s', '--seed', help='Seed for reproducibility', type=int, required=False, default=123)
    vision_parser.add_argument('--model', help='Standard CNN or Bayesian CNN, if applicable.', type=str, default='cnn',
                               choices=['cnn', 'bcnn'])

    # Create RL parser
    rl_parser = subparser.add_parser('rl', help='Run Reinforcement Learning experiments.')

    # Add RL arguments
    rl_parser.add_argument('--task', default=1, type=int)
    rl_parser.add_argument('-t', '--train', help='Should the agent be learning?', type=bool, required=False,
                           default=True)
    rl_parser.add_argument('-s', '--seed', help='Seed for reproducibility', type=int, required=False, default=123)

    # Wrap up parser args
    args = parser.parse_args()
    return args


def fgsm_float(x):
    """
    Ensure that the value of epsilon used when creating FGSM adversaries is within a set range of values.

    :param x: The value of epsilon
    :type x: float
    :return: Verified value of epsilon
    :rtype: float
    """
    x = float(x)
    if x < 0.0 or x > 2:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 2]"%(x,))
    return x


class VisionExperiments:
    """
    Object that allows for vision based experiments to be run using terminal arguments.
    """
    def __init__(self, args):
        self.args = args
        self.meta_name = 'Vision'

    def run_experiment(self):
        """
        Invoke the desired experiment.

        :return: Terminal command to run experiments.
        """
        subprocess.call('python -m src.vision.vision --mode {} --fgsmeps {} --model {}'.format(self.args.mode,
                                                                                               self.args.fgsmepsilon,
                                                                                               self.args.model),
                        shell=True)


class RLExperiments:
    """
    Object that allows for reinforcement learning based experiments to be run using terminal arguments.
    """
    def __init__(self, args):
        self.args = args
        self.meta_name = 'Reinforcement Learning'

    def run_experiment(self):
        """
        Invoke the desired experiment.

        :return: Terminal command to run experiments.
        """
        subprocess.call('python -m src.rl.rl_run --train={} --video_path=./video --logs_path=./logs'.format(self.args.train),
                        shell=True)


def gpu_setup(status=False):
    """
    Configure PyTorch to run on a GPU or CPU dependent upon hardware capabilities.

    :param status: Is a GPU present or not.
    :type status: bool
    :return: Backend status and default tensor types
    :rtype: bool str
    """
    if status:
        bend = True
        dtype = 'torch.cuda.FloatTensor' # Uncomment this to run on GPU
        box_print('CUDA Enabled')
    else:
        bend = False
        dtype = 'torch.FloatTensor'
    return bend, dtype


class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def load_data(args, reduced,  kwargs):
    """
    Load in the MNIST dataset are setup batching.

    :param args: Argparser object
    :param kwargs: GPU specific kwargs
    :return: Train and Test datasets
    """
    tr_mnist = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
    te_mnist = datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                                    transforms.Normalize((0.1307,),
                                                                                                         (0.3081,))]))
    labels = torch.from_numpy(np.arange(10))
    sampler = StratifiedSampler(class_vector=labels, batch_size=2)
    train_loader = torch.utils.data.DataLoader(tr_mnist, batch_size=args.batch_size, shuffle=True, **kwargs)
    if reduced:
        test_loader = torch.utils.data.DataLoader(te_mnist, batch_size=args.test_batch_size, sampler=sampler, **kwargs)
    else:
        box_print('Loading Reduced Dataset')
        test_loader = torch.utils.data.DataLoader(te_mnist, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def vision_parser():
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
    parser.add_argument('--small', help='Should a reduced test dataset be loaded', type=bool, default=False)
    parser.add_argument('--model', default='cnn', choices=['cnn', 'bcnn'], type=str)
    args = parser.parse_args()
    args.cuda = False # not args.no_cuda and torch.cuda.is_available()
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
