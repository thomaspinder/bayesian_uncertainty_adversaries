import argparse
import subprocess
import torch


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
        subprocess.call('python -m src.rl.main_rl --train={} --video_path=./video --logs_path=./logs'.format(self.args.train),
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