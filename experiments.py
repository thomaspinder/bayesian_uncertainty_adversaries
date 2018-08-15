import argparse
import subprocess


def build_parser():
    # Initialise parser
    parser = argparse.ArgumentParser(description="Run Vision or RL tasks.")

    # Create vision parser
    subparser = parser.add_subparsers(help='Experiment mode (e.g. run Vision or RL Experiments.')

    # Add vision parser arugments
    vision_parser = subparser.add_parser('vision', help='Run vision experiments.')
    vision_parser.add_argument('--task', default=0, type=int)
    vision_parser.add_argument('-m', '--mode', help='Which experiment to run: \nTraining (0) \nCNN vs. BCNN (1)'
                                                    '\nMNIST Rotation (4) \nFGSM Adversary(3)', default=0,
                               type=int, choices=[0, 1, 2, 3], required=True)
    vision_parser.add_argument('-f', '--fgsm-epsilon', help='Value of epsilon for crafting adversaries.', type=float,
                               required=False, default=0.1)
    vision_parser.add_argument('-e', '--epochs', help='Number of epochs to train the CNN and BCNN for.', type=int,
                               required=False, default=20)
    vision_parser.add_argument('-s', '--seed', help='Seed for reproducibility', type=int, required=False, default=123)

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

class VisionExperiments:
    def __init__(self, args):
        self.args = args
        self.meta_name = 'Vision'

    def run_experiment(self):
        subprocess.call('python -m vision.cnn_un --mode {}'.format(self.args.mode), shell=True)


class RLExperiments:
    def __init__(self, args):
        self.args = args
        self.meta_name = 'Reinforcement Learning'

    def run_experiment(self):
        subprocess.call('python -m rl.main --train={} --video_path=./video --logs_path=./logs'.format(self.args.train),
                        shell=True)


if __name__=='__main__':
    args = build_parser()
    if args.task == 0:
        exps = VisionExperiments(args)
    elif args.task == 1:
        exps = RLExperiments(args)
    else:
        raise ValueError('Incorrect Task Type Selected. Please specify either "vision" or "rl".')
    console_output = 'Conducting {} Experiments'.format(exps.meta_name)
    print('-'*80)
    print('|{}|'.format(console_output.center(78)))
    print('-' * 80)
    exps.run_experiment()

