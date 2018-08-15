import os
import argparse
import numpy as np
from visdom import Visdom

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs_path', dest='logs_path', help='path of the checkpoint folder',
                        default='./logs', type=str)
    args = parser.parse_args()
    
    return args

args = parse_args()

def main():
    logs_path = args.logs_path
    # Split the file into episode and reward 1d arrays
    episode, reward = zip(*np.load(os.path.join(logs_path, 'reward.npy')))
    step, loss = zip(*np.load(os.path.join(logs_path, 'loss.npy')))

    avg_reward = np.cumsum(reward) / np.arange(1, len(reward) + 1)

    test = np.column_stack([reward, avg_reward, loss])

    viz = Visdom(env='main')
    viz.line(X=np.array(step).reshape(-1, 1).repeat(3, 1),
             Y=np.column_stack([reward, avg_reward, loss]),
             opts=dict(
                 legend=['Reward', 'Average Reward', 'Loss']
             )
             )


if __name__ == '__main__':
    main()
