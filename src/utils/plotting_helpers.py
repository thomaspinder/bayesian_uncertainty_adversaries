import glob
import pandas as pd
import argparse
import re


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fgsm', help='FGSM Results from vision experiment 3', type=bool, default=False)
    parser.add_argument('-d', '--directory', help='Directory containing results', type=str, default=False)
    parser.add_argument('-o', '--outfile', help='Save path', type=str, default=False)
    parser.add_argument('--forgot', type=bool, help='I forgot to add epsilon to the results once so this takes epsilon from the file name', default=False)
    parser.add_argument('-t', '--modeltype', type=str, choices=['cnn', 'bcnn'], default='cnn', help='cnn or bcnn')
    arguments = parser.parse_args()
    return arguments


def merge_results(directory, outname, forget=False, model_type='cnn'):

    files = glob.glob('{}/*.csv'.format(directory))

    # Load in result set
    results = pd.read_csv(files[0])
    if forget:
        epsilon = get_epsilon(files[0])
        results['epsilon'] = epsilon
    for file in files[1:]:
        intermediate = pd.read_csv(file)
        if forget:
            if 'bcnn_fgsm_0.01_MNIST.csv' in file:
                epsilon = 0.01
            else:
                epsilon = get_epsilon(file)
            intermediate['epsilon'] = epsilon
        results = results.append(intermediate)
    if model_type == 'cnn':
        results.columns = ['original', 'adversary', 'truth', 'epsilon']
    elif model_type =='bcnn':
        results.columns = ['X', 'original', 'original_confidence', 'adversary', 'adversary_confidence', 'truth', 'epsilon']
    results.to_csv(outname, index=False)


def get_epsilon(filepath):
    path = 'results/experiment3/bcnn/'
    print(filepath)
    file = filepath.split(path, 1)[1]
    digits = [s for s in file if s.isdigit()]
    if len(digits)==3:
        return 0.01
    elif len(digits)==2:
        return float('.'.join(str(x) for x in digits))


if __name__ == '__main__':
    args = build_parser()
    if args.fgsm:
        merge_results(args.directory, args.outfile, args.forgot, args.modeltype)
