from src.utils.utility_funcs import *


if __name__ == '__main__':
    args = experiment_parser()
    if args.task == 0:
        exps = VisionExperiments(args)
    elif args.task == 1:
        exps = RLExperiments(args)
    else:
        raise ValueError('Incorrect Task Type Selected. Please specify either "vision" or "rl".')
    box_print('Conducting {} Experiments'.format(exps.meta_name))
    exps.run_experiment()
