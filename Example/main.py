import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy

from runners.rs256_guided_diffusion import Diffusion

# Explanation: The import statements here bring in various modules such as argparse, traceback, and logging.
# Explanation: argparse is used for parsing command-line options; traceback helps with detailed stack trace output.
# Explanation: shutil, logging, yaml, sys, os, torch, numpy are used for file operations, logging, configuration loading, system calls, deep learning tasks, and numeric operations.
# Explanation: We also import torch.utils.tensorboard (aliased as tb) and copy for additional deep learning utilities and object duplication features.
# Explanation: Finally, from runners.rs256_guided_diffusion import Diffusion, which is the class handling the diffusion-based fluid reconstruction process.

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--repeat_run', type=int, default=1, help='Repeat run')
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--r', dest='reverse_steps', type=int, default=20, help='Revserse steps')
    parser.add_argument('--comment', type=str, default='', help='Comment')
    args = parser.parse_args()

    # Explanation: The parse_args_and_config function handles reading command-line arguments and merges them with a YAML config file.
    # Explanation: It sets up default values for seeds, sampling steps, reverse steps, and so on, then loads the specified config from the 'configs' directory.
    # Explanation: Next, the function checks the model type (conditional or otherwise) to decide how to name the output directory.
    # Explanation: The logging system is also set up here, with a file handler writing to 'logging_info.txt'.  
    # Explanation: A device (GPU / CPU) is chosen based on system availability, and random seeds are set for reproducibility.
    # Explanation: Finally, it returns the parsed arguments, the merged config, a logger object, and the directory path for logs.

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    os.makedirs(config.log_dir, exist_ok=True)
    if config.model.type == 'conditional':

        dir_name = 'recons_{}_t{}_r{}_w{}_smoothing{}'.format(config.data.data_kw,
                                                    args.t, args.reverse_steps,
                                                    config.sampling.guidance_weight,
                                                    config.data.smoothing)
    else:

        dir_name = 'recons_{}_t{}_r{}_lam{}_smoothing{}'.format(config.data.data_kw,
                                                    args.t, args.reverse_steps,
                                                    config.sampling.lambda_,
                                                    config.data.smoothing)

    if config.model.type == 'conditional':
        print('Use residual gradient guidance during sampling')
        dir_name = 'guided_' + dir_name
    elif config.sampling.lambda_ > 0:
        print('Use residual gradient penalty during sampling')
        dir_name = 'pi_' + dir_name
    else:
        print('Not use physical gradient during sampling')

    log_dir = os.path.join(config.log_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.yml'), 'w') as outfile:
        yaml.dump(config, outfile)

    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'logging_info'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config, logger, log_dir


def dict2namespace(config):
    # Explanation: The dict2namespace function recursively converts a nested dictionary structure into a Python namespace.  
    # Explanation: This allows attributes to be accessed like object fields, rather than dictionary keys.
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    # Explanation: In the main function, we first parse arguments and config via parse_args_and_config, then print out debugging info.
    # Explanation: A Diffusion object is created with the parsed arguments, config, logger, and log directory, and the reconstruct method is called.
    # Explanation: This reconstruct method presumably handles the main reconstruction or sampling logic for the fluid-based diffusion model.
    # Explanation: If any unexpected exception occurs, traceback.format_exc() is used to log the entire error stack for easier debugging.
    # Explanation: The function then returns 0, indicating a successful run, and the script ends by calling sys.exit(main()) when __main__ is invoked.
    args, config, logger, log_dir = parse_args_and_config()
    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print("<" * 80)

    try:
        runner = Diffusion(args, config, logger, log_dir)
        runner.reconstruct()

    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
