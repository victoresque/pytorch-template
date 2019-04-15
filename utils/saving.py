from os.path import join
import datetime

from utils.util import ensure_dir


def arch_path(config):
    return ensure_dir(join(config['save_dir'], config['name']))


def arch_datetime_path(config):
    start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    return ensure_dir(join(arch_path(config), start_time))


def log_path(config):
    return ensure_dir(join(arch_path(config), 'logs'))


def trainer_paths(config):
    """Returns the paths to save checkpoints and tensorboard runs. eg.

    saved/Mnist_LeNet/<start time>/checkpoints
    saved/Mnist_LeNet/<start time>/runs
    """
    arch_datetime = arch_datetime_path(config)
    return (
        ensure_dir(join(arch_datetime, 'checkpoints')), 
        ensure_dir(join(arch_datetime, 'runs'))
    )
