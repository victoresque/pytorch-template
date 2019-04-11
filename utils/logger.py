import os
import json
import yaml
import logging.config

def setup_logging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration
    """
    config_f = default_path
    value = os.getenv(env_key, None)
    if value:
        config_f = value
    if os.path.exists(config_f):
        with open(config_f, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


logging_level_dict = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}

def setup_logger(cls, verbose=0):
    logger = logging.getLogger(cls.__class__.__name__)
    if verbose not in logging_level_dict:
        raise KeyError('Verbose option {} for {} not valid. '
                        'Valid options are {}.'.format(
                            verbose, cls, logging_level_dict.keys()))
    logger.setLevel(logging_level_dict[verbose])
    return logger


setup_logging()