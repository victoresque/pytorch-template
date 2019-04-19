import os
import logging
from pathlib import Path
from shutil import copyfile
from datetime import datetime
from logger import setup_logging
from utils import read_json


class ConfigParser:
    def __init__(self, args):
        args = args.parse_args()
        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume:
            self.resume = Path(args.resume)
            self.cfg_fname = self.resume.parent / 'config.json'
        else:
            self.resume = None
            self.cfg_fname = Path(args.config)
        assert self.cfg_fname is not None, "Configuration file need to be specified. Add '-c config.json', for example."
        
        self.config = read_json(self.cfg_fname)
        self.exper_name = self.config['name']

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])
        timestamp = datetime.now().strftime('%m%d_%H%M%S')# if timestamp else ''

        self.save_dir = save_dir / 'models' / self.exper_name / timestamp
        self.log_dir = save_dir / 'log' / self.exper_name / timestamp

        self.save_dir.mkdir(parents=True)
        self.log_dir.mkdir(parents=True)

        # copy the config file to the checkpoint dir # NOTE: str() can be removed from here on python 3.6 
        copyfile(str(self.cfg_fname), str(self.save_dir / 'config.json'))

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def initialize(self, name, module, *args):
        """
        return 'module.name' instance initialized with configuration given in file.
        equivalent to `module.name(*args, **kwargs_on_cfg)`
        """
        module_cfg = self[name]
        return getattr(module, module_cfg['type'])(*args, **module_cfg['args'])

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        logger = logging.getLogger(name)
        assert verbosity in self.log_levels, 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        logger.setLevel(self.log_levels[verbosity])
        return logger