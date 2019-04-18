import yaml
import logging
import logging.config
from pathlib import Path


def setup_logging(save_dir, log_config='logger/logger_config.yaml', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        with log_config.open('rt') as f:
            config = yaml.safe_load(f.read())

        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
