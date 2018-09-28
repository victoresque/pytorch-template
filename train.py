import os
import json
import logging
import argparse
import torch
from model.model import get_model_instance
from model.loss import get_loss_function
from model.metric import get_metric_functions
from data_utils.data_loaders import get_data_loaders
from data_utils.datasets import get_dataset
from trainer import Trainer
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')


def main(config, resume):
    train_logger = Logger()

    dataset = get_dataset(ds_config=config['dataset'])

    loader_train, loader_val = get_data_loaders(config['data_loader'],
                                                dataset,
                                                config_full=config)

    model = get_model_instance(model_arch=config['arch'],
                               model_params=config['model'])
    model.summary()

    loss = get_loss_function(config['loss'], **config['loss_args'])
    metrics = get_metric_functions(config['metrics'])

    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=loader_train,
                      valid_data_loader=loader_val,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    arg_group = parser.add_mutually_exclusive_group(required=True)
    arg_group.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    arg_group.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        assert not os.path.exists(path), "Path {} already exists!".format(path)

    main(config, args.resume)
