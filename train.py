import os
import json
import logging
import argparse
import torch
from data_loader import data_loaders
from model import loss as losses
from model import metric
from model import model as models
from trainer import Trainer
from logger import Logger


logging.basicConfig(level=logging.INFO, format='')


def main(config, resume):
    train_logger = Logger()

    data_loader = getattr(data_loaders, config['data_loader']['type'])(config)
    valid_data_loader = data_loader.split_validation()

    model = getattr(models, config['arch'])(config)
    model.summary()

    loss = getattr(losses, config['loss']['type'])
    metrics = [getattr(metric, met) for met in config['metrics']]

    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-f', '--force', action='store_true',
                        help='')
    arg_group = parser.add_mutually_exclusive_group(required=True)
    arg_group.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    arg_group.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    if args.force:

        print('force')
    else:
        print('no force')

    if args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if os.path.exists(path):
            if not args.force:
                raise AssertionError("Path {} already exists! \nAdd '-f' or '--force' option to start training anyway, or change 'name' given in config file.".format(path))

    main(config, args.resume)
