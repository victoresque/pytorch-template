import argparse
import logging
import torch.optim as optim
from model.model import *
from model.loss import *
from model.metric import *
from data_loader import MnistDataLoader
from trainer import Trainer
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')


def main(config, resume):
    train_logger = Logger()

    data_loader = MnistDataLoader(**config['data_loader'])
    valid_data_loader = data_loader.split_validation(**config['validation'])

    model = eval(config['arch'])(config['model'])
    model.summary()

    optimizer = getattr(optim, config['optimizer_type'])(model.parameters(), **config['optimizer'])
    loss = eval(config['loss'])
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = Trainer(model, loss, metrics,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      optimizer=optimizer,
                      train_logger=train_logger,
                      training_name=config['name'],
                      with_cuda=config['cuda'],
                      resume=resume,
                      config=config,
                      **config['trainer'])

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()
    assert (args.config is None) != (args.resume is None), "Specify exactly one of --config and --resume"
    if args.resume is not None:
        import torch
        config = torch.load(args.resume)['config']
    else:
        import json
        import os
        config = json.load(open(args.config))
        assert not os.path.exists(os.path.join(config['trainer']['save_dir'], config['name']))

    main(config, args.resume)
