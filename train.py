import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import setup_logging, setup_logger


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class TrainingRunner:

    def run(self, config, resume):
        setup_logging(config)
        logger = setup_logger(self, verbosity=config['training']['verbosity'])

        logger.info('Using config: \n{}'.format(json.dumps(config, indent=4)))

        logger.info('Setting up data_loader instance...')
        data_loader = get_instance(module_data, 'data_loader', config)
        valid_data_loader = data_loader.split_validation()

        logger.info('Building model architecture...')
        model = get_instance(module_arch, 'arch', config)
        logger.info('Using: \n{}'.format(model))

        logger.info('Loading function handles for loss and metrics...')
        loss = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        logger.info('Initialising optimiser and scheduler...')
        # delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

        logger.info('Initialising Trainer...')
        trainer = Trainer(model, loss, metrics, optimizer,
                        resume=resume,
                        config=config,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

        trainer.train()
        logger.info('Finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = json.load(handle)

    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)['config']
        
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    TrainingRunner().run(config, args.resume)
