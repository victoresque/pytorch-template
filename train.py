import argparse
import logging
import torch.optim as optim
from model.model import MnistModel
from model.loss import my_loss
from model.metric import my_metric, my_metric2
from data_loader import MnistDataLoader
from trainer import Trainer
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=32, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=2, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save-dir', default='saved', type=str,
                    help='directory of saved model (default: saved)')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 1)')
parser.add_argument('--data-dir', default='datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--validation-split', default=0.1, type=float,
                    help='ratio of split validation data, [0.0, 1.0) (default: 0.1)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU instead of GPU')


def main(args):
    # Model
    model = MnistModel()
    model.summary()

    # A logger to store training process information
    train_logger = Logger()

    # Specifying loss function, metric(s), and optimizer
    loss = my_loss
    metrics = [my_metric, my_metric2]
    optimizer = optim.Adam(model.parameters())

    # Data loader and validation split
    data_loader = MnistDataLoader(args.data_dir, args.batch_size)
    valid_data_loader = data_loader.split_validation(args.validation_split)

    # An identifier for this training session
    training_name = type(model).__name__

    # Trainer instance
    trainer = Trainer(model, loss, metrics,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      train_logger=train_logger,
                      save_dir=args.save_dir,
                      save_freq=args.save_freq,
                      resume=args.resume,
                      verbosity=args.verbosity,
                      training_name=training_name,
                      with_cuda=not args.no_cuda,
                      monitor='loss',
                      monitor_mode='min')

    # Start training!
    trainer.train()

    # See training history
    print(train_logger)


if __name__ == '__main__':
    main(parser.parse_args())
