import argparse
import torch.optim as optim
from models.model import Model
from models.loss import nll_loss
from models.metric import accuracy
from data_loader.data_loader import DataLoader
from trainers.trainer import Trainer
from logger.logger import Logger

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=5, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-v', '--verbosity', default=2, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save-dir', default='models/saved', type=str,
                    help='directory of saved model (default: models/saved)')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 5)')
parser.add_argument('--data-dir', default='data/datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU in case there\'s no GPU support')


def main(args):
    model = Model()
    model.summary()
    logger = Logger()

    loss = nll_loss
    metrics = [accuracy]
    optimizer = optim.Adam(model.parameters())
    data_loader = DataLoader(args.data_dir, args.batch_size)
    trainer = Trainer(model, data_loader, loss, metrics,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      logger=logger,
                      save_dir=args.save_dir,
                      save_freq=args.save_freq,
                      resume=args.resume,
                      verbosity=args.verbosity,
                      with_cuda=not args.no_cuda)
    trainer.train()
    logger.print()


if __name__ == '__main__':
    main(parser.parse_args())
