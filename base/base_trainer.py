import os
import math
import logging
import torch
from utils.util import ensure_dir


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, epochs,
                 save_dir, save_freq, resume, verbosity, training_name,
                 with_cuda, train_logger=None, monitor='loss', monitor_mode='min'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.epochs = epochs
        self.save_freq = save_freq
        self.verbosity = verbosity
        self.training_name = training_name
        self.train_logger = train_logger
        self.with_cuda = with_cuda and torch.cuda.is_available()
        if with_cuda and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        assert monitor_mode == 'min' or monitor_mode == 'max'
        self.monitor_best = math.inf if monitor_mode == 'min' else -math.inf
        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(save_dir, training_name)
        ensure_dir(self.checkpoint_dir)
        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs+1):
            result = self._train_epoch(epoch)
            if self.train_logger:
                log = {'epoch': epoch}
                for key, value in result.items():
                    if key == 'metrics':
                        for i, metric in enumerate(self.metrics):
                            log[metric.__name__] = result['metrics'][i]
                    elif key == 'val_metrics':
                        for i, metric in enumerate(self.metrics):
                            log['val_'+metric.__name__] = result['val_metrics'][i]
                    else:
                        log[key] = value
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))
            if (self.monitor_mode == 'min' and result[self.monitor] < self.monitor_best)\
                    or (self.monitor_mode == 'max' and result[self.monitor] > self.monitor_best):
                self.monitor_best = result[self.monitor]
                self._save_checkpoint(epoch, result['loss'], save_best=True)
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, result['loss'])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, loss, save_best=False):
        """
        Saving checkpoints

        :param epoch: Current epoch number
        :param loss: Prefix of checkpoint name
        :param save_best: If True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'logger': self.train_logger,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
        }
        filename = os.path.join(self.checkpoint_dir,
                                'checkpoint_epoch{:02d}_loss_{:.5f}.pth.tar'.format(epoch, loss))
        torch.save(state, filename)
        if save_best:
            os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))
        else:
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
