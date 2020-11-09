import os
import signal
import torch
import torch.distributed as dist
from abc import abstractmethod, ABCMeta
from pathlib import Path
from shutil import copyfile
from numpy import inf

from srcs.utils import write_conf, is_master, get_logger
from srcs.logger import TensorboardWriter, EpochMetrics


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = get_logger('trainer')

        self.device = config.local_rank
        self.model = model.to(self.device)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.log_step = cfg_trainer['logging_step']

        # setup metric monitoring for monitoring model performance and saving best-checkpoint
        self.monitor = cfg_trainer.get('monitor', 'off')

        metric_names = ['loss'] + [met.__name__ for met in self.metric_ftns]
        self.ep_metrics = EpochMetrics(metric_names, phases=('train', 'valid'), monitoring=self.monitor)

        self.checkpt_top_k = cfg_trainer.get('save_topk', -1)
        self.early_stop = cfg_trainer.get('early_stop', inf)

        write_conf(self.config, 'config.yaml')

        self.start_epoch = 1
        self.checkpt_dir = Path(self.config.save_dir)
        log_dir = Path(self.config.log_dir)
        if is_master():
            self.checkpt_dir.mkdir()
            # setup visualization writer instance
            log_dir.mkdir()
            self.writer = TensorboardWriter(log_dir, cfg_trainer['tensorboard'])
        else:
            self.writer = TensorboardWriter(log_dir, False)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            self.ep_metrics.update(epoch, result)

            # print result metrics of this epoch
            max_line_width = max(len(line) for line in str(self.ep_metrics).splitlines())
            # divider ---
            self.logger.info('-'*max_line_width)
            self.logger.info(str(self.ep_metrics.latest()) + '\n')

            # check if model performance improved or not, for early stopping and topk saving
            is_best = False
            improved = self.ep_metrics.is_improved()
            if improved:
                not_improved_count = 0
                is_best = True
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop and is_master():
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))
                os.kill(os.getppid(), signal.SIGTERM)

            using_topk_save = self.checkpt_top_k > 0
            self._save_checkpoint(epoch, save_best=is_best, save_latest=using_topk_save)
            # keep top-k checkpoints only, using monitoring metrics
            if using_topk_save:
                self.ep_metrics.keep_topk_checkpt(self.checkpt_dir, self.checkpt_top_k)

            self.ep_metrics.to_csv('epoch-results.csv')

            # divider ===
            self.logger.info('='*max_line_width)
            dist.barrier()


    def _save_checkpoint(self, epoch, save_best=False, save_latest=True):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, save a copy of current checkpoint file as 'model_best.pth'
        :param save_latest: if True, save a copy of current checkpoint file as 'model_latest.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch_metrics': self.ep_metrics,
            'config': self.config
        }

        filename = str(self.checkpt_dir / f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(f"Model checkpoint saved at: \n    {self.config.cwd}/{filename}")
        if save_latest:
            latest_path = str(self.checkpt_dir / 'model_latest.pth')
            copyfile(filename, latest_path)
        if save_best:
            best_path = str(self.checkpt_dir / 'model_best.pth')
            copyfile(filename, best_path)
            self.logger.info(f"Renewing best checkpoint: \n    .../{best_path}")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = self.config.resume
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # TODO: support overriding monitor-metric config
        self.ep_metrics = checkpoint['epoch_metrics']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['_target_'] != self.config['optimizer']['_target_']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
