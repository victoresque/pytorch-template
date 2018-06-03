import numpy as np
import torch
from torch.autograd import Variable
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        Modify __init__() if you have additional arguments to pass.
    """
    def __init__(self, model, loss, metrics, data_loader, optimizer, epochs,
                 save_dir, save_freq, resume, device, verbosity, training_name='',
                 valid_data_loader=None, train_logger=None, writer=None, lr_scheduler=None, monitor='loss', monitor_mode='min'):
        super(Trainer, self).__init__(model, loss, metrics, data_loader, valid_data_loader, optimizer, epochs,
                                      save_dir, save_freq, resume, verbosity, training_name,
                                      device, train_logger, writer, monitor, monitor_mode)
        self.scheduler = lr_scheduler

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """
        self.model.train()
        self.model.to(self.device)

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.train_iter += 1
            self.writer.add_scalar(f'{self.training_name}/Train/loss', loss.item(), self.train_iter)
            for i, metric in enumerate(self.metrics):
                score = metric(output, target)
                self.writer.add_scalar(f'{self.training_name}/Train/{metric.__name__}', score, self.train_iter)    
                total_metrics[i] += score
                

            total_loss += loss.item()
            log_step = int(np.sqrt(self.batch_size))
            if self.verbosity >= 2 and batch_idx % log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.data_loader) * len(data),
                    100.0 * batch_idx / len(self.data_loader), loss.item()))

        avg_loss = total_loss / len(self.data_loader)
        avg_metrics = (total_metrics / len(self.data_loader)).tolist()
        log = {'loss': avg_loss, 'metrics': avg_metrics}

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_metrics = np.zeros(len(self.metrics))
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss(output, target)
                total_val_loss += loss.item()

                self.valid_iter += 1
                self.writer.add_scalar(f'{self.training_name}/Valid/loss', loss.item(), self.valid_iter)
                for i, metric in enumerate(self.metrics):
                    score = metric(output, target)
                    total_val_metrics[i] += score
                    self.writer.add_scalar(f'{self.training_name}/Valid/{metric.__name__}', score, self.valid_iter)

            avg_val_loss = total_val_loss / len(self.valid_data_loader)
            self.scheduler.step(avg_val_loss)
            avg_val_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()
        return {'val_loss': avg_val_loss, 'val_metrics': avg_val_metrics}
