import numpy as np
import torch
from torch.autograd import Variable
from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, data_loader, loss, metrics, optimizer, epochs,
                 save_dir, save_freq, resume, with_cuda, verbosity, logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, epochs,
                                      save_dir, save_freq, resume, verbosity, logger)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.with_cuda = with_cuda

    def _train_epoch(self, epoch):
        n_batch = len(self.data_loader)
        self.model.train()
        if self.with_cuda:
            self.model.cuda()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx in range(n_batch):
            data, target = self.data_loader.next_batch()
            data, target = torch.FloatTensor(data), torch.LongTensor(target)
            data, target = Variable(data), Variable(target)
            if self.with_cuda:
                data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            for i, metric in enumerate(self.metrics):
                y_output = output.data.cpu().numpy()
                y_output = np.argmax(y_output, axis=1)
                y_target = target.data.cpu().numpy()
                total_metrics[i] += metric(y_output, y_target)

            total_loss += loss.data[0]
            log_step = int(np.sqrt(self.batch_size))
            if self.verbosity >= 2 and batch_idx % log_step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), n_batch * len(data),
                    100.0 * batch_idx / n_batch, loss.data[0]))

        avg_loss = total_loss / n_batch
        avg_metrics = (total_metrics / n_batch).tolist()
        return avg_loss, avg_metrics
