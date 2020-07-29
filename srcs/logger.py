import logging
import pandas as pd
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


logger = logging.getLogger('logger')

class TensorboardWriter():
    def __init__(self, log_dir, enabled):
        self.writer = SummaryWriter(log_dir) if enabled else None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

        self.step = 0

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.timer = datetime.now()

    def set_step(self, step):
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            attr = getattr(self.writer, name)
            return attr

class BatchMetrics:
    def __init__(self, *keys, postfix='', writer=None):
        self.writer = writer
        self.postfix = postfix
        if postfix:
            keys = [k+postfix for k in keys]
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.postfix:
            key = key + self.postfix
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        if self.postfix:
            key = key + self.postfix
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

class EpochMetrics:
    def __init__(self, metric_names, phases=('train', 'valid'), monitoring='off'):
        # setup pandas DataFrame with hierarchical columns
        columns = tuple(product(metric_names, phases))
        self._data = pd.DataFrame(columns=columns) # TODO: add epoch duration

        self.monitor_mode, self.monitor_metric = self._parse_monitoring_mode(monitoring)

    def _parse_monitoring_mode(self, monitor_mode):
        if monitor_mode == 'off':
            return 'off', None
        else:
            monitor_mode, monitor_metric = monitor_mode.split()
            monitor_metric = tuple(monitor_metric.split('/'))
            assert monitor_mode in ['min', 'max']
        return monitor_mode, monitor_metric

    def is_improved(self):
        if self.monitor_mode == 'off':
            return True
        try:
            # check whether model performance improved or not
            current_score = self._data[self.monitor_metric][-1:].item()
            improved = (self.monitor_mode == 'min' and current_score <= self._data[self.monitor_metric].min() or \
                       (self.monitor_mode == 'max' and current_score >= self._data[self.monitor_metric].max()))
        except KeyError:
            logger.warning("Warning: Metric '{}' is not found. "
                           "Model performance monitoring is disabled.".format(self.monitor_metric))
            self.monitor_mode = 'off'
            improved = True
        return improved

    def update(self, epoch, result):
        self._data.loc[f'epoch-{epoch}'] = {tuple(k.split('/')):v for k, v in result.items()}

    def latest(self):
        return self._data[-1:]

    def to_csv(self, save_path=None):
        self._data.to_csv(save_path)

    def __str__(self):
        return str(self._data)
