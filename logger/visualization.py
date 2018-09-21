import os
import importlib
from datetime import datetime


class WriterTensorboardX():
    def __init__(self, config):
        self.writer = None
        if config['visualization']['tensorboardX']:
            logdir = config['visualization']['log_dir']
            log_path = os.path.join(logdir, f"{config['name']}/{datetime.now().strftime('%y%m%d%H%M%S')}")
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                print('Package tensorboardX is not installed.')
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        if name in self.tensorboard_writer_ftns:
            # get add_something method of tensorboard summary writer
            attr = getattr(self.writer, name, None)
            # wrap default methods to be harmless when writer is not set
            def wrapper(tag, data, *args, **kwargs):
                if attr is None:
                    pass
                else:
                    attr(f'{self.mode}/{tag}', data, self.step, *args, **kwargs)
            return wrapper
        else:
            return super(WriterTensorboardX, self).__getattr__(name)
