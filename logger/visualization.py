from datetime import datetime
import importlib


class WriterTensorboardX():
    def __init__(self, config):
        self.writer = None
        if config['visualization']['tensorboardX']:
            log_path = f"saved/{config['name']}/{datetime.now().strftime('%y%m%d%H%M%S')}"
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                print('Package tensorboardX is not installed.')
        self.step = 0
        self.mode = ''

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def add_image(self, tag, image):
        if self.writer is None:
            pass
        else:
            self.writer.add_image(f'{self.mode}/{tag}', image, self.step)

    def add_scalar(self, tag, data):
        if self.writer is None:
            pass
        else:
            self.writer.add_scalar(f'{self.mode}/{tag}', data, self.step)

    def add_scalars(self, tag, data):
        if self.writer is None:
            pass
        else:
            self.writer.add_scalars(f'{self.mode}/{tag}', data, self.step)   

    def add_audio(self, tag, audio, sample_rate=44100):
        if self.writer is None:
            pass
        else:
            self.writer.add_audio(f'{self.mode}/{tag}', audio, self.step, sample_rate=sample_rate)

    def add_text(self, tag, data):
        if self.writer is None:
            pass
        else:
            self.writer.add_text(f'{self.mode}/{tag}', data, self.step)   

    def add_histogram(self, tag, data):
        if self.writer is None:
            pass
        else:
            self.writer.add_histogram(f'{self.mode}/{tag}', data, self.step)

    def add_pr_curve(self, tag, pred, data=None):
        if self.writer is None:
            pass
        else:
            self.writer.add_pr_curve(f'{self.mode}/{tag}', pred, data, self.step)

    def add_embedding(self, tag, features, metadata=None, label_img=None):
        if self.writer is None:
            pass
        else:
            self.writer.add_embedding(features, metadata=metadata, label_img=label_img)


if __name__ == '__main__':
    pass
