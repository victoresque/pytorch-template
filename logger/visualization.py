import importlib
from utils import Timer


class TensorboardWriter():
    def __init__(self, log_dir, logger, config):
        self.writer = None

        self.viz_methods = ["pytorch_tensorboard", "tensorboardX"]
        self.selected_module = ""

        if config["enabled"]:
            log_dir = str(log_dir)

            # Try to find a vizualization writer.
            succeeded = False
            for module in config["modules"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    self.selected_module = module
                    logger.info("Selected Tensorboard writer {}".format(module))
                    break
                except ImportError:
                    logger.warning("{} failed to load.".format(module))
                    succeeded = False

            if (not succeeded):
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install either TensorboardX with 'pip install tensorboardx', " \
                    "install PyTorch 1.1 for using 'torch.utils.tensorboard' or turn off the option in " \
                    "the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = [
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        ]
        self.tag_mode_exceptions = ['add_histogram', 'add_embedding']
        
        if(self.selected_module == "pytorch.utils.tensorboard"):
            self.tb_writer_ftns = self.tb_writer_ftns + self.tag_mode_exceptions
            self.tag_mode_exceptions = []
            
        self.timer = Timer()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer.reset()
        else:
            duration = self.timer.check()
            self.add_scalar('steps_per_sec', 1 / duration)

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
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
