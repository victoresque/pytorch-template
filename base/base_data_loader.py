
class BaseDataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def next_batch(self):
        return NotImplementedError

    def __len__(self):
        return NotImplementedError

