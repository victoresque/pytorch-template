import json
import time
from pathlib import Path
from datetime import datetime
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class Timer:
    def __init__(self):
        self.cache = None
        self.recording = False

    def tic(self):
        self.recording = True
        self.cache = datetime.now()

    def toc(self):
        assert self.recording, 'start timer by calling timer.tic() first.'
        self.recording = False
        return datetime.now() - self.cache


if __name__ == '__main__':
    timer = Timer()
    a = 0
    for i in range(3):

        timer.tic()
        time.sleep(1)
        # record = float(timer.toc())
        print(timer.toc())
        print(timer.toc())
