import os
from datetime import datetime


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def time_stamp():
    return datetime.now().strftime('%m%d_%H%M%S')