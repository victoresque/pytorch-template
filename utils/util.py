import os
import json
from collections import OrderedDict


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_json(filename):
    with open(filename, 'rt') as handle:
        result = json.load(handle, object_pairs_hook=OrderedDict)
    return result
    
def write_json(obj, filename):
    with open(filename, 'wt') as handle:
        json.dump(obj, handle, indent=4)