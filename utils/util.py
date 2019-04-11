import json
from pathlib import Path
from collections import OrderedDict


def ensure_dir(path):
    path = Path(path)
    if not path.is_dir():
        path.mkdir()

def read_json(filename):
    filename = Path(filename)
    with filename.open('rt') as handle:
        result = json.load(handle, object_pairs_hook=OrderedDict)
    return result
    
def write_json(obj, filename):
    filename = Path(filename)
    with filename.open('wt') as handle:
        json.dump(obj, handle, indent=4)