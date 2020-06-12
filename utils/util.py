import hydra
from importlib import import_module
from functools import partial, update_wrapper
from itertools import repeat


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def instantiate(config, *args, is_func=False, **kwargs):
    """
    wrapper function for hydra.utils.instantiate.
    1. return None if config.class is None
    2. return function handle if is_func is True
    """
    assert 'cls' in config, f'Config should have \'cls\' and \'params\' for class instantiation.'
    if config['cls'] is None:
        return None
    if is_func:
        # get function handle
        modulename, funcname = config['cls'].rsplit('.', 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # make partial function with arguments given in config, code
        kwargs.update(config.get('params', {}))
        partial_func = partial(func, *args, **kwargs)

        # update original function's __name__ and __doc__ to partial function
        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(config, *args, **kwargs)
