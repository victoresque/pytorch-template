import numpy as np
import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pathlib import Path
from srcs.trainer import Trainer
from srcs.utils import instantiate, get_logger


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train_worker(config):
    logger = get_logger('train')
    # setup data_loader instances
    data_loader, valid_data_loader = instantiate(config.data_loader)

    # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(model)
    logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

    # get function handles of loss and metrics
    criterion = instantiate(config.loss, is_func=True)
    metrics = [instantiate(met, is_func=True) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()

def init_worker(rank, ngpus, working_dir, config):
    # initialize training config
    config = OmegaConf.create(config)
    config.local_rank = rank
    config.cwd = working_dir
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:34567',
        world_size=ngpus,
        rank=rank)
    torch.cuda.set_device(rank)

    # start training processes
    train_worker(config)

@hydra.main(config_path='conf/', config_name='train')
def main(config):
    n_gpu = torch.cuda.device_count()
    assert n_gpu, 'Can\'t find any GPU device on this machine.'

    working_dir = str(Path.cwd().relative_to(hydra.utils.get_original_cwd()))

    if config.resume is not None:
        config.resume = hydra.utils.to_absolute_path(config.resume)
    config = OmegaConf.to_yaml(config, resolve=True)
    torch.multiprocessing.spawn(init_worker, nprocs=n_gpu, args=(n_gpu, working_dir, config))

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
