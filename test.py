import os
import argparse
import torch
import json
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
from utils import setup_logging, setup_logger


class TestRunner:

    def run(self, config, resume):
        setup_logging(config)
        logger = setup_logger(self, verbosity=config['testing']['verbosity'])

        logger.info('Using config: \n{}'.format(json.dumps(config, indent=4)))

        logger.info('Setting up data_loader instance...')
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=512,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2
        )

        logger.info('Building model architecture...')
        model = get_instance(module_arch, 'arch', config)
        logger.info('Using: \n{}'.format(model))

        logger.info('Loading function handles for loss and metrics...')
        loss_fn = getattr(module_loss, config['loss'])
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        logger.info('Loading state: {}'.format(resume))
        checkpoint = torch.load(resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))

        logger.info('Starting...')
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                #
                # save sample images, or do something with output here
                #

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
        logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    TestRunner().run(config, args.resume)
