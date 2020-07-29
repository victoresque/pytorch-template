# PyTorch Template Project
Simple project template for PyTorch deep Learning project.

<!-- TOC depthFrom:1 depthTo:6 orderedList:false -->

- [PyTorch Template Project](#pytorch-template-project)
    - [Requirements](#requirements)
    - [Features](#features)
    - [Folder Structure](#folder-structure)
    - [Usage](#usage)
        - [Hierarchical configurations with Hydra](#hierarchical-configurations-with-hydra)
        - [Using config files](#using-config-files)
        - [Checkpoints](#checkpoints)
        - [Resuming from checkpoints](#resuming-from-checkpoints)
        - [Using Multiple GPU](#using-multiple-gpu)
    - [Customization](#customization)
        - [Project initialization](#project-initialization)
        - [Custom CLI options](#custom-cli-options)
        - [Data Loader](#data-loader)
        - [Trainer](#trainer)
        - [Model](#model)
        - [Loss](#loss)
        - [Metrics](#metrics)
        - [Additional logging](#additional-logging)
        - [Testing](#testing)
        - [Validation data](#validation-data)
        - [Checkpoints](#checkpoints-1)
        - [Tensorboard Visualization](#tensorboard-visualization)
    - [Contribution](#contribution)
    - [TODOs](#todos)
    - [License](#license)

<!-- /TOC -->


## Requirements
* Python >= 3.6
* PyTorch >= 1.2
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))
* tqdm (Optional for `test.py`)
* hydra-core >= 1.0.0rc1

## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.yaml` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.

## Folder Structure
```yaml
  pytorch-template/
  ├── train.py                  # main script to start training.
  ├── evaluate.py               # script to evaluate trained model on testset.
  ├── conf # config files. explained in separated section below.
  │   └── ...
  ├── srcs # source code.
  │   ├── data_loader           # data loading, preprocessing
  │   │   └── data_loaders.py
  │   ├── model
  │   │   ├── loss.py
  │   │   ├── metric.py
  │   │   └── model.py
  │   ├── trainer               # customized class managing training process
  │   │   ├── base.py
  │   │   └── trainer.py
  │   ├── logger.py             # tensorboard, train / validation metric logging
  │   └── utils
  │       └── util.py
  ├── new_project.py            # script to initialize new project
  ├── requirements.txt
  ├── README.md
  └── LICENSE
```

## Usage
This repository itself is an working example project which trains simple model(LeNet) on Fashion-MNIST dataset.
Try `python train.py` to run code.

### Hierarchical configurations with Hydra
This repository is designed to be used with [Hydra](https://hydra.cc/) framework, which has useful key features as following.

- Hierarchical configuration composable from multiple sources
- Configuration can be specified or overridden from the command line
- Dynamic command line tab completion
- Run your application locally or launch it to run remotely
- Run multiple jobs with different arguments with a single command

`conf/` directory contains `.yaml`config files which are structured into multiple **config groups**.

```yaml
  conf/ # hierarchical, structured config files to be used with 'Hydra' framework
  ├── config.yaml               # main config file used for train.py
  ├── evaluate.yaml             # main config file used for evaluate.py
  ├── hparams                   # define global hyper-parameters
  │   └── lenet_baseline.yaml
  ├── dataset
  │   ├── mnist_test.yaml
  │   └── mnist_train.yaml
  ├── structure                 # define structure of NN to train
  │   └── mnist_lenet.yaml
  ├── status                    # select train/debug mode.
  │   ├── debug.yaml            #   debug mode runs faster, and don't use tensorboard
  │   └── train.yaml            #   train mode is default with full logging
  │
  └── hydra                     # configure hydra framework
      ├── job_logging           #   config for python logging module
      │   └── custom.yaml
      └── run/dir               #   setup working directory
          ├── job_timestamp.yaml
          └── no_chdir.yaml
```

At runtime, one file from each config group is selected and combined to be used as one global config.

```yaml
name: MnistLeNet # experiment name.

save_dir: models/
log_dir: ${name}/
resume:

# Global hyper-parameters defined in conf.hparams
#   you can change the values by either editing yaml file directly,
#   or using command line arguments, like `python3 train.py batch_size=128`
batch_size: 256
learning_rate: 0.001
weight_decay: 0
scheduler_step_size: 50
scheduler_gamma: 0.1

# configuration for data loading
data_loader:
  cls: srcs.data_loader.data_loaders.get_data_loaders
  params:
    data_dir: data/
    batch_size: ${batch_size}
    shuffle: true
    validation_split: 0.1
    num_workers: ${n_cpu}

arch:
  cls: srcs.model.model.MnistModel
  params:
    num_classes: 10
loss:
  cls: srcs.model.loss.nll_loss
optimizer:
  cls: torch.optim.Adam
  params:
    lr: ${learning_rate}
    weight_decay: ${weight_decay}
    amsgrad: true
lr_scheduler:
  cls: torch.optim.lr_scheduler.StepLR
  params:
    step_size: ${scheduler_step_size}
    gamma: ${scheduler_gamma}

metrics:
- cls: srcs.model.metric.accuracy
- cls: srcs.model.metric.top_k_acc

n_gpu: 1
n_cpu: 8
trainer:
  epochs: 20
  logging_step: 100
  verbosity: 2
  monitor: min loss/valid
  early_stop: 10
  tensorboard: true
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.yaml` files in `conf/` dir, then run:
  ```
  python train.py
  ```

### Checkpoints

```yaml
outputs/train/2020-07-29/12-44-37/
├── config.yaml # composed config file
├── epoch-results.csv # epoch-wise evaluation results
├── MnistLeNet/ # tensorboard log file
├── model
│   ├── checkpoint-epoch1.pth
│   ├── checkpoint-epoch2.pth
│   ├── ...
│   ├── model_best.pth # checkpoint with best score
│   └── model_latest.pth # checkpoint which is saved last
└── train.log
```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py resume=path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training(with DataParallel) by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py n_gpu=2
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=0,1 python train.py
  ```

## Customization

### Project initialization
Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file.

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the yaml file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target`
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.yaml --bs 256` runs training with options given in `config.yaml` except for the `batch size`
which is increased to 256 by command line options.


### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### Trainer
* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```yaml
  "metrics": ["accuracy", "top_k_acc"],
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```yaml
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'epoch_metrics': self.ep_metrics,
    'config': self.config
  }
  ```

### Tensorboard Visualization
This template supports Tensorboard visualization with `torch.utils.tensorboard`.

1. **Run training**

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

2. **Open Tensorboard server**

    Type `tensorboard --logdir outputs/train/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules.

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `srcs.logger.py` will track current steps.

## Contribution
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs
- [ ] Multi-GPU training with DistributedDataParallel
- [ ] Option to specify GPU indices to be used
- [ ] Option to keep top-k checkpoints only
- [ ] Simple unittest code for `nn.Module`

## License
This project is licensed under the MIT License. See  LICENSE for more details
