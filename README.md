# PyTorch Template Project
Simple project base template for PyTorch deep Learning project.

<!-- TOC depthFrom:1 depthTo:6 orderedList:false -->

- [PyTorch Template Project](#pytorch-template-project)
    - [Installation](#installation)
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

## Installation
### Requirements
* Python >= 3.6
* PyTorch >= 1.2
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))
* tqdm
* hydra-core >= 1.0.3

### Features
* Simple and clear directory structure, suitable for most of deep learning projects.
* Hierarchical management of project configurations with [Hydra](https://hydra.cc/docs/intro).
* Advanced logging and monitoring for validation metrics. Automatic handling of model checkpoints.
* **Note**: This repository is detached from [victorisque/pytorch-template](https://github.com/victoresque/pytorch-template), in order to introduce advanced features rapidly without concerning much for backward compatibility.

### Folder Structure
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
This template itself is an working example project which trains a simple model(LeNet) on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.
Try `python train.py` to run training.

### Hierarchical configurations with Hydra
This repository is designed to be used with [Hydra](https://hydra.cc/) framework, which has useful key features as following.

- Hierarchical configuration composable from multiple sources
- Configuration can be specified or overridden from the command line
- Dynamic command line tab completion
- Run your application locally or launch it to run remotely
- Run multiple jobs with different arguments with a single command

Check [Hydra documentation](https://hydra.cc/), for more information.

`conf/` directory contains `.yaml`config files which are structured into multiple **config groups**.

```yaml
  conf/ # hierarchical, structured config files to be used with 'Hydra' framework
  ├── train.yaml                # main config file used for train.py
  ├── evaluate.yaml             # main config file used for evaluate.py
  ├── hparams                   # define global hyper-parameters
  │   └── lenet_baseline.yaml
  ├── data
  │   ├── mnist_test.yaml
  │   └── mnist_train.yaml
  ├── model                     # select NN architecture to train
  │   └── mnist_lenet.yaml
  ├── status                    # set train/debug mode.
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

### Using config files
Modify the configurations in `.yaml` files in `conf/` dir, then run:
  ```
  python train.py
  ```

At runtime, one file from each config group is selected and combined to be used as one global config.

```yaml
name: MnistLeNet # experiment name.

save_dir: models/
log_dir: ${name}/
resume:

# Global hyper-parameters defined in conf/hparams/
#   you can change the values by either editing yaml file directly,
#   or using command line arguments, like `python3 train.py batch_size=128`
batch_size: 256
learning_rate: 0.001
weight_decay: 0
scheduler_step_size: 50
scheduler_gamma: 0.1


# configuration for data loading.
data_loader:
  _target_: srcs.data_loader.data_loaders.get_data_loaders
  data_dir: data/
  batch_size: ${batch_size}
  shuffle: true
  validation_split: 0.1
  num_workers: ${n_cpu}

arch:
  _target_: srcs.model.model.MnistModel
  num_classes: 10
loss:
  _target_: srcs.model.loss.nll_loss
optimizer:
  _target_: torch.optim.Adam
  lr: ${learning_rate}
  weight_decay: ${weight_decay}
  amsgrad: true
lr_scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: ${scheduler_step_size}
  gamma: ${scheduler_gamma}

metrics:
- _target_: srcs.model.metric.accuracy
- _target_: srcs.model.metric.top_k_acc

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

`conf/hparams/lenet_baseline.yaml` contains

```yaml
batch_size: 256
learning_rate: 0.001
weight_decay: 0
scheduler_step_size: 50
scheduler_gamma: 0.1
```


Those config items containing `_target_` are designed to be used with `instantiate` function of Hydra. For example,
When your config looks like
```yaml
# @package _global_
classitem:
  _target_: location.to.class.definition
  arg1: 123
  arg2: 'example'
```

then usage of instantiate as

```python
example_object = instantiate(config.classitem)
```

is equivalent to

```python
from location.to.class import definition

example_object = definition(arg1=1, arg2='example')
```

This feature is especially useful, when you switch between multiple models with same interface(input, output),
like choosing ResNet or MobileNet for CNN backbone of detection model.
You can change architecture by simply using different config file, even not needing to importing both in code.

### Checkpoints

```yaml
# new directory with timestamp will be created automatically.
# if you enable debug mode by status=debug either in command line or main config,
# checkpoints will be saved under separate directory `outputs/debug`.
outputs/train/2020-07-29/12-44-37/
├── config.yaml # composed config file
├── epoch-results.csv # epoch-wise evaluation metrics
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
  python train.py resume=output/train/path/to/checkpoint.pth
  ```

### Using Multiple GPU
You can enable multi-GPU training(with DataParallel) by setting `n_gpu` argument of the config file to larger number. If configured to use smaller number of gpu than available, first n devices will be used by default. When you want to run multiple instances of training on larger maching, specify indices of available GPUs by cuda environmental variable.
  ```bash
  # assume running on a machine with 4 GPUs.
  python train.py n_gpu=2 # This will use first two GPU, which are on index 0 and 1
  CUDA_VISIBLE_DEVICES=2,3 python train.py n_gpu=2 # This will use remaining 2 GPUs on index 2 and 3
  ```

## Customization

### Project initialization
Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file.


### Data Loader
* **Writing your own data loader**

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
<!-- deprecated -->
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
- [ ] Support DistributedDataParallel
- [x] Option to keep top-k checkpoints only
- [ ] Simple unittest code for `nn.Module` and others

## License
This project is licensed under the MIT License. See  LICENSE for more details
