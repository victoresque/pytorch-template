# PyTorch Template Project
PyTorch deep learning project made easy.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [PyTorch Template Project](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
	* [Customization](#customization)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss and metrics](#loss-and-metrics)
			* [Multiple metrics](#multiple-metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [TensorboardX Visualization](#tensorboardx-visualization)
	* [Contributing](#contributing)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgments](#acknowledgments)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5
* PyTorch >= 0.4

If TensorboardX is used:

* tensorboard >= 1.7.0
* tensorboardX >= 1.2

## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - example main
  ├── config.json - example config file
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for data loaders
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer.py - abstract base class for trainers
  │
  ├── data_utilsr/ - anything about data loading goes here
  │   ├── data_loaders.py - data loaders (both training and validation)
  │   ├── datasets.py - dataset class instances 
  │   └── transforms.py - handles composing transforms (for datasets)
  │
  ├── datasets/ - default location for datasets files
  │
  ├── logger/ - for training process logging
  │   └── logger.py
  │   └── visualization.py
  │
  ├── model/ - models, losses, and metrics
  │   ├── modules/ - submodules of your model
  │   ├── loss.py
  │   ├── metric.py
  │   └── model.py
  │
  ├── saved/ - default checkpoints folder
  │   └── runs/ - default logdir for tensorboardX
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  └── utils/
      ├── util.py
      └── ...
  ```

## Usage
The code in this repo is an MNIST example of the template.

### Config file format
Config files are in `.json` format:
  ```
  {
    "name": "Mnist_LeNet",        // training session name
    "cuda": true,                 // use cuda

	"dataset": {				  // Contains all dataset settings
        "type": "MnistDataset",   // Match name in 'data_utils/datasets.py'
        "transforms": [			  // List of valid PyTorch transforms that are to be applied
            {"op": "ToTensor"},   // to the data samples
            {"op": "Normalize",
                "mean": "(0.1307,)",
                "std": "(0.3081,)"}
        ],
        "kwargs": {				  // Arguments passed as **kwargs to the dataset class initialization
            "root": "./datasets/mnist/",
            "train": true,
            "download": true
        }
    },
    "data_loader": {              // Contains all data loader settings
        "type": "MnistDataLoader",// Either 'PyTorch' (for default) or name of data loader
                                  // class implemented in 'data_utils/data_loaders.py'
        "shuffle_data": true,     // Shuffle data in data loader per epoch (NB: only valid for
                                  // custom loaders. **PyTorch data loader will alway shuffle**)                                  
        "train": {                // Training parameters
            "kwargs": {           // Arguments passed as **kwargs to training set data loader initialization
                "batch_size": 32
            }
        },
        "validation": {           // Validation parameters
            "split": 0.1,         // Fraction of samples used for validation set
            "kwargs": {           // Arguments passed as **kwargs to validation set data loader initialization
                "batch_size": 32
            }
        }
    },
    "optimizer_type": "Adam",
    "optimizer": {
        "lr": 0.001,              // (optional) learning rate
        "weight_decay": 0         // (optional) weight decay
    },
    "loss": "NLLLoss",            // loss
    "loss_args": {
        "reduction": "elementwise_mean"
    },                            // elements in "loss_args" will be passed as kwargs to loss object
    "metrics": [                  // metrics
      "my_metric",
      "my_metric2"
    ],
    "trainer": {
        "epochs": 1000,           // number of training epochs
        "save_dir": "saved/",     // checkpoints are saved in save_dir/name
        "save_freq": 1,           // save checkpoints every save_freq epochs
        "verbosity": 2,           // 0: quiet, 1: per epoch, 2: full
        "monitor": "val_loss",    // monitor value for best model
        "monitor_mode": "min"     // "min" if monitor value the lower the better, otherwise "max" 
    },
    "visualization":{
        "tensorboardX": false,    // enable tensorboardX visualization support
        "log_dir": "saved/runs"   // directory to save log files for visualization
    },
    "arch": "MnistModel",         // model architecture
    "model": {}                   // model configs
  }
  ```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

## Customization
### Data Loader
**NB:** Work in progress to enable selecting the standard PyTorch dataloader
from the configuration file. This is done by setting:

    ["data_loader"]["type"]: "PyTorch"

Note that the implementation usage of the PyTorch data loader use
'SubsetRandomSampler', which shuffles the dataset in each epoch.

* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is similar to `torch.utils.data.DataLoader`, you can use either of them.

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
    * Reconfigurable monitored value for saving current best
      * Controlled by the configs `monitor` and `monitor_mode`, if `monitor_mode == 'min'` then the trainer will save a checkpoint `model_best.pth.tar` when `monitor` is a current minimum

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `summary()`: Model summary

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss
Valid values for 'loss' in the configuration file are all class names inside 'torch.nn.modules.loss' (see [PyTorch documentation](https://pytorch.org/docs/stable/nn.html#loss-functions)). 
The configuration key 'loss_args' is a dictionary that is passed dictionary that is passed as keyword arguments during loss object
initialization.

Custom loss functions can be implemented in 'model/loss.py', however using them currently requires explicitly importing the custom loss function manually in 'train.py'.

#### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["my_metric", "my_metric2"],
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log = {**log, **additional_log}
  return log
  ```
  
### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, it will return a validation data loader, with the number of samples according to the specified ratio in your config file.

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name`.

The config file is saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'logger': self.train_logger,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.monitor_best,
    'config': self.config
  }
  ```

### TensorboardX Visualization
This template supports [TensorboardX](https://github.com/lanpa/tensorboardX) visualization.
* **TensorboardX Usage**

1. **Install**

    Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Set `tensorboardX` option in config file true.

3. **Open tensorboard server** 

    Type `tensorboard --logdir saved/runs/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, and input image will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` module. 

**Note**: You don't have to specify current steps, since `WriterTensorboardX` class defined at `logger/visualization.py` will track current steps.

## Contributing
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs
- [ ] Iteration-based training (instead of epoch-based)
- [ ] Multi-GPU support
- [ ] Multiple optimizers
- [ ] Configurable logging layout, checkpoint naming
- [ ] `visdom` logger support
- [x] `tensorboardX` logger support
- [x] Update the example to PyTorch 0.4
- [x] Learning rate scheduler
- [x] Deprecate `BaseDataLoader`, use `torch.utils.data` instesad
- [x] Load settings from `config` files

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgments
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
