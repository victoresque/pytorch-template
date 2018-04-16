# PyTorch Template Project

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [PyTorch Template Project](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Usage](#usage)
	* [Folder Structure](#folder-structure)
	* [Customization](#customization)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss & Metrics](#loss-metrics)
			* [Multiple metrics](#multiple-metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoint naming](#checkpoint-naming)
	* [Contributing](#contributing)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgments](#acknowledgments)

<!-- /code_chunk_output -->

## Requirements
* Python 3.x
* PyTorch

## Features
* Clear folder structure which is suitable for many projects
* Separate `trainer`, `model`, and `data_loader` for more structured code
* `BaseDataLoader` handles batch loading, data shuffling, and validation data aplitting for you
* `BaseTrainer` handles checkpoint saving/loading, training process logging

## Usage
The code in this repo is an MNIST example of the template, try run:
```
python train.py
```
The default arguments list is shown below:
```
usage: train.py [-h] [-b BATCH_SIZE] [-e EPOCHS] [--resume RESUME]
               [--verbosity VERBOSITY] [--save-dir SAVE_DIR]
               [--save-freq SAVE_FREQ] [--data-dir DATA_DIR]
               [--validation-split VALIDATION_SPLIT] [--no-cuda]

PyTorch Template

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        mini-batch size (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        number of total epochs (default: 32)
  --resume RESUME       path to latest checkpoint (default: none)
  --verbosity VERBOSITY
                        verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)
  --save-dir SAVE_DIR   directory of saved model (default: saved)
  --save-freq SAVE_FREQ
                        training checkpoint frequency (default: 1)
  --data-dir DATA_DIR   directory of training/testing data (default: datasets)
  --validation-split VALIDATION_SPLIT
                        ratio of split validation data, [0.0, 1.0) (default: 0.1)
  --no-cuda             use CPU instead of GPU
```

## Folder Structure
```
pytorch-template/
│
├── base/ - abstract base classes
│   ├── base_data_loader.py - abstract base class for data loaders
│   ├── base_model.py - abstract base class for models
│   └── base_trainer.py - abstract base class for trainers
│
├── data_loader/ - anything about data loading goes here
│   └── data_loaders.py
│
├── datasets/ - default datasets folder
│
├── logger/ - for training process logging
│   └── logger.py
│
├── model/ - models, losses, and metrics
│   ├── modules/ - submodules of your model
│   ├── loss.py
│   ├── metric.py
│   └── model.py
│
├── saved/ - default checkpoints folder
│
├── trainer/ - trainers
│   └── trainer.py
│
└── utils/
    ├── util.py
    └── ...

```

## Customization
### Data Loader
* **Writing your own data loader**
  1. **Inherit ```BaseDataLoader```**

     ```BaseDataLoader``` handles:
     * Generating next batch
     * Data shuffling
     * Generating validation data loader ```BaseDataLoader.split_validation()```

  2. **Implementing abstract methods**

     There are some abstract methods you need to implement before using the methods in ```BaseDataLoader``` 
     * ```_pack_data()```: pack data members into a list of tuples
     * ```_unpack_data```: unpack packed data
     * ```_update_data```: updata data members
     * ```_n_samples```: total number of samples

* **DataLoader Usage**

  ```BaseDataLoader``` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to ```data_loader/data_loaders.py``` for an MNIST example

### Trainer
* **Writing your own trainer**
  1. **Inherit ```BaseTrainer```**

     ```BaseTrainer``` handles:
     * Training process logging
     * Checkpoint saving
     * Checkpoint resuming
     * Reconfigurable monitored value for saving current best 
       - Controlled by the arguments ```monitor``` and ```monitor_mode```, if ```monitor_mode == 'min'``` then the trainer will save a checkpoint ```model_best.pth.tar``` when ```monitor``` is a current minimum

  2. **Implementing abstract methods**

     You need to implement ```_train_epoch()``` for your training process, if you need validation then you can implement ```_valid_epoch()``` as in ```trainer/trainer.py```

* **Example**

  Please refer to ```trainer/trainer.py```

### Model
* **Writing your own model**
  1. **Inherit ```BaseModel```**

     ```BaseModel``` handles:
     * Inherited from ```torch.nn.Module```
     * ```summary()```: Model summary

  2. **Implementing abstract methods**

     Implement the foward pass method ```forward()```
     
* **Example**

  Please refer to ```model/model.py```

### Loss & Metrics
If you need to change the loss function or metrics, first ```import``` those function in ```train.py```, then modify:
```python
loss = my_loss
metrics = [my_metric]
```
They will appear in the logging during training
#### Multiple metrics
If you have multiple metrics for your project, just add them to the ```metrics``` list:
```python
loss = my_loss
metrics = [my_metric, my_metric2]
```
Additional metric will be shown in the logging
### Additional logging
If you have additional information to be logged, in ```_train_epoch()``` of your trainer class, merge them with ```log``` as shown below before returning:
```python
additional_log = {"gradient_norm": g, "sensitivity": s}
log = {**log, **additional_log}
return log
```
### Validation data
If you need to split validation data from a data loader, call ```BaseDataLoader.split_validation(validation_split)```, it will return a validation data loader, with the number of samples according to the specified ratio
**Note**: the ```split_validation()``` method will modify the original data loader
### Checkpoint naming
You can specify the name of the training session in ```train.py```
```python
training_name = type(model).__name__
```
Then the checkpoints will be saved in ```saved/training_name```

## Contributing
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

## TODOs
- [ ] Multi-GPU support
- [ ] `TensorboardX` support
- [ ] Support iteration-based training (instead of epoch)
- [ ] Load settings from `config` files
- [ ] Configurable logging layout
- [ ] Configurable checkpoint naming
- [ ] Options to save logs to file
- [ ] More clear trainer structure

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgments
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
