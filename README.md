# PyTorch Template Project
A simple template project using PyTorch which can be modified to fit many deep learning projects.

## Basic Usage
The code in this repo is an MNIST example of the template, try run:
```
python main.py
```
The default arguments list is shown below:
```
usage: main.py [-h] [-b BATCH_SIZE] [-e EPOCHS] [--resume RESUME]
               [--verbosity VERBOSITY] [--save-dir SAVE_DIR]
               [--save-freq SAVE_FREQ] [--data-dir DATA_DIR]
               [--validation-split VALIDATION_SPLIT] [--no-cuda]

PyTorch Template

optional arguments:
  -h, --help    show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        mini-batch size (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        number of total epochs (default: 32)
  --resume RESUME
                        path to latest checkpoint (default: none)
  --verbosity VERBOSITY
                        verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)
  --save-dir SAVE_DIR
                        directory of saved model (default: model/saved)
  --save-freq SAVE_FREQ
                        training checkpoint frequency (default: 1)
  --data-dir DATA_DIR
                        directory of training/testing data (default: datasets)
  --validation-split VALIDATION_SPLIT
                        ratio of split validation data, [0.0, 1.0) (default: 0.0)
  --no-cuda   use CPU in case there's no GPU support
```
You can add your own arguments.

## Structure
```
├──  base/ - abstract base classes
│   ├── base_data_loader.py - abstract base class for data loaders.
│   ├── base_model.py - abstract base class for models.
│   └── base_trainer.py - abstract base class for trainers
│
├── data_loader/ - anything about data loading goes here
│   └── data_loader.py
│
├── datasets/ - default dataset folder
│
├── logger/ - for training process logging
│   └── logger.py
│
├── model/ - models, losses, and metrics
│   ├── modules/ - submodules of your model
│   ├── saved/ - default checkpoint folder
│   ├── loss.py
│   ├── metric.py
│   └── model.py
│
├── trainer/ - trainers for your project
│   └── trainer.py
│
└── utils
     ├── logger.py
     └── any_other_utils_you_need

```

## Customization
### Data loading
You can customize data loader to fit your project, just modify ```data_loader/data_loader.py``` or add other files.
### Model
Implement your model under ```model/```
### Loss/Metrics
If you need to change the loss function or metrics, first ```import``` those function in ```main.py```, then modify this part:
```python
loss = my_loss
metrics = [my_metric]
```
You'll see the logging has changed during training:
```
⋯
Train Epoch: 1 [53920/53984 (100%)] Loss: 0.033256
{'epoch': 1, 'loss': 0.14182623870152963, 'my_metric': 0.9568761114404268, 'val_loss': 0.06394806604976841, 'val_my_metric': 0.9804478609625669}
Saving checkpoint: model/saved/Model_checkpoint_epoch01_loss_0.14183.pth.tar ...
Train Epoch: 2 [0/53984 (0%)] Loss: 0.013225
⋯
```
#### Multiple metrics
If you have multiple metrics in your project, just add it to the ```metrics``` list:
```python
loss = my_loss
metrics = [my_metric, my_metric2]
```
Now the logging shows two metrics:
```
⋯
Train Epoch: 1 [53920/53984 (100%)] Loss: 0.003278
{'epoch': 1, 'loss': 0.13541310020907665, 'my_metric': 0.9590804682868999, 'my_metric2': 1.9181609365737997, 'val_loss': 0.05264156081223173, 'val_my_metric': 0.9837901069518716, 'val_my_metric2': 1.9675802139037433}
Saving checkpoint: model/saved/Model_checkpoint_epoch01_loss_0.13541.pth.tar ...
Train Epoch: 2 [0/53984 (0%)] Loss: 0.023072
⋯
```
Currently the name shown in log is the name of the function.
### Validation data
If you have separate validation data, try implement another data loader for validation, otherwise if you just want to split validation data from training data, try pass ```--validation-split 0.1```, in some cases you might need to modify ```utils/util.py```

## Contributing
Feel free to contribute any sort of function or enhancement, here the coding style follows PEP8

## Acknowledgments
This project is heavily inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template), be sure to star it!