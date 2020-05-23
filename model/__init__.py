# -*- coding: UTF-8 -*-
"""
@Project -> File   ：cnn_models_comparation.pytorch -> __init__
@IDE    ：PyCharm
@Author ：QiangZiBro
@Date   ：2020/5/23 12:56 下午
@Desc   ： use factory method to add model,loss,Metrics,Optimizer
    Q: How to add a pytorch model?
    A: 1.edit a or  a set of models in a single file in `model/models/`, we may
    obtain a model by a class or function
    ```
    class ResNet():
        ...
    def resnet18(*kwargs):
        ...
    ```
       2.import the model in `model/__init__.py`
    ```
    from .ResNet import ResNet,resnet18
    ```
       3.change model name in a new json file
       Done~
"""
import torch
import model.models as module_models
import model.loss as module_loss
import model.metric as module_metric

__all__ = ["makeModel", "makeLoss", "makeMetrics", "makeOptimizer"]


def makeModel(config):
    return config.init_obj('arch', module_models)


def makeLoss(config):
    return getattr(module_loss, config['loss'])


def makeMetrics(config):
    return [getattr(module_metric, met) for met in config['metrics']]


def makeOptimizer(config, model):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    return optimizer
