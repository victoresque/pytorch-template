import os
from model.model import models
from model.loss import losses
from model.metric import metrics


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_model(config):
    model_name = config['arch']
    if model_name in models:
        return models[model_name]
    else:
        raise IOError(
            f"Model {model_name} not exists! "
            "Valid modeles are: {models.keys()}"
        )


def get_loss(config):
    loss_name = config['loss']
    if loss_name in losses:
        return losses[loss_name]
    else:
        raise IOError(
            f"Loss {loss_name} not exists! "
            "Valid losses are: {losses.keys()}"
        )


def get_metrics(config):
    metric_name = config['metric']
    if metric_name in metrics:
        return metrics[metric_name]
    else:
        raise IOError(
            f"Metric {metric_name} not exists! "
            "Valid metrics are: {metrics.keys()}"
        )
