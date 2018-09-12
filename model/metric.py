import numpy as np


def get_metric_functions(metric_names):
    try:
        metric_fns = [eval(metric) for metric in metric_names]
    except NameError as e:
        raise NameError(f"One of metric functions ({metric_names}) not found.")

    return metric_fns


def my_metric(y_input, y_target):
    assert len(y_input) == len(y_target)
    correct = 0
    for y0, y1 in zip(y_input, y_target):
        if np.array_equal(y0, y1):
            correct += 1
    return correct / len(y_input)


def my_metric2(y_input, y_target):
    assert len(y_input) == len(y_target)
    correct = 0
    for y0, y1 in zip(y_input, y_target):
        if np.array_equal(y0, y1):
            correct += 1
    return correct / len(y_input) * 2
