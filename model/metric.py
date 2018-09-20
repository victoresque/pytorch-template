import torch


def get_metric_functions(metric_names):
    try:
        metric_fns = [eval(metric) for metric in metric_names]
    except NameError as e:
        raise NameError(f"One of metric functions ({metric_names}) not found.")

    return metric_fns

def my_metric(output, target):
    pred = torch.argmax(output, dim=1)
    assert pred.shape[0] == len(target)
    correct = 0
    correct += torch.sum(pred == target).item()
    return correct / len(target)

def my_metric2(output, target, k=3):
    pred = torch.topk(output, k, dim=1)[1]
    assert pred.shape[0] == len(target)
    correct = 0
    for i in range(k):
        correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
