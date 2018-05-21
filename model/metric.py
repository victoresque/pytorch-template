import numpy as np
import torch


def accuracy(y_input, y_target):
    pred = torch.argmax(y_input, dim=1)
    assert pred.shape[0] == y_target.shape[0]
    correct = torch.sum(pred == y_target, dtype=torch.float32)
    return (correct / y_target.shape[0]).item()

def top3(y_input, y_target):
    _, pred = torch.topk(y_input, 3, dim=1)
    assert pred.shape[0] == y_target.shape[0]
    score = torch.zeros(1).cuda()
    for i in range(3):
        score += torch.sum(pred[:, i] == y_target, dtype=torch.float32) / (i + 1)
    return (score / y_target.shape[0]).item()


# def my_metric(y_input, y_target):
#     assert len(y_input) == len(y_target)
#     correct = 0
#     for y0, y1 in zip(y_input, y_target):
#         if np.array_equal(y0, y1):
#             correct += 1
#     return correct / len(y_input)


# def my_metric2(y_input, y_target):
#     assert len(y_input) == len(y_target)
#     correct = 0
#     for y0, y1 in zip(y_input, y_target):
#         if np.array_equal(y0, y1):
#             correct += 1
#     return correct / len(y_input) * 2

