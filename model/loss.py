import torch.nn.functional as F
import torch.nn as nn
import torch
from .model_arc.decoder import *

# 交叉熵损失
def cross_entropy2d_loss(output, target):
    #样本权重设置
    weight = torch.tensor([0.9,0.1]).to(torch.device('cuda:0'))
    loss = nn.CrossEntropyLoss(weight=weight)
    # loss = nn.CrossEntropyLoss()
    return loss(output, target)

# Pointrend交叉熵损失
def point_cross_entropy_loss(result,target):
    #样本权重设置
    weight = torch.tensor([0.509871476,25.825493353]).to(torch.device('cuda:0'))
    pred = F.interpolate(result["coarse"], (512,512), mode="bilinear", align_corners=True)
    seg_loss = F.cross_entropy(pred, target, weight=weight, ignore_index=255)
    gt_points = point_sample(
            target.float().unsqueeze(1),
            result["points"],
            mode="nearest",
            align_corners=False
        ).squeeze_(1).long()
    points_loss = F.cross_entropy(result["rend"], gt_points, ignore_index=255)
    loss = seg_loss + points_loss
    return loss