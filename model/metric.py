import torch

def PA(pred, target):
    '''
    像素准确率
    '''
    with torch.no_grad():
        target = target.squeeze()
        B, _, h, w = pred.shape
        pred1 = pred.permute(0, 2, 3, 1).reshape(-1, 2).argmax(axis=1).reshape(B, h, w)
        all_acc = 0
        for x,y in zip(pred1,target):         
            pred_inds_1 = x==1
            target_inds_1 = y ==1
            intersection = pred_inds_1[target_inds_1].sum()
            total = pred_inds_1.sum()
            if total!=0:
                all_acc += (intersection/total).item()
            else:
                all_acc +=0
        if all_acc==0:
            acc=0
        else:
            acc = all_acc/B
    return acc

def IOU(pred, target):
    '''
    iou交并比
    '''
    with torch.no_grad():
        target = target.squeeze()
        pred = pred.argmax(dim=1)
        b = pred.shape[0]
        all_iou = 0
        for x,y in zip(pred,target):         
            pred_inds = x==1
            target_inds = y ==1
            intersection = pred_inds[target_inds].sum()
            union = pred_inds.sum()+target_inds.sum()-intersection
            if union!=0:
                all_iou += (intersection/union).item()
            else:
                all_iou +=0
        if all_iou==0:
            iou=0
        else:
            iou = all_iou/b    
    return iou
