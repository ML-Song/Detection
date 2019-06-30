#coding=utf-8
import torch
from torch import nn
from torch.nn import functional as F

class CenterLoss(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()
        self.ratio = ratio
        
    def forward(self, pred, target):
        n, c, h, w = pred.shape
        per_pixel_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        target_num = target.max(dim=1)[0].view(target.size(0), -1).sum(dim=1).type(torch.int64)
        weights = torch.zeros((n, 1, h, w), dtype=torch.uint8, device=pred.device)
        for i, num in enumerate(target_num):
            num = num * self.ratio
            x = torch.randint(w, (num, ))
            y = torch.randint(h, (num, ))
            weights[i, 0, y, x] = 1
            
        weights[target.max(dim=1, keepdim=True)[0] > 0.5] = 1
        loss = per_pixel_loss[weights.repeat(1, c, 1, 1)].mean()
        return loss
    
    
class SegmentationLosses(nn.Module):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
#         criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
#                                         size_average=self.size_average)
#         if self.cuda:
#             criterion = criterion.cuda()

        loss = self.criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
#         criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
#                                         size_average=self.size_average)
#         if self.cuda:
#             criterion = criterion.cuda()

        logpt = -self.criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
    
        
class CountLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target, rate=None, backward=False):
        pred_box, pred_mask = pred
        box, mask = target
        
#         print(box[:, 3], pred_box[:, 3])
        box_loss = F.mse_loss(pred_box, box, reduction='none')
        box_loss = box_loss * (mask.unsqueeze(dim=1) != 0).type(torch.float32)
#         box_loss = box_loss * F.softmax(pred_mask, dim=1).max(dim=1, keepdim=True)[0].detach()
        box_loss = box_loss.mean()
#         front = mask != 0
#         if front.sum() > 0:
#             box_loss = box_loss[mask != 0].mean()
#         else:
#             box_loss = torch.zeros((1, ), device=pred_box.device)
        
        logpt = -F.cross_entropy(pred_mask, mask, ignore_index=255, reduction='mean')
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        mask_loss = -((1 - pt) ** self.gamma) * logpt
        
        loss = 3 * mask_loss + box_loss
        if backward:
            loss.backward(retain_graph=True)
        return loss
    