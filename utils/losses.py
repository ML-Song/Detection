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
    
    
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

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
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
    
        
class CountLoss(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.seg_criterion = SegmentationLosses()
        
    def forward(self, pred, target, rate=None, backward=False):
        pred_hm, pred_mask = pred
        hm, mask, num = target
        
        hm_loss = F.mse_loss(pred_hm, hm, reduction='none')
        hm_loss = hm_loss.view(hm_loss.size(0), hm_loss.size(1), -1)
        hm_loss = torch.topk(hm_loss, int(hm_loss.size(2) * rate), dim=-1)[0]
        hm_loss = hm_loss.mean()
        
        mask_loss = self.seg_criterion.FocalLoss(pred_mask, mask)
        
#         mask_loss = F.binary_cross_entropy(pred_mask, mask, reduction='none')
#         mask_loss = mask_loss.view(mask_loss.size(0), mask_loss.size(1), -1)
#         mask_loss = torch.topk(mask_loss, int(mask_loss.size(2) * rate), dim=-1)[0]
#         mask_loss = mask_loss.mean()
        
        pred_num = pred_hm.sum(-1).sum(-1) / self.scale
        num_loss = F.mse_loss(pred_num, num, reduction='none')
        num_loss = torch.topk(num_loss, int(num_loss.size(1) * rate), dim=-1)[0]
        num_loss = num_loss.mean()
        loss = mask_loss
#         loss = hm_loss + mask_loss + num_loss
        if backward:
            loss.backward(retain_graph=True)
        return loss
    