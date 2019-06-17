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
    
    
class CountLoss(nn.Module):
    def __init__(self, scale, step=10):
        super().__init__()
        self.scale = scale
        self.step = step
        
    def forward(self, pred, target, weight=None):
        n, c, h, w = pred.shape
        pred_num = pred.sum(-1).sum(-1) / self.scale
        hm, num = target
        per_pixel_loss = F.mse_loss(pred, hm, reduction='none')
        hm_loss = per_pixel_loss[hm > 0.5].mean() + per_pixel_loss.mean()
        num_loss = F.l1_loss(pred_num, num)
        return hm_loss, num_loss
    