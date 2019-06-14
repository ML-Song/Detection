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
        
    def forward(self, pred, target):
        n, c, h, w = pred.shape
        pred_num = pred.sum(-1).sum(-1) / self.scale
        hm, num = target
        hm_loss = F.mse_loss(pred, hm, reduction='none')
        hm_loss = sum([hm_loss[(hm >= i / self.step) & (hm <= (i + 1) / self.step)].mean() 
                       for i in range(self.step)]) / self.step
        num_loss = F.l1_loss(pred_num, num)
        return hm_loss, num_loss
# class CenterLoss(nn.Module):
#     def __init__(self, ratio=2, threshold=0.5):
#         super().__init__()
#         self.ratio = ratio
#         self.threshold = threshold
        
#     def forward(self, pred, target):
#         n, c, h, w = pred.shape
#         per_pixel_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
#         prob = torch.sigmoid(pred)
        
#         tp = (prob >= self.threshold) & (target >= self.threshold)
#         fp = (prob >= self.threshold) & (target < self.threshold)
#         tn = (prob < self.threshold) & (target < self.threshold)
#         fn = (prob < self.threshold) & (target >= self.threshold)
        
#         per_pixel_loss_tp = per_pixel_loss.clone()
#         per_pixel_loss_tp[~tp] = 0
#         per_pixel_loss_tp = per_pixel_loss_tp.view(n, -1)
#         loss_tp = torch.topk(per_pixel_loss_tp, k=10, dim=1)[0]
#         loss_tp = loss_tp.mean()
        
#         per_pixel_loss_fp = per_pixel_loss.clone()
#         per_pixel_loss_fp[~fp] = 0
#         per_pixel_loss_fp = per_pixel_loss_fp.view(n, -1)
#         loss_fp = torch.topk(per_pixel_loss_fp, k=10, dim=1)[0]
#         loss_fp = loss_fp.mean()
        
#         per_pixel_loss_tn = per_pixel_loss.clone()
#         per_pixel_loss_tn[~tn] = 0
#         per_pixel_loss_tn = per_pixel_loss_tn.view(n, -1)
#         loss_tn = torch.topk(per_pixel_loss_tn, k=10, dim=1)[0]
#         loss_tn = loss_tn.mean()
        
#         per_pixel_loss_fn = per_pixel_loss.clone()
#         per_pixel_loss_fn[~fn] = 0
#         per_pixel_loss_fn = per_pixel_loss_fn.view(n, -1)
#         loss_fn = torch.topk(per_pixel_loss_fn, k=10, dim=1)[0]
#         loss_fn = loss_fn.mean()
        
        
        
# #         loss_tp = per_pixel_loss[tp].mean() if tp.sum() > 0 else torch.zeros((1, ), device=per_pixel_loss.device)
# #         loss_fp = per_pixel_loss[fp].mean() if fp.sum() > 0 else torch.zeros((1, ), device=per_pixel_loss.device)
# #         loss_tn = per_pixel_loss[tn].mean() if tn.sum() > 0 else torch.zeros((1, ), device=per_pixel_loss.device)
# #         loss_fn = per_pixel_loss[fn].mean() if fn.sum() > 0 else torch.zeros((1, ), device=per_pixel_loss.device)
#         loss = loss_tp + loss_fp + loss_tn + loss_fn
# #         loss = loss_fp + loss_fn
# #         target_num = target.max(dim=1)[0].view(target.size(0), -1).sum(dim=1).type(torch.int64)
# #         weights = torch.zeros((n, 1, h, w), dtype=torch.uint8, device=pred.device)
# #         for i, num in enumerate(target_num):
# #             num = num * self.ratio
# #             x = torch.randint(w, (num, ))
# #             y = torch.randint(h, (num, ))
# #             weights[i, 0, y, x] = 1
            
# #         weights[target.max(dim=1, keepdim=True)[0] > 0.5] = 1
# #         loss = per_pixel_loss[weights.repeat(1, c, 1, 1)].mean()
#         return loss
    