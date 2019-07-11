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
    def __init__(self, gamma=2, alpha=0.5, prob_threshold=0.7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.prob_threshold = prob_threshold

    def forward(self, pred, target, backward=False):
        pred_box, pred_mask = pred
        box, mask = target
        
        offset_loss = F.smooth_l1_loss(pred_box[:, : 2], box[:, : 2], reduction='none')
        size_loss = F.smooth_l1_loss(pred_box[:, 2:], box[:, 2:], reduction='none')
        box_loss = (offset_loss + size_loss) / 2
        
        weight_prob = (F.softmax(pred_mask, dim=1).max(dim=1, keepdim=True)[0] > 
                       self.prob_threshold).type(torch.float32)
        weight_frontal = (mask.unsqueeze(dim=1) != 0).type(torch.float32)
        weight = weight_prob * weight_frontal
        
        box_loss = box_loss * weight
        box_loss = box_loss.mean()
        
        logpt = -F.cross_entropy(pred_mask, mask, ignore_index=255, reduction='mean')
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        mask_loss = -((1 - pt) ** self.gamma) * logpt
        
        loss = mask_loss + box_loss
        if backward:
            loss.backward(retain_graph=True)
        return loss
    
    
class InstanceSegmentLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, prob_threshold=0.7, threshold=4):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.prob_threshold = prob_threshold
        self.eps = 0.5
        self.threshold = threshold
        
    def forward(self, pred, target, backward=False):
        pred_mask, pred_box = pred
        mask, box = target

        logpt = -F.cross_entropy(pred_mask, mask, ignore_index=255, reduction='mean')
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        mask_loss = -((1 - pt) ** self.gamma) * logpt
        
        n, c, h, w = pred_box.shape
        scale = torch.tensor([h, w], dtype=torch.float32, device=pred_box.device).view(1, 2, 1, 1)
        pred_pos = pred_box[:, : 2]
        pred_size = pred_box[:, 2:]
        pos = box[:, : 2]
        size = box[:, 2:]
        pos_loss = F.smooth_l1_loss(pred_pos, pos, reduction='none')
        pos_loss_per = pos_loss / torch.clamp(size, min=1)
        pos_loss_abs = torch.clamp(pos_loss - self.threshold, min=0) / scale
        pos_loss = (pos_loss_per + pos_loss_abs) * (mask.unsqueeze(dim=1) != 0).type(torch.float32)
        pos_loss = pos_loss.mean()
        
        size_loss = F.smooth_l1_loss(pred_size, size, reduction='none')
        size_loss_per = size_loss / torch.clamp(size, min=1)
        size_loss_abs = torch.clamp(size_loss - self.threshold, min=0) / scale
        size_loss = (size_loss_per + size_loss_abs) * (mask.unsqueeze(dim=1) != 0).type(torch.float32)
        size_loss = size_loss.mean()
        box_loss = pos_loss + size_loss
#         n, c, h, w = pred_feat.shape
#         instance_pos_loss = torch.zeros((1, ), device=pred_feat.device)
#         instance_neg_loss = torch.zeros((1, ), device=pred_feat.device)
#         num = 0
#         pred_feat = pred_feat.permute(0, 2, 3, 1)
#         pred_prob, pred_cls = F.softmax(pred_mask, dim=1).max(dim=1)
#         frontal = (mask != 0) & (pred_prob > self.prob_threshold) & (pred_cls != 0)
#         for i in range(n):
#             for l in torch.unique(instance_map[i]):
#                 index = (instance_map[i] == l)# & (frontal[i])
#                 if index.sum() == 0 or l == 0 or (~index).sum() == 0:
#                     continue
#                 instance_feat = pred_feat[i, index]
#                 other_feat = pred_feat[i, ~index]
#                 center = instance_feat.mean(0, keepdim=True)
#                 pos_dis = F.pairwise_distance(instance_feat, center)
#                 pos_loss = torch.clamp(pos_dis - 0.5 * self.eps, min=0.)
#                 pos_loss = pos_loss.mean() + pos_loss.max()
#                 neg_dis = F.pairwise_distance(other_feat, center)
#                 neg_loss = torch.clamp(1.5 * self.eps - neg_dis, min=0.)
#                 neg_loss = neg_loss.mean() + neg_loss.max()
#                 instance_pos_loss = instance_pos_loss + pos_loss
#                 instance_neg_loss = instance_neg_loss + neg_loss
# #                 pos_loss = torch.clamp(0.7 - F.cosine_similarity(instance_feat, center).min(), min=0)
# #                 neg_loss = torch.clamp(F.cosine_similarity(other_feat, center).max() - 0.3, min=0)
# #                 instance_loss = instance_loss + pos_loss + neg_loss
#                 num += 1
#         if num > 0:
#             instance_pos_loss = instance_pos_loss / num
#             instance_neg_loss = instance_neg_loss / num
#             loss = instance_pos_loss + instance_neg_loss + mask_loss
#         else:
#             loss = mask_loss
        loss = mask_loss + box_loss
        if backward:
            loss.backward(retain_graph=True)
        return mask_loss, box_loss