import math
import time
import torch
import numpy as np
from torch import nn
import torchvision as tv
from torch.nn import functional as F


def gaussian2d(mu, sigma, prob, pos, cov):
    n = mu.size(0)
    _, h, w, _ = pos.shape
    if n == 0:
        return torch.zeros((1, h, w), device=mu.device)
    x = pos - mu.view(n, 1, 1, 2)
    x = x / (sigma.view(n, 1, 1, 2) / cov)
    x = (x ** 2).sum(dim=-1)
    x = torch.exp(-x / 2)# * prob.view(-1, 1, 1)# / (2 * math.pi * sigma.prod(dim=-1).view(-1, 1, 1))
    x = x.max(dim=0, keepdim=True)[0]
    return x


class PanopticSegment(nn.Module):
    def __init__(self, backbone, iou_threshold=0.5, prob_threshold=0.5, cov=10, topk=500):
        super().__init__()
        self.backbone = backbone
        
        self.iou_threshold = iou_threshold
        self.prob_threshold = prob_threshold
        self.cov = cov
        self.topk = topk
        self.pos = None
        
    def forward(self, x):
        n, c, h, w = x.shape
#         torch.autograd.set_detect_anomaly(True)
        mask, reg = self.backbone(x)
        
#         start = time.time()
        if self.pos is None:
            self.pos = np.dstack(np.mgrid[0: h, 0: w])
            self.pos = torch.from_numpy(self.pos).unsqueeze(dim=0).type(torch.float32)
            self.pos = self.pos.to(mask.device)
        
#         print('stage 1: ', time.time() - start)
#         start = time.time()

        bias = F.tanh(reg[:, [0, 1]])
        size = reg[:, [2, 3]]
#         bias[:, 0] = bias[:, 0] * h
#         bias[:, 1] = bias[:, 1] * w
        size = torch.clamp(size, min=4)
        
        boxes = self.pos.permute(0, 3, 1, 2)
        boxes = boxes + bias
        boxes = torch.cat((boxes - size, boxes + size), dim=1)
        boxes = boxes.permute(0, 2, 3, 1)
        boxes = torch.clamp(boxes, min=0, max=max(h, w))
        
        prob, cls = F.softmax(mask, dim=1).max(dim=1)
        prob = prob.detach()
        frontal = (cls != 0) & (prob > self.prob_threshold)
        
#         print('stage 2: ', time.time() - start)
#         start = time.time()
        
        boxes = [boxes[i, frontal[i]] for i in range(n)]
        prob = [prob[i, frontal[i]] for i in range(n)]
        
        rois_index = [tv.ops.nms(boxes[i], prob[i], self.iou_threshold) for i in range(n)]
        boxes = [boxes[i][rois_index[i]] for i in range(n)]
        prob = [prob[i][rois_index[i]] for i in range(n)]
        topk_index = [torch.topk(prob[i], min(self.topk, prob[i].size(0)), -1)[1] for i in range(n)]
        
        boxes = [boxes[i][topk_index[i]] for i in range(n)]
        prob = [prob[i][topk_index[i]] for i in range(n)]
        
        mu = [(b[:, : 2] + b[:, 2: ]) / 2 for b in boxes]
        sigma = [(b[:, 2: ] - b[:, : 2]) for b in boxes]
        
#         print('stage 3: ', time.time() - start)
#         start = time.time()
        
        heatmaps = [gaussian2d(mu[i], sigma[i], prob[i], self.pos, self.cov) for i in range(n)]
        
#         print('stage 4: ', time.time() - start)
        heatmaps = torch.cat(heatmaps, dim=0).unsqueeze(dim=1)
#         print('hm: ', heatmaps)
#         print(heatmaps)
        return heatmaps, mask, boxes

# class Gaussian2D(nn.Module):
#     '''
#     convert mu and sigma to 2d gaussian dis
#     '''
#     def __init__(self, w, h):
#         super().__init__()
#         self.w = w
#         self.h = h
#         pos = np.dstack(np.mgrid[0: h, 0: w])
#         self.pos = torch.from_numpy(pos).unsqueeze(dim=0).type(torch.float32)
        
#     def forward(self, mu, sigma, prob):
#         '''
#         mu: tensor [n, 2]
#         sigma: tensor [n, 2]
#         prob: tensor [n]
#         '''
#         n = mu.size(0)
#         if n == 0:
#             return torch.zeros((1, self.h, self.w), device=mu.device)
#         x = self.pos - mu.view(n, 1, 1, 2)
#         x = x / sigma.view(n, 1, 1, 2)
#         x = (x ** 2).sum(dim=-1)
#         x = torch.exp(-x / 2) / (2 * math.pi * sigma.prod(dim=-1).view(-1, 1, 1)) * prob.view(-1, 1, 1)
#         x = x.max(dim=0, keepdim=True)[0]
#         return x