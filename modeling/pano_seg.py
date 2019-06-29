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


def generate_box(bias, size, mask, pos=None, iou_threshold=0.5, prob_threshold=0.5, topk=500):
    n, c, h, w = bias.shape
    if pos is None:
        pos = np.dstack(np.mgrid[0: h, 0: w])
        pos = torch.from_numpy(pos).unsqueeze(dim=0).type(torch.float32)
        pos = pos.to(bias.device)
    
    boxes = pos.permute(0, 3, 1, 2)
    boxes = boxes + bias
    boxes = torch.cat((boxes - size / 2, boxes + size / 2), dim=1)
    boxes = boxes.permute(0, 2, 3, 1)
    
    if len(mask.shape) == 3:
        prob = torch.ones_like(mask, dtype=torch.float32)
        cls = mask
    else:
        prob, cls = F.softmax(mask, dim=1).max(dim=1)
    frontal = (cls != 0) & (prob > prob_threshold)

    boxes = [boxes[i, frontal[i]] for i in range(n)]
    prob = [prob[i, frontal[i]] for i in range(n)]
    
    rois_index = [tv.ops.nms(boxes[i], prob[i], iou_threshold) for i in range(n)]
    boxes = [boxes[i][rois_index[i]] for i in range(n)]
    prob = [prob[i][rois_index[i]] for i in range(n)]
    
    topk_index = [torch.topk(prob[i], min(topk, prob[i].size(0)), -1)[1] for i in range(n)]

    boxes = [boxes[i][topk_index[i]] for i in range(n)]
    scale = torch.tensor([h, w, h, w], dtype=torch.float32, device=bias.device)
    boxes = [b / scale for b in boxes]
#     boxes = [tv.ops.boxes.clip_boxes_to_image(boxes[i], (h, w)) for i in range(n)]
    return boxes
    
    
class PanopticSegment(nn.Module):
    def __init__(self, backbone, iou_threshold=0.5, prob_threshold=0.5, topk=500):
        super().__init__()
        self.backbone = backbone
        
        self.iou_threshold = iou_threshold
        self.prob_threshold = prob_threshold
        self.topk = topk
        self.pos = None
        
    def forward(self, x):
        n, c, h, w = x.shape
        mask, reg = self.backbone(x)
        if self.pos is None:
            self.pos = np.dstack(np.mgrid[0: h, 0: w])
            self.pos = torch.from_numpy(self.pos).unsqueeze(dim=0).type(torch.float32)
            self.pos = self.pos.to(mask.device)
            
        bias_x = reg[:, [0]]
        bias_y = reg[:, [1]]
        size = reg[:, [2, 3]]
        bias_x = torch.clamp(bias_x, min=-h, max=h)
        bias_y = torch.clamp(bias_y, min=-w, max=w)
        bias = torch.cat((bias_x, bias_y), dim=1)
        size = torch.clamp(size, min=8)
        
        box_map = torch.cat((bias, size), dim=1)
        return box_map, mask
