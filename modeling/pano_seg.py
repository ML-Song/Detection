import math
import time
import torch
import numpy as np
from torch import nn
import torchvision as tv
from sklearn.cluster import DBSCAN
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


def generate_box(offset, size, mask, pos=None, iou_threshold=0.1, prob_threshold=0.7, topk=100):
    n, c, h, w = offset.shape
    if pos is None:
        pos = np.dstack(np.mgrid[0: h, 0: w])
        pos = torch.from_numpy(pos).unsqueeze(dim=0).type(torch.float32)
        pos = pos.to(offset.device)
    
    boxes = pos.permute(0, 3, 1, 2)
    boxes = boxes + offset
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
    scale = torch.tensor([h, w, h, w], dtype=torch.float32, device=offset.device)
    boxes = [b / scale for b in boxes]
    boxes = [tv.ops.boxes.clip_boxes_to_image(boxes[i], (h, w)) for i in range(n)]
    return boxes
    
    
def generate_box_v2(offset, mask, pos=None, prob_threshold=0.7, eps=4, min_samples=5, size=(64, 64)):
    n, c, h, w = offset.shape
    if pos is None:
        pos = np.dstack(np.mgrid[0: h, 0: w])
        pos = torch.from_numpy(pos).unsqueeze(dim=0).type(torch.float32)
        pos = pos.to(offset.device)
    
    centers = pos.permute(0, 3, 1, 2)
    centers = centers + offset
    pos = pos.permute(0, 3, 1, 2)
    pos = F.interpolate(pos.type(torch.float32), size=size, mode='bilinear', align_corners=True)
    pos = pos.permute(0, 2, 3, 1)
    
    centers = F.interpolate(centers, size=size, mode='bilinear', align_corners=True)
    centers = centers.permute(0, 2, 3, 1)
    # / torch.tensor((h, w), dtype=torch.float32)
#     centers = torch.cat((centers, pos), dim=-1)
    pos = pos.type(torch.int64)
    
    if len(mask.shape) == 3:
        mask = F.interpolate(mask.unsqueeze(1).type(torch.float32), 
                             size=size, mode='bilinear', align_corners=True).squeeze(1).type(torch.int64)
        prob = torch.ones_like(mask, dtype=torch.float32)
        cls = mask
    elif len(mask.shape) == 4:
        mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=True)
        prob, cls = F.softmax(mask, dim=1).max(dim=1)
#         prob += 0.8
    else:
        raise Exception('Mask shape: {} not supported!'.format(mask.shape))
    frontal = (cls != 0) & (prob > prob_threshold)

    pos = pos.repeat(n, 1, 1, 1)
    centers = [centers[i, frontal[i]] for i in range(n)]
    pos = [pos[i, frontal[i]] for i in range(n)]
#     instances = []
    boxes = []
    for center, p in zip(centers, pos):
        if center.shape[0] == 0:
            boxes.append(np.zeros((0, 4)))
            continue
        p = p.numpy()
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(center.numpy())
        box = []
        for l in np.unique(db.labels_):
            if l != -1:
                ins = p[db.labels_ == l]
                box.append(np.concatenate((ins.min(0), ins.max(0))))
        if len(box) == 0:
            boxes.append(np.zeros((0, 4)))
            continue
        boxes.append(np.array(box) / np.array([h, w, h, w]))
#         instances.append(db.labels_)
    
    return boxes


class PanopticSegment(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
    def forward(self, x):
        n, c, h, w = x.shape
        mask, reg = self.backbone(x)
            
        offset_x = reg[:, [0]]
        offset_y = reg[:, [1]]
        size_x = reg[:, [2]]
        size_y = reg[:, [3]]
        
        size_x = torch.clamp(size_x, min=1, max=h)
        size_y = torch.clamp(size_y, min=1, max=w)
        
        offset_x = torch.clamp(offset_x, min=-h, max=h)
        offset_y = torch.clamp(offset_y, min=-w, max=w)
        
        offset = torch.cat((offset_x, offset_y), dim=1)
        size = torch.cat((size_x, size_y), dim=1)
        box_map = torch.cat((offset, size), dim=1)
        return box_map, mask
