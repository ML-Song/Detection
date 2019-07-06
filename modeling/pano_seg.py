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


def bbox_iou(bbox_a, bbox_b=None):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_b is None:
        bbox_b = bbox_a.copy()
    if len(bbox_a.shape) == 1:
        bbox_a = np.expand_dims(bbox_a, axis=0)
    if len(bbox_b.shape) == 1:
        bbox_b = np.expand_dims(bbox_b, axis=0)
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return 1 - area_i / (area_a[:, None] + area_b - area_i)


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


def generate_box_v2(feat, mask, prob_threshold=0.7, eps=0.5, min_samples=5, size=(64, 64)):
    n, c, h, w = feat.shape
    
    pos = np.dstack(np.mgrid[0: h, 0: w])
    pos = torch.from_numpy(pos).unsqueeze(dim=0).type(torch.float32)
    pos = pos.to(feat.device)
    
    pos = pos.permute(0, 3, 1, 2)
    pos = F.interpolate(pos.type(torch.float32), size=size, mode='bilinear', align_corners=True)
    pos = pos.permute(0, 2, 3, 1)
    pos = pos.type(torch.int64)
    
    if len(mask.shape) == 3:
        mask = F.interpolate(mask.unsqueeze(1).type(torch.float32), 
                             size=size).squeeze(1).type(torch.int64)
        prob = torch.ones_like(mask, dtype=torch.float32)
        cls = mask
        feat = F.interpolate(feat, size=size)
    elif len(mask.shape) == 4:
        mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=True)
        prob, cls = F.softmax(mask, dim=1).max(dim=1)
        feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=True)
    else:
        raise Exception('Mask shape: {} not supported!'.format(mask.shape))
        
    feat = feat.permute(0, 2, 3, 1)
    frontal = (cls != 0) & (prob > prob_threshold)

    pos = pos.repeat(n, 1, 1, 1)
    pts = [feat[i, frontal[i]] for i in range(n)]
    pos = [pos[i, frontal[i]] for i in range(n)]
    prob = [prob[i, frontal[i]] for i in range(n)]
    
    instances = []
    for pt, p in zip(pts, pos):
        if pt.shape[0] == 0:
            instances.append(np.zeros((0, 4)))
            continue
        p = p.numpy()
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pt.numpy())
        instance = []
        for l in np.unique(db.labels_):
            if l != -1:
                ins = p[db.labels_ == l]
                instance.append(np.concatenate((ins.min(0), ins.max(0))))
        if len(instance) == 0:
            instances.append(np.zeros((0, 4)))
            continue
        instances.append(np.array(instance) / np.array([h, w, h, w]))
    
    return instances


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
