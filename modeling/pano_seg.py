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


def generate_box(boxes, size, mask, edge=None, object_map=None, 
                 iou_threshold=0.1, prob_threshold=0.7, topk=200):
    n, c, h, w = boxes.shape
    boxes = torch.cat((boxes - size / 2, boxes + size / 2), dim=1)
    boxes = boxes.permute(0, 2, 3, 1)
    
    if len(mask.shape) == 3:
        prob = torch.ones_like(mask, dtype=torch.float32)
        cls = mask
    else:
        prob, cls = F.softmax(mask, dim=1).max(dim=1)
    if edge is not None:
        edge = F.softmax(edge, dim=1)
        prob *= edge[:, 0]
        
    if object_map is not None:
        prob *= object_map.type(torch.float32)
    frontal = (cls != 0) & (prob > prob_threshold)

    boxes = [boxes[i, frontal[i]] for i in range(n)]
    prob = [prob[i, frontal[i]] for i in range(n)]
    
    rois_index = [tv.ops.nms(boxes[i], prob[i], iou_threshold) for i in range(n)]
    boxes = [boxes[i][rois_index[i]] for i in range(n)]
    prob = [prob[i][rois_index[i]] for i in range(n)]
    
    topk_index = [torch.topk(prob[i], min(topk, prob[i].size(0)), -1)[1] for i in range(n)]

    boxes = [boxes[i][topk_index[i]] for i in range(n)]
    scale = torch.tensor([h, w, h, w], dtype=torch.float32, device=mask.device)
    boxes = [b / scale for b in boxes]
    boxes = [tv.ops.boxes.clip_boxes_to_image(boxes[i], (h, w)) for i in range(n)]
    return boxes


def generate_box_v2(feat, mask, edge=None, prob_threshold=0.7, eps=8, min_samples=5, size=(64, 64)):
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
        
    if edge is not None:
        edge = F.softmax(edge, dim=1)
        prob *= F.interpolate(edge, size=size, mode='bilinear', align_corners=True)[:, 0]
    feat = torch.cat((feat, cls.unsqueeze(dim=1).type(torch.float32)), dim=1)
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
        mask, pos, size, edge = self.backbone(x)
            
        pos_x = pos[:, [0]]
        pos_y = pos[:, [1]]
        size_x = size[:, [0]]
        size_y = size[:, [1]]
        
        size_x = torch.clamp(size_x, min=1, max=h)
        size_y = torch.clamp(size_y, min=1, max=w)
        
        pos_x = torch.clamp(pos_x, min=0, max=h)
        pos_y = torch.clamp(pos_y, min=0, max=w)
        
        pos = torch.cat((pos_x, pos_y), dim=1)
        size = torch.cat((size_x, size_y), dim=1)
        box_map = torch.cat((pos, size), dim=1)
        return mask, box_map, edge
