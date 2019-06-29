#coding=utf-8
import cv2
import torch
import numpy as np
import torchvision.utils as vutils
from torch.nn import functional as F

def draw_box(img, boxes, labels):
    '''
    boxes: [[x_min, y_min, x_max, y_max], ...]
    labels: [0, 1, 2, ...]
    '''
    h, w, c = img.shape
    img_show = img.copy()
    boxes_show = boxes.copy()
    boxes_show[:, [0, 2]] *= w
    boxes_show[:, [1, 3]] *= h
    boxes_show = boxes_show.astype(np.int64)
    line_width = max((h + w) // (200), 1)
    for (x_min, y_min, x_max, y_max), l in zip(boxes_show, labels):
        cv2.rectangle(img_show, 
                      (x_min, y_min), 
                      (x_max, y_max), 
                      (int(l * 30 % 255), int((255 - l * 40) % 255), 0), line_width)
        
    return img_show


def draw_boxes(images, boxes, labels=None):
    '''
    images: Tensor [N, 3, H, W]
    boxes: List [N]
    labels: List [N]
    
    return: Tensor [N, 3, H, W]
    '''
    n, c, h, w = images.shape
    images_show = images.permute(0, 2, 3, 1).numpy().copy()
    if labels is None:
        labels = [[0] * len(b) for b in boxes]
    result = []
    thick = max((h + w) // (200), 1)
    for img, box, label in zip(images_show, boxes, labels):
        for b, l in zip(box, label):
            img = cv2.rectangle(img, (int(b[1] * h), int(b[0] * w)), (int(b[3] * h), int(b[2] * w)), (2, 0, 0), thick)
        result.append(img)
    result = torch.from_numpy(np.array(result)).permute(0, 3, 1, 2)
    return result


def mask_to_rgb(mask, num_classes, is_gt=False):
    '''
    mask: Tensor [N, 3, H, W] / [N, H, W]
    num_classes: int
    
    return: Tensor [3, H', W']
    '''
    if not is_gt:
        prob_scaled, cls_scaled = mask.max(dim=1, keepdim=True)
        cls_scaled = cls_scaled.type(torch.float32)
    else:
        cls_scaled = mask
        prob_scaled = torch.ones_like(cls_scaled)
        
    cls_scaled = torch.round(cls_scaled) / num_classes
    
    prob_scaled = vutils.make_grid(prob_scaled)[[0]].numpy()
    cls_scaled = vutils.make_grid(cls_scaled)[[0]].numpy()
    prob_scaled = np.clip(prob_scaled, 0, 1)
    cls_scaled = np.clip(cls_scaled, 0, 1)
    v = np.zeros_like(cls_scaled)
    v[cls_scaled > 1e-6] = 1
    
    hsv = np.concatenate((180 * cls_scaled, prob_scaled, v), axis=0)
    hsv = np.transpose(hsv, (1, 2, 0))
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb = np.transpose(rgb, (2, 0, 1))
    rgb = torch.from_numpy(rgb)
    return rgb