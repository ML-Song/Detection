#coding=utf-8
import cv2
import torch
import numpy as np
import torchvision.utils as vutils
from torch.nn import functional as F

def show_detections(img, boxes, labels):
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
    line_width = (h + w) // (2 * 100)
    for (x_min, y_min, x_max, y_max), l in zip(boxes_show, labels):
        cv2.rectangle(img_show, 
                      (x_min, y_min), 
                      (x_max, y_max), 
                      (int(l * 30 % 255), int((255 - l * 40) % 255), 0), line_width)
        
    return img_show


def show_heatmap(hm):
    hm = np.clip(hm, 0, 1)
    hm_hsv = np.transpose(np.array([(180 - (hm * 180)).astype(np.uint8), np.ones_like(hm), np.ones_like(hm)]), (1, 2, 0))
    return cv2.cvtColor(hm_hsv, cv2.COLOR_HSV2RGB)


def heatmap_to_rgb(hm, num_classes, size=(64, 64), threshold=0.5):
    hm = F.interpolate(hm, size).cpu()
    prob_scaled, cls_scaled = hm.max(dim=1, keepdim=True)
    cls_scaled = cls_scaled.type(torch.float32)
    cls_scaled = torch.round(cls_scaled) / num_classes
    
    prob_scaled = vutils.make_grid(prob_scaled)[[0]].numpy()
    cls_scaled = vutils.make_grid(cls_scaled)[[0]].numpy()
    prob_scaled = np.clip(prob_scaled, 0, 1)
    cls_scaled = np.clip(cls_scaled, 0, 1)
    v = np.zeros_like(cls_scaled)
    v[prob_scaled > threshold] = 1
    
    hsv = np.concatenate((180 * cls_scaled, prob_scaled, v), axis=0)
    hsv = np.transpose(hsv, (1, 2, 0))
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb = np.transpose(rgb, (2, 0, 1))
    return rgb


def mask_to_rgb(mask, num_classes, size=(64, 64)):
    mask = F.interpolate(mask, size).cpu()
    prob_scaled, cls_scaled = mask.max(dim=1, keepdim=True)
    cls_scaled = cls_scaled.type(torch.float32)
    cls_scaled = torch.round(cls_scaled) / num_classes
    
    prob_scaled = vutils.make_grid(prob_scaled)[[0]].numpy()
    cls_scaled = vutils.make_grid(cls_scaled)[[0]].numpy()
    prob_scaled = np.clip(prob_scaled, 0, 1)
    cls_scaled = np.clip(cls_scaled, 0, 1)
    v = np.zeros_like(cls_scaled)
    v[cls_scaled != 0] = 1
    
    hsv = np.concatenate((180 * cls_scaled, prob_scaled, v), axis=0)
    hsv = np.transpose(hsv, (1, 2, 0))
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb = np.transpose(rgb, (2, 0, 1))
    return rgb