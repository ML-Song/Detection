#coding=utf-8
import cv2
import math
import torch
import numpy as np
from torch.nn import functional as F


class Resize(object):
    def __init__(self, img_size=(416, 416)):
        self.img_size = img_size
        
    def __call__(self, sample):
        img, boxes, labels = sample
        img = cv2.resize(img, self.img_size)
        return img, boxes, labels
    
    
class BoxToHeatmap(object):
    def __init__(self, num_classes=10, stride=8):
        self.num_classes = num_classes
        self.stride = stride
        
    def __call__(self, sample):
        img, boxes, labels = sample
        h, w, c = img.shape
        h /= float(self.stride)
        w /= float(self.stride)
        center = np.asarray([boxes[:, [0, 2]].mean(axis=1), boxes[:, [1, 3]].mean(axis=1)]).T
        center[:, 0] *= w
        center[:, 1] *= h
        center = np.round(center).astype(np.int64)
        
        heatmap = np.zeros((math.ceil(h), math.ceil(w), self.num_classes), dtype=np.int64)
        
        for (x, y), cls in zip(center, labels):
            heatmap[max(y, 51), max(x, 51), cls] = 1
        return img, heatmap
    
    
class ToTensor(object):
    def __call__(self, sample):
        img, heatmap = sample
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).type(torch.float32)
        heatmap = torch.from_numpy(np.transpose(heatmap, (2, 0, 1))).type(torch.float32)
        return img, heatmap
    