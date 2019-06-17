#coding=utf-8
import cv2
import math
import torch
import numpy as np
from torch.nn import functional as F
from scipy.stats import multivariate_normal


class Resize(object):
    def __init__(self, img_size=(416, 416)):
        self.img_size = img_size
        
    def __call__(self, sample):
        img, boxes, labels = sample
        img = cv2.resize(img, self.img_size)
        return img, boxes, labels
    
    
# class BoxToHeatmap(object):
#     def __init__(self, num_classes=20, stride=8, radius=1):
#         self.num_classes = num_classes
#         self.stride = stride
#         self.radius = radius
        
#     def __call__(self, sample):
#         img, boxes, labels = sample
#         h, w, c = img.shape
#         h /= float(self.stride)
#         w /= float(self.stride)
#         center = np.asarray([boxes[:, [0, 2]].mean(axis=1), boxes[:, [1, 3]].mean(axis=1)]).T
#         center[:, 0] *= w
#         center[:, 1] *= h
#         center = np.round(center).astype(np.int64)
        
#         heatmap = np.zeros((math.ceil(h), math.ceil(w), 1), dtype=np.int64)
# #         heatmap = np.zeros((math.ceil(h), math.ceil(w), self.num_classes), dtype=np.int64)
        
#         for (x, y), cls in zip(center, labels):
#             for i in range(-self.radius, self.radius + 1):
#                 for j in range(-self.radius, self.radius + 1):
#                     heatmap[max(0, min(y + i, 51)), max(0, min(x + j, 51)), 0] = 1
# #             heatmap[max(y, 51), max(x, 51), cls] = 1
#         return img, heatmap


class BoxToHeatmap(object):
    def __init__(self, num_classes, stride=8, cov=10, threshold=0.1):
        self.num_classes = num_classes
        self.stride = stride
        self.cov = cov
        self.threshold = threshold
#         self.scale = math.pi * 2 * self.cov
        
    def __call__(self, sample):
        img, boxes, labels = sample
#         assert(labels.dtype == np.int64)
        h, w, c = img.shape
        h /= float(self.stride)
        w /= float(self.stride)
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        center = np.asarray([boxes[:, [1, 3]].mean(axis=1), boxes[:, [0, 2]].mean(axis=1)]).T
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        heatmap = np.zeros((self.num_classes, math.ceil(h), math.ceil(w)), dtype=np.float32)
        num = np.zeros((self.num_classes, ), dtype=np.float32)
        pos = np.dstack(np.mgrid[0: math.ceil(h), 0: math.ceil(w)])
        for c, bw, bh, l in zip(center, box_w, box_h, labels):
#             rv = multivariate_normal(mean=c, cov=[[bh * self.cov, 0], [0, bw * self.cov]])
            rv = multivariate_normal(mean=c, cov=[[30 * self.cov, 0], [0, 30 * self.cov]])
            tmp_hm = rv.pdf(pos)
            tmp_hm /= tmp_hm.max()
#             if self.threshold is not None:
#                 tmp_hm[tmp_hm < (self.threshold / self.scale)] = 0
#                 tmp_hm /= tmp_hm.sum()
            if isinstance(l, np.int64) or isinstance(l, int):
                heatmap[l, :, :] += tmp_hm
                num[l] += 1
            else:
                heatmap[0, :, :] += tmp_hm
                num[0] += 1
        return img, heatmap, num
    
class ToTensor(object):
    def __call__(self, sample):
        img, heatmap, num = sample
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).type(torch.float32) / 255
        heatmap = torch.from_numpy(heatmap).type(torch.float32)
        return img, heatmap, num
    