#coding=utf-8
import cv2
import math
import torch
import numpy as np
from skimage.transform import resize
from torch.nn import functional as F
from scipy.stats import multivariate_normal

# from utils.k_nearest_gaussian_kernel import gaussian_filter_density


class Resize(object):
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
        
    def __call__(self, sample):
        sample['image'] = resize(sample['image'], self.img_size)
        return sample
    
    
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


class GenerateHeatmap(object):
    def __init__(self, num_classes, output_size=(256, 256), cov=10):
        self.num_classes = num_classes
        self.output_size = output_size
        self.cov = cov

    def _gen_hm(self, points, h, w):
        points = points.copy()
        points[:, 0] *= w
        points[:, 1] *= h
        center = points.mean(axis=0)
        obj_w, obj_h = points.max(axis=0) - points.min(axis=0)
#         rv = multivariate_normal(mean=[center[1], center[0]], cov=[[obj_h * self.cov, 0], [0, obj_w * self.cov]])
        rv = multivariate_normal(mean=[center[1], center[0]], cov=[[self.cov, 0], [0, self.cov]])
        return rv
        
    def __call__(self, sample):
        boxes = sample['boxes']
        polygons = sample['polygons']
        box_labels = sample['box_labels']
        polygon_labels = sample['polygon_labels']
        
        h, w = self.output_size
        pos = np.dstack(np.mgrid[0: math.ceil(h), 0: math.ceil(w)])
        
        if self.num_classes is None:
            heatmap = np.zeros((1, math.ceil(h), math.ceil(w)), dtype=np.float32)
            num = np.zeros((1, ), dtype=np.float32)
        else:
            heatmap = np.zeros((self.num_classes, math.ceil(h), math.ceil(w)), dtype=np.float32)
            num = np.zeros((self.num_classes, ), dtype=np.float32)

        for pts, l in zip(boxes, box_labels):
            rv = self._gen_hm(pts, h, w)
            tmp_hm = rv.pdf(pos)
            tmp_hm /= tmp_hm.max()
            heatmap[l, :, :] += tmp_hm
            num[l] += 1
                
        for pts, l in zip(polygons, polygon_labels):
            rv = self._gen_hm(pts, h, w)
            tmp_hm = rv.pdf(pos)
            tmp_hm /= tmp_hm.max()
            heatmap[l, :, :] += tmp_hm
            num[l] += 1
                
        sample['heatmap'] = heatmap
        sample['num'] = num
        return sample
    
    
class GenerateMask(object):
    def __init__(self, num_classes, output_size=(256, 256)):
        self.num_classes = num_classes
        self.output_size = output_size
        self.order = None
        self.ignore = None

    def _gen_mask_from_polygon(self, mask, points, label, h, w):
        points = points.copy()
        points[:, 0] *= w
        points[:, 1] *= h
        points = np.expand_dims(np.round(points).astype(np.int64), axis=0)
        cv2.fillPoly(mask[label + 1], points, self.order[label])
        
    def _gen_mask_from_box(self, mask, points, label, h, w):
        points = points.copy()
        points[:, 0] *= w
        points[:, 1] *= h
        points = np.round(points).astype(np.int64)
        mask[label + 1, points[0, 1]: points[1, 1], points[0, 0]: points[1, 0]] = self.order[label]
        
    def __call__(self, sample):
        if self.order is None:
            self.order = {v[0]: v[1] for v in sample['class_map'].values()}
            
        if self.ignore is None:
            self.ignore = [v[0] for v in sample['class_map'].values() if v[2]]
            
        boxes = sample['boxes']
        polygons = sample['polygons']
        box_labels = sample['box_labels']
        polygon_labels = sample['polygon_labels']
        
        h, w = self.output_size
        mask = np.zeros((self.num_classes + 1, math.ceil(h), math.ceil(w)), dtype=np.uint8)
        
        for pts, l in zip(boxes, box_labels):
            self._gen_mask_from_box(mask, pts, l, h, w)
            
        for pts, l in zip(polygons, polygon_labels):
            self._gen_mask_from_polygon(mask, pts, l, h, w)
                
        mask = mask.argmax(axis=0)
        for ind in self.ignore:
            mask[mask == ind] = 255
        sample['mask'] = mask
        return sample
    
    
class ToTensor(object):
    def __call__(self, sample):
        sample.pop('polygons')
        sample.pop('boxes')
        sample.pop('box_labels')
        sample.pop('polygon_labels')
        sample.pop('class_map')
        sample['image'] = torch.from_numpy(np.transpose(sample['image'], (2, 0, 1))).type(torch.float32)
        sample['heatmap'] = torch.from_numpy(sample['heatmap']).type(torch.float32)
        sample['mask'] = torch.from_numpy(sample['mask']).type(torch.int64)
        sample['num'] = torch.from_numpy(sample['num']).type(torch.float32)
#         sample['boxes'] = torch.from_numpy(sample['boxes']).type(torch.float32)
        return sample
    
    
class SampleObject(object):
    def __call__(self, sample):
        boxes = sample['boxes']
        polygons = sample['polygons']
        box_labels = sample['box_labels']
        polygon_labels = sample['polygon_labels']
        
        labels = np.concatenate((box_labels, polygon_labels))
        anchor_label = np.random.choice(labels)
        
        boxes = boxes[box_labels == anchor_label]
        polygons = polygons[polygon_labels == anchor_label]
        
        sample['anchor_label'] = anchor_label
        sample['boxes'] = boxes
        sample['polygons'] = polygons
#         sample['box_labels'] = np.zeros((boxes.shape[0], ), dtype=np.int64)
#         sample['polygon_labels'] = np.zeros((polygons.shape[0], ), dtype=np.int64)
        return sample