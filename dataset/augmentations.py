# coding=utf-8
import cv2
import math
import torch
import numpy as np
from skimage.transform import resize
from torch.nn import functional as F
from scipy.stats import multivariate_normal


class Resize(object):
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size

    def __call__(self, sample):
        sample['image'] = resize(sample['image'], self.img_size)
        return sample


class GenerateHeatmap(object):
    def __init__(self, num_classes, output_size=(256, 256), cov=10):
        assert(num_classes is None)
        self.num_classes = num_classes
        self.output_size = output_size
        self.cov = cov

    def _get_box(self, points, h, w):
        points = points.copy()
        points[:, 0] *= w
        points[:, 1] *= h
        center = points.mean(axis=0)
        obj_w, obj_h = points.max(axis=0) - points.min(axis=0)
        return (center[1], center[0]), (obj_h / self.cov, obj_w / self.cov)

    def __call__(self, sample):
        boxes = sample['boxes']
        polygons = sample['polygons']
        box_labels = sample['box_labels']
        polygon_labels = sample['polygon_labels']

        h, w = self.output_size
        pos = np.dstack(np.mgrid[0: math.ceil(h), 0: math.ceil(w)])

        if self.num_classes is None:
            num = np.zeros((1, ), dtype=np.float32)
        else:
            num = np.zeros((self.num_classes, ), dtype=np.float32)

        mu = []
        sigma = []
        for pts, l in zip(boxes, box_labels):
            c, s = self._get_box(pts, h, w)
            mu.append(c)
            sigma.append(s)
            if self.num_classes is None:
                num[0] += 1
            else:
                num[l] += 1

        for pts, l in zip(polygons, polygon_labels):
            c, s = self._get_box(pts, h, w)
            mu.append(c)
            sigma.append(s)
            if self.num_classes is None:
                num[0] += 1
            else:
                num[l] += 1

        mu = np.array(mu)
        sigma = np.array(sigma)
        n = mu.shape[0]

        if n > 0:
            heatmap = pos - mu.reshape(n, 1, 1, 2)
            heatmap = heatmap / sigma.reshape(n, 1, 1, 2)
            heatmap = (heatmap ** 2).sum(axis=-1)
            heatmap = np.exp(-heatmap / 2)
            heatmap = heatmap.max(axis=0, keepdims=True)
        else:
            heatmap = np.zeros((1, h, w))

#         if self.num_classes is None:
#             heatmap = heatmap.squeeze()
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
        mask[label + 1, points[0, 1]: points[1, 1],
             points[0, 0]: points[1, 0]] = self.order[label]

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
        mask = np.zeros((self.num_classes + 1, math.ceil(h),
                         math.ceil(w)), dtype=np.uint8)

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
        sample['image'] = torch.from_numpy(np.transpose(
            sample['image'], (2, 0, 1))).type(torch.float32)
        sample['mask'] = torch.from_numpy(sample['mask']).type(torch.int64)
        if 'heatmap' in sample:
            sample['heatmap'] = torch.from_numpy(
                sample['heatmap']).type(torch.float32)
        if 'num' in sample:
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


class GenerateBoxMap(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.h, self.w = self.output_size
        self.pos = torch.from_numpy(np.dstack(
            np.mgrid[0: math.ceil(self.h), 0: math.ceil(self.w)])).type(torch.float32)

    def _get_box(self, points):
        points = points.copy()
        points[:, 0] *= self.w
        points[:, 1] *= self.h
        pt1 = points.min(axis=0)
        pt2 = points.max(axis=0)
        center = (pt1 + pt2) / 2
        obj_w, obj_h = pt2 - pt1
        return (center[1], center[0]), (obj_h, obj_w)

    def __call__(self, sample):
        boxes = sample['boxes']
        polygons = sample['polygons']
        centers = []
        sizes = []
        for pts in boxes:
            c, s = self._get_box(pts)
            centers.append(c)
            sizes.append(s)

        for pts in polygons:
            c, s = self._get_box(pts)
            centers.append(c)
            sizes.append(s)

        if len(centers) == 0:
            sample['offset_map'] = torch.zeros((2, self.h, self.w))
            sample['size_map'] = torch.zeros((2, self.h, self.w))
            return sample

        centers = torch.tensor(centers)
        sizes = torch.tensor(sizes)

        delta = centers.view(-1, 1, 1, 2) - self.pos.view(1, self.h, self.w, 2)
        distance = (delta ** 2).sum(dim=-1)
        index = distance.argmin(dim=0).view(
            1, self.h, self.w, 1).repeat(1, 1, 1, 2)

        delta_map = torch.gather(delta, 0, index).squeeze()

        size_map = sizes.view(-1, 1, 1, 2).repeat(1, self.h, self.w, 1)
        size_map = torch.gather(size_map, 0, index).squeeze()
        sample['offset_map'] = delta_map.permute(2, 0, 1)
        sample['size_map'] = size_map.permute(2, 0, 1)
        return sample


class GenerateBoxMapV2(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.h, self.w = self.output_size
        self.pos = np.dstack(
            np.mgrid[0: math.ceil(self.h), 0: math.ceil(self.w)])

    def _get_box(self, points):
        points = points.copy()
        points[:, 0] *= self.w
        points[:, 1] *= self.h
        pt1 = points.min(axis=0)
        pt2 = points.max(axis=0)
        center = (pt1 + pt2) / 2
        obj_w, obj_h = pt2 - pt1
        return points, (center[1], center[0]), (obj_h, obj_w)

    def __call__(self, sample):
        boxes = sample['boxes']
        polygons = sample['polygons']
        box_labels = sample['box_labels']
        polygon_labels = sample['polygon_labels']
        mask = sample['mask']

        offset_map = np.zeros((self.h, self.w, 2), dtype=np.float32)
        size_map = np.zeros((self.h, self.w, 2), dtype=np.float32)
        for pts, l in zip(boxes, box_labels):
            l += 1
            pts, c, s = self._get_box(pts)
            offset_map[mask == l] = cv2.fillPoly(offset_map.copy(),
                                                 np.round(np.expand_dims(pts, axis=0)).astype(np.int64), c)[mask == l]
            size_map[mask == l] = cv2.fillPoly(size_map.copy(),
                                               np.round(np.expand_dims(pts, axis=0)).astype(np.int64), s)[mask == l]

        for pts, l in zip(polygons, polygon_labels):
            l += 1
            pts, c, s = self._get_box(pts)
            offset_map[mask == l] = cv2.fillPoly(offset_map.copy(),
                                                 np.round(np.expand_dims(pts, axis=0)).astype(np.int64), c)[mask == l]
            size_map[mask == l] = cv2.fillPoly(size_map.copy(), 
                                               np.round(np.expand_dims(pts, axis=0)).astype(np.int64), s)[mask == l]

        offset_map -= self.pos
        sample['offset_map'] = torch.from_numpy(offset_map).permute(2, 0, 1)
        sample['size_map'] = torch.from_numpy(size_map).permute(2, 0, 1)
        return sample