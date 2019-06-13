#coding=utf-8
import cv2
import numpy as np

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