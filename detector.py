#coding=utf-8
import os
import math
import time
import tqdm
import torch
import numpy as np
from torch import nn
import torchvision as tv
from sklearn import metrics
import torchvision.utils as vutils
from torch.nn import functional as F

from modeling import pano_seg
from utils import visualization
from dataset import augmentations
from utils.losses import InstanceSegmentLoss
from utils.lr_scheduler import LR_Scheduler


class Detector(object):
    def __init__(self, net, train_loader=None, test_loader=None, batch_size=None, 
                 optimizer='adam', lr=1e-3, patience=5, interval=1, num_classes=1, prob_threshold=0.9, 
                 checkpoint_dir='saved_models', checkpoint_name='', devices=[0], log_size=(96, 96)):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.log_size = log_size
        self.prob_threshold = prob_threshold
        
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            
        self.net_single = net
        self.criterion = InstanceSegmentLoss(prob_threshold=prob_threshold)
        if len(devices) == 0:
            self.device = torch.device('cpu')
        elif len(devices) == 1:
            self.device = torch.device('cuda')
            self.net = self.net_single.to(self.device)
            self.criterion = self.criterion.to(self.device)
        else:
            self.device = torch.device('cuda')
            self.net = nn.DataParallel(self.net_single, device_ids=range(len(devices))).to(self.device)
            self.criterion = nn.DataParallel(self.criterion, device_ids=range(len(devices))).to(self.device)
            
        train_params = [{'params': self.net_single.backbone.get_1x_lr_params(), 'lr': lr},
                        {'params': self.net_single.backbone.get_10x_lr_params(), 'lr': lr * 10}]
        if optimizer == 'sgd':
            self.opt = torch.optim.SGD(
                train_params, lr=lr, weight_decay=5e-4, momentum=0.9)
        elif optimizer == 'adam':
            self.opt = torch.optim.Adam(
                train_params, lr=lr, weight_decay=5e-4)
        else:
            raise Exception('Optimizer {} Not Exists'.format(optimizer))
        
    def reset_grad(self):
        self.opt.zero_grad()
        
    def train(self, max_epoch, writer=None, epoch_size=100):
        max_step = epoch_size * max_epoch
        scheduler = LR_Scheduler('poly', self.lr, max_epoch, epoch_size)
        torch.cuda.manual_seed(1)
        best_score = 0
        step = 0
        for epoch in tqdm.tqdm(range(max_epoch), total=max_epoch):
            torch.cuda.empty_cache()
            self.net.train()
            for batch_idx, data in enumerate(self.train_loader):
                img = data['image'].to(self.device)
                mask = data['mask'].to(self.device)
                offset = data['offset_map'].to(self.device)
                size = data['size_map'].to(self.device)
                box = torch.cat((offset, size), dim=1)
                scheduler(self.opt, batch_idx, epoch, best_score)
                self.reset_grad()
                
                out = self.net(img)
                
                mask_loss, box_loss = self.get_loss(out, (mask, box), backward=False)
                loss = mask_loss + box_loss
                loss.backward()
                self.opt.step()
                if writer:
                    writer.add_scalar(
                        'loss', loss.data, global_step=step)
                    writer.add_scalar(
                        'mask_loss', mask_loss.data, global_step=step)
                    writer.add_scalar(
                        'box_loss', box_loss.data, global_step=step)
                    writer.add_scalar(
                        'lr', self.opt.param_groups[0]['lr'], global_step=step)
                step += 1
                
            if epoch % self.interval == 0:
                torch.cuda.empty_cache()
                acc, imgs, pred_boxes, pred_boxes_v2, gt_boxes, gt_boxes_v2, pred_masks, gt_masks = self.test()
                if writer:
                    writer.add_scalar(
                        'Acc', acc, global_step=epoch)
                    score = acc
                    
                    pred_panos = self.draw_pano(imgs, pred_masks, pred_boxes)
                    writer.add_image('Pred Pano', pred_panos, epoch)
                    
                    pred_panos_v2 = self.draw_pano(imgs, pred_masks, pred_boxes_v2)
                    writer.add_image('Pred Pano V2', pred_panos_v2, epoch)
                    
                    gt_panos = self.draw_pano(imgs, gt_masks, gt_boxes, is_gt=True)
                    writer.add_image('GT Pano', gt_panos, epoch)
                    
                    gt_panos_v2 = self.draw_pano(imgs, gt_masks, gt_boxes_v2, is_gt=True)
                    writer.add_image('GT Pano V2', gt_panos_v2, epoch)
                    
                if best_score <= score + 0.01:
                    best_score = score
                    self.save_model(self.checkpoint_dir)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            imgs = []
            gt_boxes = []
            gt_boxes_v2 = []
            pred_boxes = []
            pred_boxes_v2 = []
            gt_masks = []
            pred_masks = []
            acc = 0
            count = 0
            for batch_idx, data in enumerate(self.test_loader):
                img = data['image'].to(self.device)
                offset_map = data['offset_map']
                size_map = data['size_map']
                mask = data['mask']
                instance_map = data['instance_map']
                feat = instance_map.unsqueeze(dim=1).type(torch.float32)

                pred_mask, pred_feat = self.net(img)
                
                pred_feat = pred_feat.detach().cpu()
                pred_mask = pred_mask.detach().cpu()
                start = time.time()
                box = pano_seg.generate_box(offset_map, size_map, mask, 
                                            prob_threshold=self.prob_threshold, iou_threshold=1, topk=500)
#                 print(time.time() - start)
                box_v2 = pano_seg.generate_box_v2(feat, mask, eps=0.1)
                start = time.time()
#                 pred_box = pano_seg.generate_box(pred_feat[:, : 2], pred_feat[:, 2:], pred_mask, 
#                                                  iou_threshold=0.7, 
#                                                  prob_threshold=self.prob_threshold, 
#                                                  topk=500)
#                 print(time.time() - start)            
                pred_box_v2 = pano_seg.generate_box_v2(pred_feat, pred_mask, 
                                                       prob_threshold=self.prob_threshold)
                pred_box = pred_box_v2
                
                acc += 0

                img = img.cpu()
                imgs.append(img)

                pred_boxes.extend(pred_box)
                pred_boxes_v2.extend(pred_box_v2)
                gt_boxes.extend(box)
                gt_boxes_v2.extend(box_v2)
            
                pred_masks.append(pred_mask)
                gt_masks.append(mask)

                count += img.shape[0]
                if count >= 40:
                    break
                    
            gt_masks = torch.cat(gt_masks)
            pred_masks = torch.cat(pred_masks)
            imgs = torch.cat(imgs)
            acc /= batch_idx + 1
        return (acc, imgs[: 40], 
                pred_boxes[: 40], pred_boxes_v2[: 40], 
                gt_boxes[: 40], gt_boxes_v2[: 40], 
                pred_masks[: 40], gt_masks[: 40])
    
    def draw_pano(self, img, mask, boxes, size=None, is_gt=False):
        if size is None:
            size = self.log_size
        img_with_boxes = F.interpolate(img, size, mode='bilinear', align_corners=True)
        img_with_boxes = visualization.draw_boxes(img_with_boxes, boxes)
        img_with_boxes = vutils.make_grid(img_with_boxes)
        if not is_gt:
            mask = F.interpolate(mask, size, mode='bilinear', align_corners=True).cpu()
        else:
            mask[mask == 255] = 0
            mask = mask.type(torch.float32).unsqueeze(dim=1)
            mask = F.interpolate(mask, size, mode='bilinear', align_corners=True).cpu()
        rgb = visualization.mask_to_rgb(mask, self.num_classes, is_gt=is_gt)
        result = torch.clamp((rgb + img_with_boxes) / 2, 0, 1)
        return result

    def save_model(self, checkpoint_dir, comment=None):
        if comment is None:
            torch.save(self.net_single, '{}/best_model_{}.pt'.format(checkpoint_dir, self.checkpoint_name))
        else:
            torch.save(self.net_single, '{}/best_model_{}_{}.pt'.format(checkpoint_dir, self.checkpoint_name, comment))
            
    def load_model(self, model_path):
        self.net_single.load_state_dict(torch.load(model_path).state_dict())
    
    def get_loss(self, pred, target, backward=False):
        mask_loss, box_loss = self.criterion(pred, target, backward)
        return mask_loss.mean(), box_loss.mean()