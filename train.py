#coding=utf-8
import os
import cv2
import torch
import torchvision as tv
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from config import *
from detector import Detector
from utils import visualization
from modeling import deeplab, pano_seg
from dataset import detection, augmentations


if __name__ == '__main__':
    checkpoint_name = 'Detection num_classes: {}'.format(num_classes)
    comment = 'Detection num_classes: {}'.format(num_classes)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))
    
    train_transforms = tv.transforms.Compose([
        augmentations.Resize(img_size), 
        augmentations.GenerateBoxMapV2(output_size), 
        augmentations.GenerateMask(num_classes, output_size), 
        augmentations.ToTensor(), 
    ])
    train_set = detection.DetectionDataset(os.path.join(train_dataset_dir, image_dir), 
                                           os.path.join(train_dataset_dir, anno_dir), 
                                           class_map, 
                                           train_transforms)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_set, True, epoch_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                               num_workers=num_workers, sampler=train_sampler, 
                                               drop_last=True, pin_memory=True)
    
    vali_transforms = tv.transforms.Compose([
        augmentations.Resize(img_size), 
        augmentations.GenerateBoxMapV2(output_size), 
        augmentations.GenerateMask(num_classes, output_size), 
        augmentations.ToTensor(), 
    ])
    vali_set = detection.DetectionDataset(os.path.join(vali_dataset_dir, image_dir), 
                                          os.path.join(vali_dataset_dir, anno_dir), 
                                          class_map, 
                                          vali_transforms)
    vali_sampler = torch.utils.data.sampler.RandomSampler(vali_set, True, epoch_size)
    vali_loader = torch.utils.data.DataLoader(vali_set, batch_size=batch_size, 
                                               num_workers=num_workers, sampler=vali_sampler, pin_memory=True)
    
    backbone = deeplab.DeepLab(num_classes)
    model = pano_seg.PanopticSegment(backbone)
    solver = Detector(model, train_loader, vali_loader, batch_size, optimizer=optimizer, lr=lr,  
                      checkpoint_name=checkpoint_name, devices=devices, 
                      num_classes=num_classes, log_size=log_size)
    
    if checkpoint_path:
        solver.load_model(checkpoint_path)
    with SummaryWriter(comment=comment) as writer:
        solver.train(max_epoch, writer, epoch_size // batch_size)