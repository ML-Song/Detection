#coding=utf-8
import os
import torch
import torchvision as tv
from tensorboardX import SummaryWriter

from config import *
from detector import Detector
from utils import visualization
from dataset import voc, augmentations
from net import center_net, xception, resnet_atrous


if __name__ == '__main__':
    checkpoint_name = 'Detection ratio: {} num_classes: {} with_FPN: {} radius: {}'.format(
        ratio, num_classes, feature_channels is not None, radius)
    comment = 'Detection ratio: {} num_classes: {} with_FPN: {} radius: {}'.format(
        ratio, num_classes, feature_channels is not None, radius)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))
    
    train_transforms = tv.transforms.Compose([
        augmentations.Resize(img_size), 
        augmentations.BoxToHeatmap(num_classes, output_stride, radius), 
        augmentations.ToTensor(), 
    ])
    
    vali_transforms = tv.transforms.Compose([
        augmentations.Resize(img_size), 
        augmentations.BoxToHeatmap(num_classes, output_stride, radius), 
        augmentations.ToTensor()
    ])
    
    train_set = voc.VOCDetection(dataset_dir, 
                           image_sets=train_image_sets, 
                           transform=train_transforms)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_set, True, epoch_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=16, sampler=train_sampler)
    
    vali_set = voc.VOCDetection(dataset_dir, 
                           image_sets=vali_image_sets, 
                           transform=vali_transforms)
    vali_sampler = torch.utils.data.sampler.RandomSampler(vali_set, True, epoch_size)
    vali_loader = torch.utils.data.DataLoader(vali_set, batch_size=batch_size, num_workers=16, sampler=vali_sampler)
    
    backbone = resnet_atrous.resnet50_atrous(pretrained=True, output_stride=output_stride)
    model = center_net.CenterNet(backbone, num_classes, feature_channels)
    solver = Detector(model, train_loader, vali_loader, batch_size, optimizer=optimizer, lr=lr, 
                      checkpoint_name=checkpoint_name, devices=devices, ratio=ratio)
    
    if checkpoint_path:
        solver.load_model(checkpoint_path)
    with SummaryWriter(comment=comment) as writer:
        solver.train(max_epoch, writer, epoch_size // batch_size)