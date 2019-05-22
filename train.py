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
    checkpoint_name = 'Detection'#.format(targeted)
    comment = 'Detection'#.format(targeted)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, devices))
    
    train_transforms = tv.transforms.Compose([
        augmentations.Resize(img_size), 
        augmentations.BoxToHeatmap(num_classes, output_stride), 
        augmentations.ToTensor(), 
    ])
    
    vali_transforms = tv.transforms.Compose([
        augmentations.Resize(img_size), 
        augmentations.BoxToHeatmap(num_classes, output_stride), 
        augmentations.ToTensor()
    ])
    
    train_set = voc.VOCDetection(dataset_dir, 
                           image_sets=train_image_sets, 
                           transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    vali_set = voc.VOCDetection(dataset_dir, 
                           image_sets=vali_image_sets, 
                           transform=vali_transforms)
    vali_loader = torch.utils.data.DataLoader(vali_set, batch_size=batch_size, shuffle=True)
    
    backbone = resnet_atrous.resnet50_atrous(pretrained=True, output_stride=output_stride)
    model = center_net.CenterNet(backbone, num_classes)
    solver = Detector(model, train_loader, vali_loader, batch_size, optimizer=optimizer, lr=lr, 
                      checkpoint_name=checkpoint_name, devices=devices)
    
    if checkpoint_path:
        solver.load_model(checkpoint_path)
    with SummaryWriter(comment=comment) as writer:
        solver.train(max_epoch, writer)