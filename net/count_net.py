#coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
from .sync_batchnorm import SynchronizedBatchNorm2d

bn_mom = 0.0003
class CountNet(nn.Module):
    def __init__(self, backbone, num_classes, feature_channels=None):
        super().__init__()
        self.backbone = backbone
        self.with_fpn = feature_channels is not None
        
        if feature_channels is None:
            feature_channels = [CountNet._get_output_shape(backbone)]
            
#         self.convs = nn.Sequential(*[
#             nn.Sequential(*[
#                 nn.Conv2d(c, 256, 3, stride=1, padding=1), 
#                 nn.ReLU(inplace=True), 
#                 SynchronizedBatchNorm2d(256, momentum=bn_mom), 
#                 nn.Conv2d(256, 256, 3, stride=1, padding=1), 
#                 nn.ReLU(inplace=True), 
#                 SynchronizedBatchNorm2d(256, momentum=bn_mom), 
#             ]) for i, c in enumerate(feature_channels)])
        
        self.heatmap = nn.Sequential(*[
            nn.Conv2d(sum(feature_channels), 256, 3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            SynchronizedBatchNorm2d(256, momentum=bn_mom), 
            nn.Conv2d(256, num_classes, 1), 
#             nn.Sigmoid()
        ])
        
        self.mask = nn.Sequential(*[
            nn.Conv2d(sum(feature_channels), 256, 3, stride=1, padding=1), 
            nn.ReLU(inplace=True), 
            SynchronizedBatchNorm2d(256, momentum=bn_mom), 
            nn.Conv2d(256, num_classes, 1), 
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        feature_map = self.backbone(x)
        if self.with_fpn:
            feature_maps = self.backbone.get_layers()
        else:
            feature_maps = [feature_map]
            
        for i, f in enumerate(feature_maps):
            if i != 0 or not self.with_fpn:
                f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=True)
            feature_maps[i] = f#self.convs[i](f)
        feature_maps = torch.cat(feature_maps, dim=1)
        hm = self.heatmap(feature_maps)
        mask = self.mask(feature_maps)
        return hm, mask
        
    @staticmethod
    def _get_output_shape(model):
        for layer in list(model.modules())[::-1]:
            if isinstance(layer, nn.Conv2d):
                break
        return layer.out_channels
    