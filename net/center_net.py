#coding=utf-8
from torch import nn
import torch.nn.functional as F
from .sync_batchnorm import SynchronizedBatchNorm2d

bn_mom = 0.0003
class CenterNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        feature_channels = CenterNet._get_output_shape(backbone)
        self.rpn = nn.Sequential(*[
            nn.Conv2d(feature_channels, 128, 3, stride=1, padding=1), 
            SynchronizedBatchNorm2d(128, momentum=bn_mom), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, num_classes, 1), 
        ])
        
    def forward(self, x):
        feature_map = self.backbone(x)
        region = self.rpn(feature_map)
        return region
        
    @staticmethod
    def _get_output_shape(model):
        for layer in list(model.modules())[::-1]:
            if isinstance(layer, nn.Conv2d):
                break
        return layer.out_channels
    