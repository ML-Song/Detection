#coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
from .sync_batchnorm import SynchronizedBatchNorm2d

bn_mom = 0.0003
class CenterNet(nn.Module):
    def __init__(self, backbone, num_classes, feature_channels=None):
        super().__init__()
        self.backbone = backbone
        self.feature_channels = feature_channels
        if not isinstance(feature_channels, list):
            if feature_channels is None:
                feature_channels = CenterNet._get_output_shape(backbone)
            self.conv = nn.Sequential(*[
                nn.Conv2d(feature_channels, 256, 3, stride=1, padding=1), 
                nn.LeakyReLU(inplace=True), 
                SynchronizedBatchNorm2d(256, momentum=bn_mom), 
                nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1), 
                nn.LeakyReLU(inplace=True), 
                SynchronizedBatchNorm2d(256, momentum=bn_mom), 
            ])
        else:
            self.conv = nn.Sequential(*[
                nn.Sequential(*[
                    nn.Conv2d(c, 256, 3, stride=1, padding=1), 
                    nn.LeakyReLU(inplace=True), 
                    SynchronizedBatchNorm2d(256, momentum=bn_mom), 
                    nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1), 
                    nn.LeakyReLU(inplace=True), 
                    SynchronizedBatchNorm2d(256, momentum=bn_mom), 
                ])
                for i, c in enumerate(feature_channels)])
        self.rpn = nn.Sequential(*[
            nn.Conv2d(256, num_classes, 3, padding=1), 
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        feature_map = self.backbone(x)
        if not isinstance(self.feature_channels, list):
            feature_map = self.conv(feature_map)
            region = self.rpn(feature_map)
        else:
            regions = []
            feature_maps = self.backbone.get_layers()
            for i, f in enumerate(feature_maps):
                if i == 0:
                    f = F.interpolate(f, scale_factor=1/2)
                f = self.conv[i](f)
                region = self.rpn(f)
                regions.append(region.unsqueeze(dim=1))
            regions = torch.cat(regions, dim=1)
            region = regions.max(dim=1)[0]
        return region
        
    @staticmethod
    def _get_output_shape(model):
        for layer in list(model.modules())[::-1]:
            if isinstance(layer, nn.Conv2d):
                break
        return layer.out_channels
    