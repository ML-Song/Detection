import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone='resnet', output_stride=16,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        self.num_classes = num_classes
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder_seg = build_decoder(num_classes + 1, backbone, BatchNorm)
        self.decoder_box = build_decoder(4, backbone, BatchNorm)
        self.pos = None
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        
        if self.pos is None:
            n, c, h, w = input.shape
            pos = np.dstack(np.mgrid[0: h, 0: w])
            pos = torch.from_numpy(pos).unsqueeze(dim=0).type(torch.float32)
            pos = pos.permute(0, 3, 1, 2)
            self.pos = pos.to(x.device)
            
        mask = self.decoder_seg(x, low_level_feat)
        box = self.decoder_box(x, low_level_feat)
        
        mask = F.interpolate(mask, size=input.size()[2:], mode='bilinear', align_corners=True)
        
        offset = box[:, : 2]
        size = box[:, 2:]
        offset = F.interpolate(offset, size=input.size()[2:], mode='bilinear', align_corners=True)
        offset = offset + self.pos
        size = F.interpolate(size, size=input.size()[2:])
        return mask, offset, size

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder_seg, self.decoder_box]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


