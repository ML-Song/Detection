#coding=utf-8
import os
import tqdm
import torch
from torch import nn
import torchvision as tv
from sklearn import metrics
import torchvision.utils as vutils
from torch.nn import functional as F

from dataset import augmentations
from utils.losses import CenterLoss


class Detector(object):
    def __init__(self, net, train_loader=None, test_loader=None, batch_size=None, 
                 optimizer='adam', lr=1e-3, patience=5, interval=1, transfrom=None, 
                 checkpoint_dir='saved_models', checkpoint_name='', devices=[0], ratio=1):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.devices = devices
        self.ratio = ratio
        
        if transfrom is None:
            self.transfrom = tv.transforms.Compose([
                augmentations.BoxToHeatmap(20, 8), 
                augmentations.ToTensor()
            ])
        else:
            self.transfrom = transfrom
        
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            
        self.net_single = net
        if len(devices) == 0:
            pass
        elif len(devices) == 1:
            self.net = self.net_single.cuda()
        else:
            self.net = nn.DataParallel(self.net_single, device_ids=range(len(devices))).cuda()
            
        if optimizer == 'sgd':
            self.opt = torch.optim.SGD(
                self.net_single.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
        elif optimizer == 'adam':
            self.opt = torch.optim.Adam(
                self.net_single.parameters(), lr=lr, weight_decay=1e-4)
        else:
            raise Exception('Optimizer {} Not Exists'.format(optimizer))

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='max', factor=0.2, patience=patience)
        self.criterion = CenterLoss(ratio=ratio)
        
    def reset_grad(self):
        self.opt.zero_grad()
        
    def train(self, max_epoch, writer=None):
        torch.cuda.manual_seed(1)
        best_score = 0
        step = 1
        for epoch in tqdm.tqdm(range(max_epoch), total=max_epoch):
            self.net.train()
            for batch_idx, data in enumerate(self.train_loader):
                img = data[0].cuda()
                label = data[1].cuda()

                self.reset_grad()
                out = self.net(img)
                loss = self.get_loss(out, label)
                loss.backward()
                self.opt.step()
                if writer:
                    writer.add_scalar(
                        'loss', loss.data, global_step=step)
                step += 1
            if epoch % self.interval == 0:
                torch.cuda.empty_cache()
                acc, imgs, detections = self.test()
                if writer:
                    writer.add_scalar(
                        'lr', self.opt.param_groups[0]['lr'], global_step=epoch)
                    writer.add_scalar(
                        'acc', acc, global_step=epoch)
                    score = acc
                    detections = vutils.make_grid(detections, normalize=False, scale_each=True)
                    writer.add_image('Detection', detections, epoch)
                    
                    imgs = vutils.make_grid(imgs, normalize=True, scale_each=True)
                    writer.add_image('Imgs', imgs, epoch)
                
                self.scheduler.step(score)
                if best_score < score:
                    best_score = score
                    self.save_model(self.checkpoint_dir)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            pred = []
            gt = []
            imgs = []
            detections = []
            for batch_idx, data in enumerate(self.test_loader):
                img = data[0].cuda()
                label = data[1]
                out = self.net(img)
                out = torch.sigmoid(out).detach().cpu()
                
                prob, cls = out.max(dim=1, keepdim=True)
                cls = (cls + 1) * 10
                cls[prob < 0.5] = 0
                detections.append(cls)
                
                imgs.append(img.cpu())
                out = out > 0
                pred.append(out)
                gt.append(label)
                if batch_idx == 8:
                    break
#             pred = torch.cat(pred).numpy()
#             gt = torch.cat(gt).numpy()
            detections = torch.cat(detections)
            imgs = torch.cat(imgs)
#             acc = metrics.accuracy_score(gt, pred)
        return 0, imgs, detections

    def save_model(self, checkpoint_dir, comment=None):
        if comment is None:
            torch.save(self.net_single, '{}/best_model_{}.pt'.format(checkpoint_dir, self.checkpoint_name))
        else:
            torch.save(self.net_single, '{}/best_model_{}_{}.pt'.format(checkpoint_dir, self.checkpoint_name, comment))
            
    def load_model(self, model_path):
        self.net_single.load_state_dict(torch.load(model_path).state_dict())
    
    def predict(self, img, target=None):
        x = torch.cat([self.transfrom(i).unsqueeze(dim=0) for i in img], dim=0).cuda()
        self.net.eval()
        with torch.no_grad():
            out = self.net(x).detach().cpu().numpy()
        return out
    
    def get_loss(self, pred, target):
        loss = self.criterion(pred, target)
        return loss