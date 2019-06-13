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
                 optimizer='adam', lr=1e-3, patience=5, interval=1, 
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

#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.opt, mode='max', factor=0.2, patience=patience)
        self.criterion = CenterLoss(ratio=ratio)
        
    def reset_grad(self):
        self.opt.zero_grad()
        
    def train(self, max_epoch, writer=None, epoch_size=100):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, max_epoch * epoch_size)
        torch.cuda.manual_seed(1)
        best_score = 1000
        step = 1
        for epoch in tqdm.tqdm(range(max_epoch), total=max_epoch):
            torch.cuda.empty_cache()
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
                    writer.add_scalar(
                        'lr', self.opt.param_groups[0]['lr'], global_step=step)
                step += 1
                scheduler.step(step)
            if epoch % self.interval == 0:
                torch.cuda.empty_cache()
                total_loss, imgs, detections, gt = self.test()
                if writer:
                    writer.add_scalar(
                        'Test Loss', total_loss, global_step=epoch)
                    score = total_loss
                    detections = vutils.make_grid(detections, normalize=False, scale_each=True)
                    writer.add_image('Detection', detections, epoch)
                    
                    gt = vutils.make_grid(gt, normalize=False, scale_each=True)
                    writer.add_image('GroundTruth', gt, epoch)
                    
                    imgs = vutils.make_grid(imgs, normalize=True, scale_each=True)
                    writer.add_image('Imgs', imgs, epoch)
                
#                 self.scheduler.step(score)
                if best_score > score:
                    best_score = score
                    self.save_model(self.checkpoint_dir)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            pred = []
            gt = []
            imgs = []
            detections = []
            total_loss = 0
            for batch_idx, data in enumerate(self.test_loader):
                img = data[0].cuda()
                label = data[1].cuda()
                out = self.net(img)
                loss = self.get_loss(out, label)
                total_loss += loss.detach().cpu().data
                out = torch.sigmoid(out).detach().cpu()
                
                prob, cls = out.max(dim=1, keepdim=True)
#                 cls = (cls + 1) * 10
#                 cls[prob < 0.5] = 0
                detections.append(prob)
                
                imgs.append(F.interpolate(img, scale_factor=1/4).cpu())
                out = out > 0
                pred.append(out)
                gt.append(label.cpu())
                if batch_idx == 8:
                    break
#             pred = torch.cat(pred).numpy()
            gt = torch.cat(gt)
            detections = torch.cat(detections)
            imgs = torch.cat(imgs)
            total_loss /= batch_idx
#             acc = metrics.accuracy_score(gt, pred)
        return total_loss, imgs, detections, gt

    def save_model(self, checkpoint_dir, comment=None):
        if comment is None:
            torch.save(self.net_single, '{}/best_model_{}.pt'.format(checkpoint_dir, self.checkpoint_name))
        else:
            torch.save(self.net_single, '{}/best_model_{}_{}.pt'.format(checkpoint_dir, self.checkpoint_name, comment))
            
    def load_model(self, model_path):
        self.net_single.load_state_dict(torch.load(model_path).state_dict())
    
    def predict(self, img):
        x = torch.from_numpy(img).type(torch.float32).permute(0, 3, 1, 2).cuda()
        self.net.eval()
        with torch.no_grad():
            out = torch.sigmoid(self.net(x)).detach().cpu().numpy()
        return out
    
    def get_loss(self, pred, target):
        loss = self.criterion(pred, target)
        return loss