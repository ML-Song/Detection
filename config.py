#coding=utf-8

train_dataset_dir = 'data/VOCdevkit/VOC2012'#'/mnt/Data/songliang/meisu/美素-新地堆-自采-20190531-2/'#'data/VOCdevkit/'
vali_dataset_dir = 'data/VOCdevkit/VOC2007'#'/mnt/Data/songliang/meisu/美素-新地堆-自采-20190531-2/'
image_dir = 'JPEGImages/'
anno_dir = 'Annotations/'
# train_image_sets = [('2007', 'trainval'), ('2012', 'train')]
# vali_image_sets = [('2012', 'val')]
batch_size = 96
epoch_size = 100 * batch_size
output_stride = 4
feature_channels = [256, 512, 1024, 2048]
img_size = (256, 256)
cov = 4
loss_step = 1

optimizer = 'sgd'
lr = 1e-2
max_epoch = 50
ratio = 3

devices = [1, 2, 3]

checkpoint_path = None#'saved_models/best_model_Detection ratio: 3 num_classes: 20 with_FPN: True cov: 30.pt'

CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
num_classes = 1 if CLASSES is None else len(CLASSES)
