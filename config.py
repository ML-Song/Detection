#coding=utf-8

dataset_dir = '/mnt/Data/songliang/meisu/美素-新地堆-自采-20190531-2/'#'data/VOCdevkit/'
# train_image_sets = [('2007', 'trainval'), ('2012', 'train')]
# vali_image_sets = [('2012', 'val')]
batch_size = 32
epoch_size = 100 * batch_size
output_stride = 8
feature_channels = [256, 512, 1024, 2048]
num_classes = 1
img_size = (416, 416)
# radius = 1
cov = 10

optimizer = 'sgd'
lr = 1e-3
max_epoch = 500
ratio = 3

devices = [0, 1, 2, 3]

checkpoint_path = 'saved_models/best_model_Detection ratio: 3 num_classes: 1 with_FPN: True radius: 1.pt'
