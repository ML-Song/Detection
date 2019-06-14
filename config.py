#coding=utf-8

dataset_dir = '/mnt/Data/songliang/meisu/美素-新地堆-自采-20190531-2/'#'data/VOCdevkit/'
# train_image_sets = [('2007', 'trainval'), ('2012', 'train')]
# vali_image_sets = [('2012', 'val')]
batch_size = 48
epoch_size = 1 * batch_size
output_stride = 8
feature_channels = [256, 512, 1024, 2048]
num_classes = 1
img_size = (416, 416)
cov = 100

optimizer = 'sgd'
lr = 1e-3
max_epoch = 50
ratio = 3

devices = [0, 1, 2, 3]

checkpoint_path = None#'saved_models/best_model_Detection ratio: 3 num_classes: 1 with_FPN: True cov: 20.pt'
