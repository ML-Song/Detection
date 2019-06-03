#coding=utf-8

dataset_dir = 'data/VOCdevkit/'
train_image_sets = [('2007', 'trainval'), ('2012', 'train')]
vali_image_sets = [('2012', 'val')]
batch_size = 16
epoch_size = 100 * batch_size
output_stride = 8
feature_channels = [256, 512, 1024, 2048]
num_classes = 1
img_size = (416, 416)
radius = 1

optimizer = 'sgd'
lr = 1e-3
max_epoch = 50
ratio = 3

devices = [0, 1]

checkpoint_path = None
