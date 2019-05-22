#coding=utf-8

dataset_dir = 'data/VOCdevkit/'
image_sets = [('2007', 'trainval'), ('2012', 'train')]
batch_size = 1
output_stride = 8
num_classes = 20

optimizer = 'sgd'
lr = 1e-2
max_epoch = 10

devices = [0]

checkpoint_path = None