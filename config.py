#coding=utf-8

dataset_dir = 'data/VOCdevkit/'
train_image_sets = [('2007', 'trainval'), ('2012', 'train')]
vali_image_sets = [('2012', 'val')]
batch_size = 8
epoch_size = 100 * batch_size
output_stride = 8
num_classes = 20
img_size = (416, 416)

optimizer = 'sgd'
lr = 1e-2
max_epoch = 10
ratio = 3

devices = [0]

checkpoint_path = None