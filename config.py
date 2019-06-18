#coding=utf-8

# train_dataset_dir = 'data/VOCdevkit/VOC2012'#'/mnt/Data/songliang/meisu/美素-新地堆-自采-20190531-2/'#'data/VOCdevkit/'
# vali_dataset_dir = 'data/VOCdevkit/VOC2007'#'/mnt/Data/songliang/meisu/美素-新地堆-自采-20190531-2/'
# image_dir = 'JPEGImages/'
# anno_dir = 'Annotations/'
train_dataset_dir = '/mnt/Data/songliang/jinmailang/20190531/'
vali_dataset_dir = '/mnt/Data/songliang/jinmailang/20190531/'
image_dir = 'images'
anno_dir = 'Annotations_original'

# train_image_sets = [('2007', 'trainval'), ('2012', 'train')]
# vali_image_sets = [('2012', 'val')]
output_stride = 8
feature_channels = [256, 512, 1024, 2048]
img_size = (256, 256)
cov = 16

optimizer = 'sgd'
lr = 1e-2
max_epoch = 50
ratio = 3
log_size = (128, 128)

devices = [0, 1, 2, 3]
batch_size = 8 * len(devices)
epoch_size = 100 * batch_size

checkpoint_path = None#'saved_models/best_model_Detection ratio: 3 num_classes: 20 with_FPN: True cov: 2.pt'

# CLASSES = None
# CLASSES = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = {'02686828-cd28-11e8-a38f-0242cb7ccd7c',
 '028527fe-709c-11e9-b9c1-0242cb7ccd7c',
 '0c1d3150-cd28-11e8-a49f-0242cb7ccd7c',
 '1008bc9c-cd28-11e8-8f4a-0242cb7ccd7c',
 '178c96fe-cd28-11e8-8fbf-0242cb7ccd7c',
 '17c1389c-81b9-11e9-9166-0242cb7ccd7c',
 '1e4a4a28-81b9-11e9-b3cf-0242cb7ccd7c',
 '4e81344c-d834-11e8-b88a-0242cb7ccd7c',
 '58cdd902-709c-11e9-bf8f-0242cb7ccd7c',
 '58e96762-7e06-11e9-9716-0242cb7ccd7c',# 冰柜
 '5b6e848c-709c-11e9-beab-0242cb7ccd7c',
 '5f26f1dc-709b-11e9-8245-0242cb7ccd7c',
 '61fbeba6-cd28-11e8-ac51-0242cb7ccd7c',
 '633a3c80-709c-11e9-a5e1-0242cb7ccd7c',
 '65116290-709b-11e9-812f-0242cb7ccd7c',
 '6611be40-709c-11e9-9cfc-0242cb7ccd7c',
 '67edf198-709b-11e9-b9f1-0242cb7ccd7c',
 '68f6e19e-709c-11e9-a5f3-0242cb7ccd7c',
 '6aba6d7e-709b-11e9-b6f1-0242cb7ccd7c',
 '6ba10f1e-709c-11e9-8afc-0242cb7ccd7c',
 '6da0931e-709b-11e9-9c71-0242cb7ccd7c',
 '6edeccac-7526-11e9-b968-0242cb7ccd7c',
 '6f43882c-cd28-11e8-aa18-0242cb7ccd7c',
 '710befa4-709b-11e9-b0f9-0242cb7ccd7c',
 '724d4a2c-cd28-11e8-af66-0242cb7ccd7c',
 '7253cdc8-7526-11e9-a703-0242cb7ccd7c',
 '73132888-cd27-11e8-bf24-0242cb7ccd7c',
 '74e3a862-709b-11e9-87da-0242cb7ccd7c',
 '7572ca8c-cd28-11e8-a7d5-0242cb7ccd7c',
 '76f09252-709c-11e9-b1fd-0242cb7ccd7c',
 '779fe0f0-cd27-11e8-b9eb-0242cb7ccd7c',
 '79d3738a-cd28-11e8-b645-0242cb7ccd7c',
 '7b042386-709c-11e9-a377-0242cb7ccd7c',
 '7b4162c8-709b-11e9-98ac-0242cb7ccd7c',
 '7bd4597a-cd27-11e8-a068-0242cb7ccd7c',
 '7ce45306-cd28-11e8-8944-0242cb7ccd7c',
 '7e1b216c-709b-11e9-87fb-0242cb7ccd7c',
 '7f062f80-cd27-11e8-bde8-0242cb7ccd7c',
 '80a22440-709b-11e9-b969-0242cb7ccd7c',
 '820b72fe-709c-11e9-a8be-0242cb7ccd7c',
 '8238ed06-cd28-11e8-b8ae-0242cb7ccd7c',
 '83676630-709b-11e9-8c3e-0242cb7ccd7c',
 '84db92c2-cd28-11e8-a2a9-0242cb7ccd7c',
 '863c67a6-709b-11e9-ba02-0242cb7ccd7c',
 '8746f692-cd28-11e8-b025-0242cb7ccd7c',
 '88f1b242-709b-11e9-86c9-0242cb7ccd7c',
 '91cef4ee-709b-11e9-949e-0242cb7ccd7c',
 '94747274-709b-11e9-90ec-0242cb7ccd7c',
 '96c5b082-709b-11e9-ba00-0242cb7ccd7c',
 '99343764-709b-11e9-b781-0242cb7ccd7c',
 '9bcea7ca-709b-11e9-9bdd-0242cb7ccd7c',
 'a3401c42-709b-11e9-ac43-0242cb7ccd7c',
 'b334aa8a-709b-11e9-9ade-0242cb7ccd7c',
 'b74835d2-cd27-11e8-926b-0242cb7ccd7c',
 'ba31adf4-cd27-11e8-9c32-0242cb7ccd7c',
 'bd379b08-cd27-11e8-ad33-0242cb7ccd7c',
 'c3a3ffb6-cd27-11e8-9051-0242cb7ccd7c',
 'e8248ad0-81bb-11e9-8643-0242cb7ccd7c',
 'ec89a3fe-709b-11e9-b9c9-0242cb7ccd7c',
 'f29f889a-cd27-11e8-99ba-0242cb7ccd7c',
 'f58f7258-cd27-11e8-bf8c-0242cb7ccd7c',
 'fcacc7e8-cd27-11e8-9f14-0242cb7ccd7c',
 'ff5872ba-cd27-11e8-bd2d-0242cb7ccd7c'}

num_classes = 1 if CLASSES is None else len(CLASSES)