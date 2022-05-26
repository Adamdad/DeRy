# dataset settings
dataset_type = 'CUB'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
policies = [
    dict(type='AutoContrast'),
    dict(type='Brightness', magnitude_key='magnitude',
         magnitude_range=(0.05, 0.95)),
    dict(type='ColorTransform', magnitude_key='magnitude',
         magnitude_range=(0.05, 0.95)),
    dict(type='Contrast', magnitude_key='magnitude',
         magnitude_range=(0.05, 0.95)),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(type='Sharpness', magnitude_key='magnitude',
         magnitude_range=(0.05, 0.95)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 8)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(0, 256)),
    dict(type='Rotate', interpolation='bicubic', magnitude_key='angle',
         pad_val=tuple([int(x) for x in img_norm_cfg['mean']]),
         magnitude_range=(-30, 30)),
    dict(
        type='Shear',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(-0.3, 0.3),
        pad_val=tuple([int(x) for x in img_norm_cfg['mean']]),
        direction='horizontal'),
    dict(
        type='Shear',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(-0.3, 0.3),
        pad_val=tuple([int(x) for x in img_norm_cfg['mean']]),
        direction='vertical'),
    dict(
        type='Translate',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(-0.3, 0.3),
        pad_val=tuple([int(x) for x in img_norm_cfg['mean']]),
        direction='horizontal'),
    dict(
        type='Translate',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(-0.3, 0.3),
        pad_val=tuple([int(x) for x in img_norm_cfg['mean']]),
        direction='vertical')
]

train_pipeline = [
    dict(type='LoadImageFromFile'),  # **
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandAugment',
         policies=policies,
         num_policies=2,
         total_level=10,
         magnitude_level=9,
         magnitude_std=0.5),
    dict(type='Albu',
         transforms=[
              dict(type='Blur', blur_limit=3, p=0.1),
             dict(type='GaussNoise', var_limit=10.0, p=0.1)
         ]),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

dataset_num_classes = 200
data_root = 'data/CUB_200_2011/'
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'images.txt',
        image_class_labels_file=data_root + 'image_class_labels.txt',
        train_test_split_file=data_root + 'train_test_split.txt',
        data_prefix=data_root + 'images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'images.txt',
        image_class_labels_file=data_root + 'image_class_labels.txt',
        train_test_split_file=data_root + 'train_test_split.txt',
        data_prefix=data_root + 'images',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'images.txt',
        image_class_labels_file=data_root + 'image_class_labels.txt',
        train_test_split_file=data_root + 'train_test_split.txt',
        data_prefix=data_root + 'images',
        test_mode=True,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metric='accuracy',
    save_best='auto')  # save the checkpoint with highest accuracy