# dataset settings
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)

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
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
     dict(type='Albu',
         transforms=[
             dict(
                 type='RandomBrightnessContrast',
                 brightness_limit=[0.1, 0.3],
                 contrast_limit=[0.1, 0.3],
                 p=0.2),
             dict(
                 type='OneOf',
                 transforms=[
                     dict(type='Blur', blur_limit=3, p=1.0),
                     dict(type='MedianBlur', blur_limit=3, p=1.0),
                     dict(type='GaussNoise', var_limit=20.0,p=1.0)
                 ],
                 p=0.5),
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/cifar10',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True))
