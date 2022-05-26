# dataset settings
dataset_type = 'CALTECH'
# to_rgb first

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),  # **
    dict(type='Resize', size=(224, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
dataset_num_classes = 102
data = dict(
    samples_per_gpu=64,  # batchsize
    workers_per_gpu=3,
    train=dict(type=dataset_type,
               data_prefix='data/caltech101/train/',
            #    ann_file='data/dataset_name/train_meta.list',
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             data_prefix='data/caltech101/test/',
            #  ann_file='data/dataset_name/test_meta.list',
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              data_prefix='data/caltech101/test/',
            #   ann_file='data/dataset_name/test_meta.list',
              pipeline=test_pipeline))