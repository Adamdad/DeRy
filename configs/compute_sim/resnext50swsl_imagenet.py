_base_ = [
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    train_cfg=dict(model_name='swsl_resnext50_32x4d'),
    backbone=dict(
        type='TIMMBackbone',
        model_name='swsl_resnext50_32x4d',
        pretrained=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))