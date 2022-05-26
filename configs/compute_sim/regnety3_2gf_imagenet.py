_base_ = [
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    train_cfg=dict(model_name='regnet_y_3_2gf'),
    backbone=dict(
        type='TORCHBackbone',
        model_name='regnet_y_3_2gf',
        pretrained=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))