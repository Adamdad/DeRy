_base_ = [
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=128,
)

model = dict(
    type='ImageClassifier',
    train_cfg=dict(model_name='resnet18'),
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnet18',
        pretrained=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))