_base_ = [
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=128,
)

model = dict(
    type='ImageClassifier',
    train_cfg=dict(model_name='tf_mobilenetv3_small_100'),
    backbone=dict(
        type='TIMMBackbone',
        model_name='tf_mobilenetv3_small_100',
        pretrained=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))