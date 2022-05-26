_base_ = [
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=128,
    val=dict(ann_file=None),
    test=dict(ann_file=None))

model = dict(
    type='ImageClassifier',
    train_cfg=dict(model_name='resnet50', train_strategy='inat2021'),
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnet50',
        remove_keys=['fc.weight', 'fc.bias'],
        checkpoint_path='/Checkpoint/yangxingyi/Pretrained/inat2021_supervised_large.pth.tar'
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
