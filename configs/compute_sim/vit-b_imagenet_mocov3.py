_base_ = [
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=64,
    val=dict(ann_file=None),
    test=dict(ann_file=None))

model = dict(
    type='ImageClassifier',
    train_cfg=dict(model_name='vit_base_patch16_224',train_strategy='mocov3'),
    backbone=dict(
        type='TIMMBackbone',
        model_name='vit_base_patch16_224',
        prefix='momentum_encoder.',
        checkpoint_path='/Checkpoint/yangxingyi/Pretrained/vit-b-300ep.pth.tar'
        ),
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
