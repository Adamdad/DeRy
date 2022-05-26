_base_ = [
    '../../_base_/datasets/imagenet_bs64_swin_224.py',
    '../../_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    '../../_base_/default_runtime.py',
]

fp16 = dict(loss_scale=512.0)
dist_params = dict(backend='nccl')
# Schedule settings
runner = dict(max_epochs=100)
optimizer = dict(paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    val=dict(ann_file=None),
    test=dict(ann_file=None))
evaluation = dict(interval=10, metric='accuracy')
checkpoint_config = dict(interval=5, max_keep_ckpts=2)

# Input shape: (3, 224, 224)
# Flops: 6.43 GFLOPs
# Params: 40.41 M
# My Params: 40.414054M

fp16 = dict(loss_scale=512.0)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=True,
        base_channels=64, # False if want to fine-tune all blocks
        block_list=[[
            'resnet50', 'layer1.0', 'layer1.1', 'pytorch'
        ],
            [
            'regnet_y_3_2gf', 'trunk_output.block2.block2-1',
            'trunk_output.block3.block3-1', 'pytorch'
        ],
            [
            'resnet101', 'layer3.8', 'layer3.22', 'pytorch'
        ],
            [
            'resnet50', 'layer3.4', 'layer4.2', 'pytorch'
        ]],
        adapter_list=[
            dict(
                input_channel=256,
                output_channel=216,
                stride=2,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=576,
                output_channel=1024,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=1024,
                output_channel=1024,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn')
        ],
        out_indices=(3, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1000,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.1,
             num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0,
             num_classes=1000, prob=0.5)
    ]))
