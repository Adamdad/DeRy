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
# Flops: 4.47 GFLOPs
# Params: 24.89 M
# My Params: 24.893616M

fp16 = dict(loss_scale=512.0)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=True, # False if want to fine-tune all blocks
        base_channels=64,
        block_list=[['swsl_resnext50_32x4d', 'layer1.0', 'layer1.2', 'mytimm'],
                    ['resnet50', 'layer1.2', 'layer3.1', 'pytorch'],
                    ['resnet50', 'layer2.2', 'layer2.2', 'pytorch'],
                    [
                        'swin_small_patch4_window7_224', 'stages.2.blocks.16',
                        'stages.3.blocks.1', 'mmcv'
        ]],
        adapter_list=[
            dict(
                input_channel=256,
                output_channel=256,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=1024,
                output_channel=512,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=512,
                output_channel=384,
                num_fc=0,
                stride=1,
                num_conv=1,
                mode='cnn2vit')
        ],
        out_indices=(3, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
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
    ])
)
