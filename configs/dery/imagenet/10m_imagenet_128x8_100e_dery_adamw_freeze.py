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
# Flops: 2.99 GFLOPs
# Params: 7.83 M
# My Params: 7.834434M

fp16 = dict(loss_scale=512.0)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=True, # False if want to fine-tune all blocks
        base_channels=32,
        block_list=[[
            'regnet_y_16gf', 'trunk_output.block1.block1-0',
            'trunk_output.block1.block1-1', 'pytorch'
        ], ['mobilenetv3_large_100', 'blocks.2.2', 'blocks.2.2', 'mytimm'],
            [
            'regnet_y_800mf', 'trunk_output.block3.block3-3',
            'trunk_output.block3.block3-7', 'pytorch'
        ],
            [
            'regnet_y_1_6gf', 'trunk_output.block3.block3-15',
            'trunk_output.block4.block4-1', 'pytorch'
        ]],
        adapter_list=[
            dict(
                input_channel=224,
                output_channel=40,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=40,
                output_channel=320,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=320,
                output_channel=336,
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
        in_channels=888,
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

