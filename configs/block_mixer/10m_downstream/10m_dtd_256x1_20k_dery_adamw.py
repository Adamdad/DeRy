_base_ = [
    '../../_base_/datasets/dtd_bs256_strong.py',
    '../../_base_/schedules/downstream_bs256_adamw.py',
    '../../_base_/default_runtime.py',
]

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=5e-4 * 256 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))

fp16 = dict(loss_scale=512.0)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=False,
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
        num_classes={{_base_.dataset_num_classes}},
        in_channels=888,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes={{_base_.dataset_num_classes}},
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.1,
             num_classes={{_base_.dataset_num_classes}}, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0,
             num_classes={{_base_.dataset_num_classes}}, prob=0.5)
    ])
)

custom_hooks = [
    dict(type='EMAHook', momentum=0.0005)
]

evaluation = dict(interval=2000, metric='accuracy')
# evaluation = dict(interval=2000, metric='per_class_acc')
checkpoint_config = dict(interval=2000, max_keep_ckpts=1)

            
