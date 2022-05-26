_base_ = [
    '../../_base_/datasets/flower_bs64_strong.py',
    '../../_base_/schedules/downstream_bs256_adamw.py',
    '../../_base_/default_runtime.py',
]

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=5e-4 * 64 / 512,
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
        num_classes={{_base_.dataset_num_classes}},
        in_channels=768,
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
    ]))

custom_hooks = [
    dict(type='EMAHook', momentum=0.0005)
]

# evaluation = dict(interval=2000, metric='accuracy')
evaluation = dict(interval=2000, metric='per_class_acc')
checkpoint_config = dict(interval=2000, max_keep_ckpts=1)
