# Input shape: (3, 224, 224)
# Flops: 13.29 GFLOPs
# Params: 80.66 M
# My Params: 80.661279M
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=False,
        base_channels=64,
        block_list=[['swsl_resnext50_32x4d', 'layer1.0', 'layer1.2', 'mytimm'],
                    [
                        'swin_small_patch4_window7_224', 'stages.0.blocks.1',
                        'stages.0.blocks.1', 'mmcv'
                    ],
                    [
                        'swin_small_patch4_window7_224', 'stages.0.downsample',
                        'stages.2.blocks.15', 'mmcv'
                    ],
                    [
                        'swin_base_patch4_window7_224', 'stages.2.blocks.11',
                        'stages.3.blocks.1', 'mmcv'
                    ]],
        adapter_list=[
            dict(
                input_channel=256,
                output_channel=96,
                num_fc=0,
                stride=1,
                num_conv=1,
                mode='cnn2vit'),
            dict(
                input_channel=96,
                output_channel=96,
                num_fc=1,
                num_conv=0,
                mode='vit2vit'),
            dict(
                input_channel=384,
                output_channel=512,
                num_fc=1,
                num_conv=0,
                mode='vit2vit')
        ],
        out_indices=(3, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1000,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False))
