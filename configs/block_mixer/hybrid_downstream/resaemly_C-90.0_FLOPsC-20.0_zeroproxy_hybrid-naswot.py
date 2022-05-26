# Input shape: (3, 224, 224)
# Flops: 49.72 GFLOPs
# Params: 84.16 M
# My Params: 84.16068M
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=False,
        base_channels=64,
        block_list=[['resnet50', 'layer1.0', 'layer2.1', 'pytorch'],
                    [
                        'swin_base_patch4_window7_224', 'stages.2.blocks.0',
                        'stages.2.blocks.8', 'mmcv'
                    ],
                    [
                        'vit_tiny_patch16_224', 'blocks.2', 'blocks.10',
                        'mytimm'
                    ],
                    [
                        'swin_base_patch4_window7_224', 'stages.2.blocks.11',
                        'stages.3.blocks.1', 'mmcv'
                    ]],
        adapter_list=[
            dict(
                input_channel=512,
                output_channel=512,
                num_fc=0,
                stride=1,
                num_conv=1,
                mode='cnn2vit'),
            dict(
                input_channel=512,
                output_channel=192,
                num_fc=1,
                num_conv=0,
                mode='vit2vit'),
            dict(
                input_channel=192,
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
