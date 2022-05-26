# Input shape: (3, 224, 224)
# Flops: 4.47 GFLOPs
# Params: 24.89 M
# My Params: 24.893616M

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
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1000,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False))
