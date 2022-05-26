# Input shape: (3, 224, 224)
# Flops: 2.99 GFLOPs
# Params: 7.83 M
# My Params: 7.834434M

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
        num_classes=1000,
        in_channels=888,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1000,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False))
