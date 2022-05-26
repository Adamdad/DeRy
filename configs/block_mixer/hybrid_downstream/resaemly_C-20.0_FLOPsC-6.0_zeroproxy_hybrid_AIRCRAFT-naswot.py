model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=False,
        base_channels=64,
        block_list=[['swsl_resnext50_32x4d', 'layer1.0', 'layer1.2', 'mytimm'],
                    [
                        'regnet_y_3_2gf', 'trunk_output.block2.block2-1',
                        'trunk_output.block3.block3-1', 'pytorch'
                    ],
                    [
                        'regnet_y_800mf', 'trunk_output.block3.block3-3',
                        'trunk_output.block3.block3-7', 'pytorch'
                    ],
                    [
                        'vit_tiny_patch16_224', 'blocks.11', 'blocks.11',
                        'mytimm'
                    ]],
        adapter_list=[
            dict(
                input_channel=256,
                output_channel=216,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=576,
                output_channel=320,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=320,
                output_channel=192,
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
        in_channels=192,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1000,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False))
