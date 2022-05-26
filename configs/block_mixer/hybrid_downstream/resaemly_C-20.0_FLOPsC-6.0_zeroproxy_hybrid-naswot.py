model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=False,
        base_channels=64,
        block_list=[
            ['swsl_resnext50_32x4d', 'layer1.0', 'layer1.2', 'mytimm'],
            ['resnet50', 'layer1.2', 'layer2.1', 'pytorch'],
            ['vit_tiny_patch16_224', 'blocks.2', 'blocks.10', 'mytimm'],
            ['vit_tiny_patch16_224', 'blocks.11', 'blocks.11', 'mytimm']
        ],
        adapter_list=[
            dict(
                input_channel=256,
                output_channel=256,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=512,
                output_channel=192,
                num_fc=0,
                stride=1,
                num_conv=1,
                mode='cnn2vit'),
            dict(
                input_channel=192,
                output_channel=192,
                num_fc=1,
                num_conv=0,
                mode='vit2vit')
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