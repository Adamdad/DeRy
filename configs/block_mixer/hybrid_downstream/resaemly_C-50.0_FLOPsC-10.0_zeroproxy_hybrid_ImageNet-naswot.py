# Input shape: (3, 224, 224)
# Flops: 6.43 GFLOPs
# Params: 40.41 M
# My Params: 40.414054M
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=False,
        base_channels=64,
        block_list=[['resnet50', 'layer1.0', 'layer1.1', 'pytorch'],
                    [
                        'regnet_y_3_2gf', 'trunk_output.block2.block2-1',
                        'trunk_output.block3.block3-1', 'pytorch'
                    ], ['resnet101', 'layer3.8', 'layer3.22', 'pytorch'],
                    ['resnet50', 'layer3.4', 'layer4.2', 'pytorch']],
        adapter_list=[
            dict(
                input_channel=256,
                output_channel=216,
                stride=2,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=576,
                output_channel=1024,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=1024,
                output_channel=1024,
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
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1000,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False))
