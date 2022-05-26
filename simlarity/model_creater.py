import math
from utils import Block
from mmcls.models import build_classifier
from torch import nn
import numpy as np


class Model_wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.extract_feat(x)
        if isinstance(x, tuple):
            x = x[-1]
        return self.model.head.fc(x)


class Model_Creator(object):

    def create_hybrid(self, blocks):
        for b in blocks:
            if not isinstance(b, Block):
                return None

        blocks = list(sorted(blocks, key=lambda x: x.block_index))
        if len(blocks[0].in_size) == 2:
            return None
        model_cfg = self.create_hybrid_cfg(blocks)

        model = build_classifier(model_cfg)
        return Model_wrapper(model)

    def create_cnn(self, blocks):
        for b in blocks:
            if not isinstance(b, Block):
                return None
        blocks = list(sorted(blocks, key=lambda x: x.block_index))
        model_cfg = self.create_cnn_cfg(blocks)

        model = build_classifier(model_cfg)
        return Model_wrapper(model)

    def create_hybrid_noshuffle(self, blocks):
        for b in blocks:
            if not isinstance(b, Block):
                return None

        model_cfg = self.create_hybrid_cfg(blocks)

        model = build_classifier(model_cfg)
        return Model_wrapper(model)

    def create_cnn_cfg(self, blocks):

        model_cfg = dict(
            type='ImageClassifier',
            backbone=dict(
                type='MixerBackbonev3',
                block_fixed=False,
                base_channels=blocks[0].in_size[0],
                block_list=[b.print_split() for b in blocks],
                adapter_list=[
                    dict(input_channel=blocks[bid].out_size[0],
                         output_channel=blocks[bid+1].in_size[0],
                         stride=1 if blocks[bid].out_size[1] /
                         blocks[bid+1].in_size[1] < 2 else 2,
                         num_fc=0,
                         num_conv=1) for bid in range(len(blocks)-1)
                ],
                out_indices=(3, )),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=blocks[-1].out_size[0],
                loss=dict(
                    type='LabelSmoothLoss',
                    label_smooth_val=0.1,
                    num_classes=1000,
                    reduction='mean',
                    loss_weight=1.0),
                topk=(1, 5),
                cal_acc=False),
        )

        return model_cfg

    def create_hybrid_cfg(self, blocks):
        adapt_list = []
        # print(downs)
        for bid in range(len(blocks)-1):
            stride = np.random.choice([1,2], 1, p=[0.9, 0.1])[0]
            
            if len(blocks[bid].out_size) == 3 and len(blocks[bid+1].in_size) == 3:
                # CNN 2 CNN
                adapt = dict(input_channel=blocks[bid].out_size[0],
                             output_channel=blocks[bid+1].in_size[0],
                             stride=stride,
                             num_fc=0,
                             num_conv=1,
                             mode='cnn2cnn')
            elif len(blocks[bid].out_size) == 3 and len(blocks[bid+1].in_size) == 2:
                # CNN 2 Transformer
                adapt = dict(input_channel=blocks[bid].out_size[0],
                             output_channel=blocks[bid+1].in_size[1],
                             num_fc=0,
                             stride=stride,
                             num_conv=1,
                             mode='cnn2vit')

            elif len(blocks[bid].out_size) == 2 and len(blocks[bid+1].in_size) == 2:
                # Transformer 2 Transformer
                adapt = dict(input_channel=blocks[bid].out_size[1],
                             output_channel=blocks[bid+1].in_size[1],
                             num_fc=1,
                             num_conv=0,
                             mode='vit2vit')

            elif len(blocks[bid].out_size) == 2 and len(blocks[bid+1].in_size) == 3:
                # Transformer 2 CNN
                adapt = dict(input_channel=blocks[bid].out_size[1],
                             output_channel=blocks[bid+1].in_size[0],
                             num_fc=0,
                             num_conv=1,
                             mode='vit2cnn')

            adapt_list.append(adapt)
        
        model_cfg = dict(
            type='ImageClassifier',
            backbone=dict(
                type='MixerFormerv3',
                block_fixed=False,
                base_channels=blocks[0].in_size[0],
                block_list=[b.print_split() for b in blocks],
                adapter_list=adapt_list,
                out_indices=(3, )),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=blocks[-1].out_size[1]
                if adapt_list[-1]['mode'].endswith('vit') else blocks[-1].out_size[0],
                loss=dict(
                    type='LabelSmoothLoss',
                    label_smooth_val=0.1,
                    num_classes=1000,
                    reduction='mean',
                    loss_weight=1.0),
                topk=(1, 5),
                cal_acc=False),
        )

        return model_cfg
