import math
import os
import timm
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.modules.normalization import LayerNorm
from torchvision import models

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
import torch.nn.functional as F


def network_to_module(model_name, block_name, backend):
    if backend == 'timm':
        backbone = timm.create_model(
            model_name, pretrained=True, scriptable=True)
    elif backend == 'pytorch':
        backbone = getattr(models, model_name)(pretrained=True)

    for name, module in backbone.named_modules():
        if name == block_name:
            return module


@BACKBONES.register_module()
class MixerBackbonev2(BaseBackbone):
    """
    """

    def __init__(
        self,
        block_list,
        adapter_list=None,
        block_fixed=True,
        base_channels=64,
        in_channels=3,
        pool_first=True,
        out_indices=(3, ),
        init_cfg=None,
        **kwargs,
    ):
        super(MixerBackbonev2, self).__init__(init_cfg)
        assert isinstance(block_list, list), 'block_list should be a list'
        assert isinstance(adapter_list, list), 'adapter_list should be a list'
        assert len(
            block_list)-1 == len(adapter_list), 'len(block_list)-1 should be len(adapter_list)'
        if pool_first:
            self.base = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(base_channels)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
        else:
            self.base = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(base_channels)),
                ('relu0', nn.ReLU(inplace=True)),
            ]))
        blocks = []
        for model, block_name, backend in block_list:
            block = network_to_module(model, block_name, backend)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        if block_fixed:
            for param in self.blocks.parameters():
                param.requires_grad = False

        if adapter_list is not None:
            adapters = []
            for adapter_cfg in adapter_list:
                adapters.append(NeuralAdapter(**adapter_cfg))
            self.adapters = nn.ModuleList(adapters)
        else:
            self.adapters = None

        self.out_indices = out_indices

    def forward(self, x):
        x = self.base(x)
        outs = []
        for i, block in enumerate(self.blocks):
            if i > 0 and self.adapters is not None:
                x = self.adapters[i-1](x)
            x = block(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class MixerFormerv2(BaseBackbone):
    """
    """

    def __init__(
        self,
        block_list,
        adapter_list=None,
        block_fixed=True,
        all_fixed=False,
        base_channels=64,
        in_channels=3,
        out_indices=(3, ),
        out_resolution=(7, 7),
        out_channels=1024,
        init_cfg=None,
        **kwargs,
    ):
        super(MixerFormerv2, self).__init__(init_cfg)
        assert isinstance(block_list, list), 'block_list should be a list'
        assert isinstance(adapter_list, list), 'adapter_list should be a list'
        assert len(
            block_list)-1 == len(adapter_list), 'len(block_list)-1 should be len(adapter_list)'
        self.base = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3,
                                bias=False)),
            ('norm0', nn.BatchNorm2d(base_channels)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        blocks = []
        for model, block_name, backend in block_list:
            block = network_to_module(model, block_name, backend)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        if adapter_list is not None:
            adapters = []
            for adapter_cfg in adapter_list:
                adapters.append(NeuralAdapter(**adapter_cfg))
            self.adapters = nn.ModuleList(adapters)
        else:
            self.adapters = None

        self.out_indices = out_indices
        self.out_channels = out_channels
        self.out_resolution = out_resolution
        self.norm_layer = nn.LayerNorm(self.out_channels)

        if block_fixed:
            for param in self.blocks.parameters():
                param.requires_grad = False

        if all_fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.base(x)
        outs = []
        for i, block in enumerate(self.blocks):
            if i > 0 and self.adapters is not None:
                x = self.adapters[i-1](x)
            x = block(x)
            if i in self.out_indices:
                out = self.norm_layer(x)
                out = out.view(-1, *self.out_resolution,
                               self.out_channels).permute(0, 3, 1, 2)
                outs.append(out)
        return tuple(outs)


@BACKBONES.register_module()
class MixerBackbone_CIFARv2(MixerBackbonev2):
    def __init__(self,
                 block_list,
                 adapter_list=None,
                 block_fixed=True,
                 base_channels=64,
                 in_channels=3,
                 out_indices=(3, ),
                 init_cfg=None,
                 **kwargs):
        super().__init__(block_list,
                         adapter_list=adapter_list,
                         block_fixed=block_fixed,
                         base_channels=base_channels,
                         in_channels=in_channels,
                         out_indices=out_indices,
                         init_cfg=init_cfg, **kwargs)
        self.base = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1,
                                padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(base_channels)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))


class NeuralAdapter(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 num_fc=0,
                 num_conv=1,
                 mode='cnn2cnn',
                 stride=1) -> None:
        super().__init__()
        assert (num_fc > 0 and num_conv == 0) or (
            num_fc == 0 and num_conv > 0), \
            'num_fc and num_conv can not be both positive.'

        assert mode in ['cnn2cnn', 'cnn2vit', 'vit2cnn', 'vit2vit'], 'mode is not recognized'
        layers = []
        self.mode = mode
        if num_fc > 0:
            layers.append(nn.LayerNorm(input_channel))
            for i in range(num_fc):
                if i == 0:
                    layers.append(
                        nn.Linear(input_channel, output_channel, bias=False))
                else:
                    layers.append(
                        nn.Linear(output_channel, output_channel, bias=False))
            layers.append(nn.LeakyReLU(0.1, inplace=True))

        elif num_conv > 0:
            layers.append(nn.BatchNorm2d(input_channel))
            for i in range(num_conv):
                if i == 0:
                    layers.append(nn.Conv2d(input_channel, output_channel,
                                            kernel_size=stride, stride=stride,
                                            padding=0, bias=False))
                else:
                    layers.append(nn.Conv2d(output_channel, output_channel,
                                            kernel_size=1, stride=1,
                                            padding=0, bias=False))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.adapter = nn.Sequential(*layers)

    def forward(self, x, input_shape=None):
        if self.mode == 'cnn2vit':
            # CNN 2 Vsion Transformer(VIT)
            x = self.adapter(x)
            return x.flatten(2).transpose(1, 2), (x.shape[2], x.shape[3])

        elif self.mode == 'cnn2cnn':
            # CNN 2 CNN
            x = self.adapter(x)
            return x, (x.shape[2], x.shape[3])

        elif self.mode == 'vit2cnn':
            # VIT 2 CNN
            out_channels = x.shape[2]
            token_num = x.shape[1]
            w, h = input_shape
            torch._assert(w*h == token_num,
                          'When VIT to CNN, w x h == token_num')
            
            x = x.view(-1, w, h, out_channels).permute(0, 3, 1, 2)
            x = self.adapter(x)
            return x, (x.shape[2], x.shape[3])

        elif self.mode in 'vit2vit':
            # VIT/Swin 2 VIT/Swin
            return self.adapter(x), input_shape
