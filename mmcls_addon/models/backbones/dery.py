import math
import os
import timm
import torch
import torch.nn as nn
from collections import OrderedDict
from mmcv.runner import load_checkpoint

from torchvision import models

import third_package.timm as mytimm
from mmcls.models.builder import BACKBONES, build_backbone

try:
    from ..utils.feature_extraction import (create_sub_network,
                                            create_sub_network_transformer)
except:
    pass
import mmcv

from blocklize import MODEL_STATS
from .base_backbone import BaseBackbone


def network_to_module_subnet(model_name, block_input, block_output, backend, prefix=None, ckp_path=None):
    # print(model_name, block_input, block_output, backend)
    if backend == 'timm':
        if ckp_path is not None:
            backbone = timm.create_model(
                model_name, pretrained=False, scriptable=True)
            if os.path.isfile(ckp_path):
                state_dict = torch.load(ckp_path, map_location='cpu')
                print(f'Loading checkpoint from {ckp_path}')
                miss_keys = backbone.load_state_dict(state_dict, strict=False)
                print(miss_keys)
            else:
                print(f'{ckp_path} does not exists')
        else:
            backbone = timm.create_model(
                model_name, pretrained=True, scriptable=True)
    elif backend == 'mytimm':
        if ckp_path is not None:
            backbone = mytimm.create_model(
                model_name, pretrained=False, scriptable=True)
            if os.path.isfile(ckp_path) and ckp_path.endswith('pth'):
                print(f'Loading checkpoint from {ckp_path}')
                state_dict = torch.load(ckp_path, map_location='cpu')
                if 'state_dict' in state_dict.keys():
                    state_dict = state_dict['state_dict']
                keys = list(state_dict.keys())
                for key in keys:
                    if key in ['head.weight', 'head.bias', 'fc.weight', 'fc.bias']:
                        print(f'removing {key}')
                        del state_dict[key]

                if prefix is not None:
                    new_state_dict = OrderedDict()
                    if prefix.endswith('.'):
                        pass
                    else:
                        prefix += '.'
                    for k, v in state_dict.items():
                        # strip `module.` prefix
                        name = k[len(prefix):] if k.startswith(prefix) else k
                        new_state_dict[name] = v
                    state_dict = new_state_dict

                miss_keys = backbone.load_state_dict(state_dict, strict=False)
                print(miss_keys)
            elif os.path.isfile(ckp_path) and ckp_path.endswith('npz'):
                mytimm.models.vision_transformer._load_weights(
                    backbone, ckp_path)
            else:
                print(f'{ckp_path} does not exists')
        else:
            backbone = mytimm.create_model(
                model_name, pretrained=True, scriptable=False)
    elif backend == 'mmcv':
        config = MODEL_STATS[model_name]['cfg']
        cfg = mmcv.Config.fromfile(config)
        backbone = build_backbone(cfg.model.backbone)
        if ckp_path is not None:
            if os.path.isfile(ckp_path):
                print(f'Loading checkpoint from {ckp_path}')
                load_checkpoint(backbone, ckp_path, revise_keys=[(r'^module\.backbone\.', ''),
                                                                 (r'^backbone\.', '')])
            else:
                print(f'{ckp_path} does not exists')
        else:
            ckp_path = MODEL_STATS[model_name]['load_from']
            load_checkpoint(backbone, ckp_path, revise_keys=[(r'^module\.backbone\.', ''),
                                                             (r'^backbone\.', '')])

    elif backend == 'pytorch':
        if ckp_path is not None:
            backbone = getattr(models, model_name)(pretrained=False)
            if os.path.isfile(ckp_path):
                state_dict = torch.load(ckp_path, map_location='cpu')
                print(f'Loading checkpoint from {ckp_path}')
                miss_keys = backbone.load_state_dict(state_dict, strict=False)
                print(miss_keys)
            else:
                print(f'{ckp_path} does not exists')
        else:
            backbone = getattr(models, model_name)(pretrained=True)

    if isinstance(block_input, str):
        block_input = [block_input]
    elif isinstance(block_input, tuple):
        block_input = list(block_input)
    elif isinstance(block_input, list):
        block_input = block_input
    else:
        TypeError('Block input should be a string or tuple or list')

    if isinstance(block_output, str):
        block_output = [block_output]
    elif isinstance(block_output, tuple):
        block_output = list(block_output)
    elif isinstance(block_output, list):
        block_output = block_output
    else:
        TypeError('Block output should be a string or tuple or list')

    if model_name.startswith('swin_') or model_name.startswith('vit'):
        subnet = create_sub_network_transformer(
            backbone, model_name, block_input, block_output)
    else:
        subnet = create_sub_network(backbone, block_input, block_output)
    return subnet


@BACKBONES.register_module()
class DeRy(BaseBackbone):
    """
    """

    def __init__(
        self,
        block_list,
        adapter_list=None,
        base_adapter=None,
        block_fixed=True,
        all_fixed=False,
        base_channels=64,
        in_channels=3,
        out_indices=(3, ),
        hw_ratio=1,
        init_cfg=None,
        **kwargs,
    ):
        super(DeRy, self).__init__(init_cfg)
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
        block_types = []
        for block_cfg in block_list:
            if isinstance(block_cfg, dict):
                model = block_cfg['model_name']
                block = network_to_module_subnet(**block_cfg)
            elif isinstance(block_cfg, list) and len(block_cfg) == 4:
                model, block_input, block_output, backend = block_cfg
                block = network_to_module_subnet(
                    model, block_input, block_output, backend)
            elif isinstance(block_cfg, list) and len(block_cfg) == 5:
                model, block_input, block_output, backend, ckp_path = block_cfg
                block = network_to_module_subnet(
                    model, block_input, block_output, backend, ckp_path=ckp_path)
            else:
                AssertionError('block_cfg type not supported')
            block_type = MODEL_STATS[model]['type']
            block_types.append(block_type)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.block_types = block_types

        if adapter_list is not None:
            adapters = []
            for adapter_cfg in adapter_list:
                adapters.append(NeuralAdapter(**adapter_cfg))
            self.adapters = nn.ModuleList(adapters)
        else:
            self.adapters = None

        if base_adapter is not None:
            self.base_adapter = NeuralAdapter(**base_adapter)
        else:
            self.base_adapter = None

        self.out_indices = out_indices
        self.hw_ratio = hw_ratio

        if block_fixed:
            for param in self.blocks.parameters():
                param.requires_grad = False

        if all_fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.base(x)
        out_shape = (x.shape[2], x.shape[3])
        if self.base_adapter is not None:
            x, out_shape = self.base_adapter(x, out_shape)
        outs = []
        for i, block in enumerate(self.blocks):
            if i > 0 and self.adapters is not None:

                x, out_shape = self.adapters[i-1](x, out_shape)

            # print(x.shape)
            # print(out_shape)
            if self.block_types[i] == 'swin':
                x, out_shape = block(x, out_shape)
            elif self.block_types[i] == 'cnn':
                x = block(x)
                if isinstance(x, dict):
                    x = list(x.values())[0]
                out_shape = (x.shape[2], x.shape[3])
            elif self.block_types[i] == 'vit':
                x = block(x)

            if i in self.out_indices:
                out = x
                if self.block_types[i] == 'cnn':
                    outs.append(out)
                else:
                    token_num = out.shape[1]
                    out_channels = out.shape[2]
                    w, h = out_shape
                    
                    torch._assert(w*h == token_num,
                                  'When VIT to CNN, w x h == token_num')
                    out = out.view(-1, w, h,
                                   out_channels).permute(0, 3, 1, 2)
                    outs.append(out)
        return tuple(outs)


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