import math
import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from mmcv.cnn.bricks import NORM_LAYERS
from tkinter.messagebox import NO
from torch.nn.modules.normalization import LayerNorm
from torchvision import models

import third_package.timm as mytimm
from ..builder import BACKBONES
try:
    from ..utils.feature_extraction import (create_sub_network,
                                            create_sub_network_transformer)
except:
    pass
from .base_backbone import BaseBackbone
from .mixerv2 import NeuralAdapter


@NORM_LAYERS.register_module('LN2d')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.
    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        # assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
        #     f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)

def network_to_module_subnet(model_name, block_input, block_output, backend, ckp_path=None):
    if backend == 'timm':
        if ckp_path is not None:
            print(f'Loading checkpoint from {ckp_path}')
            backbone = timm.create_model(
                model_name, pretrained=False)
            state_dict = torch.load(ckp_path, map_location='cpu')
            miss_keys = backbone.load_state_dict(state_dict, strict=False)
            print(miss_keys)
        else:
            backbone = timm.create_model(
                model_name, pretrained=True)
    elif backend == 'mytimm':
        if ckp_path is not None:
            print(f'Loading checkpoint from {ckp_path}')
            backbone = mytimm.create_model(
                model_name, pretrained=False)
            state_dict = torch.load(ckp_path, map_location='cpu')
            keys = list(state_dict.keys())
            for key in keys:
                if key in ['head.weight', 'head.bias']:
                    print(f'removing {key}')
                    del state_dict[key]

            miss_keys = backbone.load_state_dict(state_dict, strict=False)
            print(miss_keys)
        else:
            backbone = mytimm.create_model(
                model_name, pretrained=True, scriptable=False)
    elif backend == 'pytorch':
        if ckp_path is not None:
            backbone = getattr(models, model_name)(pretrained=False)
            state_dict = torch.load(ckp_path, map_location='cpu')
            print(f'Loading checkpoint from {ckp_path}')
            miss_keys = backbone.load_state_dict(state_dict, strict=False)
            print(miss_keys)
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
            backbone, block_input, block_output)
    else:
        subnet = create_sub_network(backbone, block_input, block_output)
    return subnet


@BACKBONES.register_module()
class MixerBackbonev4(BaseBackbone):
    """
    """

    def __init__(
        self,
        block_list,
        adapter_list=None,
        base_adapter=None,
        block_fixed=True,
        base_channels=64,
        in_channels=3,
        stem_patch_size = 4,
        pool_first=True,
        out_indices=(3, ),
        init_cfg=None,
        **kwargs,
    ):
        super(MixerBackbonev4, self).__init__(init_cfg)
        assert isinstance(block_list, list), 'block_list should be a list'
        assert isinstance(adapter_list, list), 'adapter_list should be a list'
        assert len(
            block_list)-1 == len(adapter_list), 'len(block_list)-1 should be len(adapter_list)'
        
        self.base = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, base_channels, kernel_size=stem_patch_size, stride=stem_patch_size)),
            ('norm0', LayerNorm2d(base_channels)),
        ]))
     
        blocks = []
        for block_cfg in block_list:
            if len(block_cfg) == 4:
                model, block_input, block_output, backend = block_cfg
                block = network_to_module_subnet(
                    model, block_input, block_output, backend)
            elif len(block_cfg) == 5:
                model, block_input, block_output, backend, ckp_path = block_cfg
                block = network_to_module_subnet(
                    model, block_input, block_output, backend, ckp_path)
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

        if base_adapter is not None:
            self.base_adapter = NeuralAdapter(**base_adapter)
        else:
            self.base_adapter = None

        self.out_indices = out_indices

    def forward(self, x):
        x = self.base(x)
        if self.base_adapter is not None:
            x = self.base_adapter(x)
        outs = []
        for i, block in enumerate(self.blocks):
            if i > 0 and self.adapters is not None:
                x = self.adapters[i-1](x)
            x = block(x)
            if isinstance(x, dict):
                x = list(x.values())[0]
            # print(x.shape)
            if i in self.out_indices:
                outs.append(x)
                
        # exit()
        return tuple(outs)

@BACKBONES.register_module()
class MixerBackbonev4bn(BaseBackbone):
    """
    """

    def __init__(
        self,
        block_list,
        adapter_list=None,
        base_adapter=None,
        block_fixed=True,
        base_channels=64,
        in_channels=3,
        stem_patch_size = 4,
        pool_first=True,
        out_indices=(3, ),
        init_cfg=None,
        **kwargs,
    ):
        super(MixerBackbonev4bn, self).__init__(init_cfg)
        assert isinstance(block_list, list), 'block_list should be a list'
        assert isinstance(adapter_list, list), 'adapter_list should be a list'
        assert len(
            block_list)-1 == len(adapter_list), 'len(block_list)-1 should be len(adapter_list)'
        
        self.base = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, base_channels, kernel_size=stem_patch_size, stride=stem_patch_size)),
            ('norm0', nn.BatchNorm2d(base_channels)),
        ]))
     
        blocks = []
        for block_cfg in block_list:
            if len(block_cfg) == 4:
                model, block_input, block_output, backend = block_cfg
                block = network_to_module_subnet(
                    model, block_input, block_output, backend)
            elif len(block_cfg) == 5:
                model, block_input, block_output, backend, ckp_path = block_cfg
                block = network_to_module_subnet(
                    model, block_input, block_output, backend, ckp_path)
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

        if base_adapter is not None:
            self.base_adapter = NeuralAdapter(**base_adapter)
        else:
            self.base_adapter = None

        self.out_indices = out_indices

    def forward(self, x):
        x = self.base(x)
        if self.base_adapter is not None:
            x = self.base_adapter(x)
        outs = []
        for i, block in enumerate(self.blocks):
            if i > 0 and self.adapters is not None:
                x = self.adapters[i-1](x)
            x = block(x)
            if isinstance(x, dict):
                x = list(x.values())[0]
            # print(x.shape)
            if i in self.out_indices:
                outs.append(x)
                
        # exit()
        return tuple(outs)


@BACKBONES.register_module()
class MixerFormerv4(BaseBackbone):
    """
    """

    def __init__(
        self,
        block_list,
        adapter_list=None,
        base_adapter=None,
        block_fixed=True,
        all_fixed=False,
        stem_patch_size=4,
        base_channels=64,
        in_channels=3,
        out_indices=(3, ),
        hw_ratio=1,
        init_cfg=None,
        **kwargs,
    ):
        super(MixerFormerv4, self).__init__(init_cfg)
        assert isinstance(block_list, list), 'block_list should be a list'
        assert isinstance(adapter_list, list), 'adapter_list should be a list'
        assert len(
            block_list)-1 == len(adapter_list), 'len(block_list)-1 should be len(adapter_list)'
        self.base = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, base_channels, kernel_size=stem_patch_size, stride=stem_patch_size)),
            ('norm0', LayerNorm2d(base_channels)),
        ]))

        blocks = []
        for block_cfg in block_list:
            if len(block_cfg) == 4:
                model, block_input, block_output, backend = block_cfg
                block = network_to_module_subnet(
                    model, block_input, block_output, backend)
            elif len(block_cfg) == 5:
                model, block_input, block_output, backend, ckp_path = block_cfg
                block = network_to_module_subnet(
                    model, block_input, block_output, backend, ckp_path)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

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
        if self.base_adapter is not None:
            x = self.base_adapter(x)
        outs = []
        for i, block in enumerate(self.blocks):
            if i > 0 and self.adapters is not None:
                x = self.adapters[i-1](x)
            x = block(x)
            if isinstance(x, dict):
                x = list(x.values())[0]
            if i in self.out_indices:
                token_num = x.shape[1]
                out_channels = x.shape[2]
                w = int(math.sqrt(token_num * self.hw_ratio))
                h = int(math.sqrt(token_num / self.hw_ratio))
                torch._assert(w*h == token_num,
                              'When VIT to CNN, w x h == token_num')
                out = x.view(-1, w, h,
                               out_channels).permute(0, 3, 1, 2)
                outs.append(out)
        return tuple(outs)

@BACKBONES.register_module()
class MixerFormerv4bn(BaseBackbone):
    """
    """

    def __init__(
        self,
        block_list,
        adapter_list=None,
        base_adapter=None,
        block_fixed=True,
        all_fixed=False,
        stem_patch_size=4,
        base_channels=64,
        in_channels=3,
        out_indices=(3, ),
        hw_ratio=1,
        init_cfg=None,
        **kwargs,
    ):
        super(MixerFormerv4bn, self).__init__(init_cfg)
        assert isinstance(block_list, list), 'block_list should be a list'
        assert isinstance(adapter_list, list), 'adapter_list should be a list'
        assert len(
            block_list)-1 == len(adapter_list), 'len(block_list)-1 should be len(adapter_list)'
        self.base = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, base_channels, kernel_size=stem_patch_size, stride=stem_patch_size)),
            ('norm0', nn.BatchNorm2d(base_channels)),
        ]))

        blocks = []
        for block_cfg in block_list:
            if len(block_cfg) == 4:
                model, block_input, block_output, backend = block_cfg
                block = network_to_module_subnet(
                    model, block_input, block_output, backend)
            elif len(block_cfg) == 5:
                model, block_input, block_output, backend, ckp_path = block_cfg
                block = network_to_module_subnet(
                    model, block_input, block_output, backend, ckp_path)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

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
        if self.base_adapter is not None:
            x = self.base_adapter(x)
        outs = []
        for i, block in enumerate(self.blocks):
            if i > 0 and self.adapters is not None:
                x = self.adapters[i-1](x)
            x = block(x)
            if isinstance(x, dict):
                x = list(x.values())[0]
            if i in self.out_indices:
                token_num = x.shape[1]
                out_channels = x.shape[2]
                w = int(math.sqrt(token_num * self.hw_ratio))
                h = int(math.sqrt(token_num / self.hw_ratio))
                torch._assert(w*h == token_num,
                              'When VIT to CNN, w x h == token_num')
                out = x.view(-1, w, h,
                               out_channels).permute(0, 3, 1, 2)
                outs.append(out)
        return tuple(outs)
