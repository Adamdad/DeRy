# Copyright (c) OpenMMLab. All rights reserved.
try:
    import timm
except ImportError:
    timm = None
import logging
import os
import re
import torch
from collections import OrderedDict
from timm.utils.model import freeze, unfreeze

import third_package.timm as mytimm
from ..builder import BACKBONES
from .base_backbone import BaseBackbone

_logger = logging.getLogger(__name__)


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        if state_dict_key:
            state_dict = checkpoint[state_dict_key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(
            state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, prefix=None, use_ema=False, remove_keys=None, strict=False):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
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
    if remove_keys is not None:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k not in remove_keys:
                new_state_dict[k] = v
        state_dict = new_state_dict
    keys = model.load_state_dict(state_dict, strict=strict)
    return keys


@BACKBONES.register_module()
class TIMMBackbone(BaseBackbone):
    """Wrapper to use backbones from timm library. More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_ .

    Args:
        model_name (str): Name of timm model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        checkpoint_path (str): Path of checkpoint to load after
            model is initialized.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(
        self,
        model_name,
        remove_keys=None,
        pretrained=False,
        prefix=None,
        checkpoint_path='',
        in_channels=3,
        freeze_model=False,
        init_cfg=None,
        **kwargs,
    ):
        if timm is None:
            raise RuntimeError('timm is not installed')
        super(TIMMBackbone, self).__init__(init_cfg)
        self.timm_model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            # checkpoint_path=checkpoint_path,
            **kwargs,
        )
        if checkpoint_path:
            mis_keys = load_checkpoint(self.timm_model, checkpoint_path=checkpoint_path,
                                       prefix=prefix, remove_keys=remove_keys, strict=False)
            print('Miss Keys', mis_keys)

        # reset classifier
        self.timm_model.reset_classifier(0, '')

        if freeze_model:
            freeze(self.timm_model)
        else:
            pass

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

    def forward(self, x):
        features = self.timm_model.forward_features(x)
        return (features, )
