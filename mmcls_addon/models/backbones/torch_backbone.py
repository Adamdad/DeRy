import torch

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from torchvision import models

@BACKBONES.register_module()
class TORCHBackbone(BaseBackbone):
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
        pretrained=False,
        checkpoint_path='',
        in_channels=3,
        freeze_model=False,
        init_cfg=None,
        **kwargs,
    ):
        super(TORCHBackbone, self).__init__(init_cfg)
        self.model_name = model_name
        self.torch_model = getattr(models, model_name)(pretrained=pretrained)

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

    def forward(self, x):
        if self.model_name.startswith('eff'):
            features = self.torch_model.features(x)
            return (features, )
        elif self.model_name.startswith('regnet'):
            x = self.torch_model.stem(x)
            x = self.torch_model.trunk_output(x)
            return (x, )