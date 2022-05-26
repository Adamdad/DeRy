# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .conformer import Conformer
from .deit import DistilledVisionTransformer
from .lenet import LeNet5
from .mixer import MixerBackbone, MixerFormer
from .mixerv2 import MixerBackbone_CIFARv2, MixerBackbonev2, MixerFormerv2
from .mixerv3 import (MixerBackbone_CIFARv3, MixerBackbonev3,
                      MixerFormer_CIFARv3, MixerFormerv3)
from .mixerv4 import MixerBackbonev4, MixerFormerv4, MixerBackbonev4bn, MixerFormerv4bn
from .mlp_mixer import MlpMixer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3, MobileNetV3_CIFAR
from .regnet import RegNet, RegNet_CIFAR
from .repvgg import RepVGG
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .resnext import ResNeXt, ResNeXt_CIFAR
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .swin_transformer import SwinTransformer
from .t2t_vit import T2T_ViT
from .timm_backbone import TIMMBackbone
from .tnt import TNT
from .torch_backbone import TORCHBackbone
from .vgg import VGG
from .vision_transformer import VisionTransformer

__all__ = [
    'LeNet5', 'AlexNet', 'VGG', 'RegNet', 'ResNet', 'ResNeXt', 'ResNetV1d',
    'ResNeSt', 'ResNet_CIFAR', 'SEResNet', 'SEResNeXt', 'ShuffleNetV1',
    'ShuffleNetV2', 'MobileNetV2', 'MobileNetV3', 'VisionTransformer',
    'SwinTransformer', 'TNT', 'TIMMBackbone', 'T2T_ViT', 'Res2Net', 'RepVGG',
    'Conformer', 'MlpMixer', 'DistilledVisionTransformer', 'MixerBackbone', 'ResNeXt_CIFAR',
    'MixerFormer', 'MixerFormerv2', 'MixerBackbonev2', 'MixerBackbone_CIFARv2',
    'MixerFormerv3', 'MixerBackbonev3', 'MixerBackbone_CIFARv3', 'MixerFormer_CIFARv3',
    'RegNet_CIFAR', 'MobileNetV3_CIFAR', 'TORCHBackbone', 'MixerBackbonev4', 'MixerFormerv4'
]
