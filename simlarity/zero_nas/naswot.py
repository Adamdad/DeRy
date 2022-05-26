from turtle import pen
import torch
import numpy as np
from torch import nn
from mmcls.models.utils.attention import WindowMSA, MultiheadAttention, ShiftWindowMSA
from third_package.timm.models.vision_transformer import Attention


def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return ld


def get_batch_jacobian(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, y.detach()


def naswot(model, x, target, criterion):
    batch_size = x.shape[0]

    model.K = np.zeros((batch_size, batch_size))

    def counting_forward_hook(module, inp, out):
        inp = out
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            if isinstance(inp, tuple):
                inp = inp[0]
            assert isinstance(inp, torch.Tensor)
            try:
                inp = inp.view(batch_size, -1)
            except:
                inp = inp.reshape(batch_size, -1)
            x = (inp > 0).float()
            K = x @ x.t()
            # print(x.shape, K.shape, model.K.shape, module)
            K2 = (1. - x) @ (1. - x.t())
            model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
        except Exception as err:
            
            print('---- error on model : ')
            print(model)
            raise err

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    for name, module in model.named_modules():
        if (isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU) or
                isinstance(module, nn.GELU) or isinstance(module, nn.PReLU) or
                isinstance(module, nn.Hardswish) or isinstance(module, nn.ELU) or
                isinstance(module, nn.SELU) or isinstance(module, nn.Mish) or 
                isinstance(module, nn.Linear) or 
                isinstance(module, Attention) or
                isinstance(module, nn.Dropout) or #isinstance(module, nn.Conv2d) or
                # isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm) or
                isinstance(module, ShiftWindowMSA) or isinstance(module, MultiheadAttention)
                ):
            module.visited_backwards = True
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)

    _ = model.forward(x)

    score = logdet(model.K)

    return float(score)/batch_size
