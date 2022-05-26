# Blockswap: Fisher-guided block substitution for network compression on a budget
# 
import torch
import torch.nn as nn
import torch.nn.functional as F

import types


def fisher_forward_conv2d(self, x):
    x = F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
    #intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act

def fisher_forward_linear(self, x):
    x = F.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act

def fisher(model, x, target, criterion):

    model.train()
    all_fisher_layers = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            #variables/op needed for fisher computation
            layer.fisher = None
            layer.act = 0.
            layer.dummy = nn.Identity()

            #replace forward method of conv/linear
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(fisher_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(fisher_forward_linear, layer)

            #register backward hook on identity fcn to compute fisher info
            layer.dummy.register_backward_hook(hook_factory(layer))
            all_fisher_layers.append(layer)

    model.zero_grad()
    y = model(x)
    loss = criterion(y, target)
    loss.backward()

    fish = 0
    for layer in all_fisher_layers:
        if layer.fisher is not None:
            fish += torch.sum(torch.abs(layer.fisher.detach()))

    return fish.detach().cpu().numpy()


def hook_factory(layer):
    #function to call during backward pass (hooked on identity op at output of layer)
    def hook_backward(module, grad_input, grad_output):
        act = layer.act.detach()
        grad = grad_output[0].detach()
        if len(act.shape) > 2:
            g_nk = torch.sum((act * grad), list(range(2,len(act.shape))))
        else:
            g_nk = act * grad
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        if layer.fisher is None:
            layer.fisher = del_k
        else:
            layer.fisher += del_k
        del layer.act #without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
    return hook_backward