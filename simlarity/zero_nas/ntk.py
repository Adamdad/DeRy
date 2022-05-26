import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def empirical_ntk(net: nn.Module, batch_input: torch.Tensor) -> torch.Tensor:
    """
    :param input_batch: batch of shape (B, I) where B is the
    number of inputs points and I is the input shape to the network \n
    :param net: nn.Module, output shape of net is supposed to be 1 here \n
    :output: empirical NTK Gram matrix of shape I x I
    """
    # assert len(batch_input.shape) == 2, "in this function, batch is of dim 2"

    batch_output = net(batch_input)
    # assert len(batch_output.shape) == 2, "output should be of dim 2"
    # assert batch_output.shape[1] == 1, "output of the network should be a scalar in this function"

    # We compute each gradient
    gradient_list = []
    for b in range(batch_input.shape[0]):
        net.zero_grad()
        batch_output[b].backward(torch.ones_like(batch_output[b]), retain_graph=True) # 
        gradient = torch.cat([p.grad.flatten() for p in net.parameters()])
        gradient_list.append(gradient)

    with torch.no_grad():
        gradient_tensor = torch.stack(gradient_list)

        return torch.einsum('ij, jk->ik', gradient_tensor, gradient_tensor.T)


def nasi(model, x, target, criterion):
    ntk = empirical_ntk(model, x)
    eigenvalues, _ = torch.symeig(ntk)  # ascending
    return np.sqrt(eigenvalues.cpu().numpy().sum() / x.shape[0])
