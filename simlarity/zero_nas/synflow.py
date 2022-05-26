
from operator import mod
import torch

def synflow(model, x, target, criterion):

    device = x.device
    # convert params to their abs. Keep sign for converting it back.
    # keep signs of all params
    signs = linearize(model)
    
    # Compute gradients with input of 1s 
    model.zero_grad()
    model.double()
    input_dim = list(x[0,:].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = model(inputs)
    torch.sum(output).backward() 

    with torch.no_grad():
        saliences = []
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                score = torch.clone(p.grad * p).detach().abs_().view(-1)
                saliences.append(score)
        score = torch.sum(torch.cat(saliences)).cpu().numpy()


    # apply signs of all params
    nonlinearize(model, signs)

    return score

@torch.no_grad()
def linearize(net):
    signs = {}
    for name, param in net.state_dict().items():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs

#convert to orig values
@torch.no_grad()
def nonlinearize(net, signs):
    for name, param in net.state_dict().items():
        # if 'weight_mask' not in name:
        param.mul_(signs[name])
    net.float()

# select the gradients that we want to use for search/prune
def synflow_weight(layer):
    if layer.weight.grad is not None:
        return torch.abs(layer.weight * layer.weight.grad)
    else:
        return torch.zeros_like(layer.weight)