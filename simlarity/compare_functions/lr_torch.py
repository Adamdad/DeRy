import torch


def rearrange_activations(activations):
    batch_size = activations.shape[0]
    flat_activations = activations.view(batch_size, -1)
    return flat_activations


def lr_torch_new(x1, x2):
    x1_flat, x2_flat = rearrange_activations(x1), rearrange_activations(x2)
    q2, r2 = torch.linalg.qr(x2_flat)
    return (torch.linalg.norm(q2.T @ x1_flat)) ** 2 / (torch.linalg.norm(x1_flat)) ** 2

def lr_torch(x1, x2):
    x1_flat, x2_flat = rearrange_activations(x1), rearrange_activations(x2)
    q2, r2 = torch.qr(x2_flat)
    return (torch.norm(q2.T @ x1_flat)) ** 2 / (torch.norm(x1_flat)) ** 2