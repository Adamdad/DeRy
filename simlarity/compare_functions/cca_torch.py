import torch


def rearrange_activations(activations):
    batch_size = activations.shape[0]
    flat_activations = activations.view(batch_size, -1)
    return flat_activations


def cca_torch(x1, x2):
    x1_flat, x2_flat = rearrange_activations(x1), rearrange_activations(x2)

    q1, _ = torch.qr(x1_flat)
    q2, _ = torch.qr(x2_flat)

    return (torch.norm(q2.T @ q1)) ** 2 / q1.shape[1]