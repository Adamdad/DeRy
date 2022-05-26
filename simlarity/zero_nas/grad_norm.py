import torch


def grad_norm(model, x, target, criterion):
    model.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, target)
    loss.backward()

    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2
        grad_norm = float(torch.sqrt(norm2_sum))

    return grad_norm
