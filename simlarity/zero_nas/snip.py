import torch


def snip(model, x, target, criterion):
    model.zero_grad()
    y = model(x)
    loss = criterion(y, target)
    loss.backward()
    grads = [
        p.grad.detach().clone().abs()
        for p in model.parameters()
        if p.grad is not None
    ]

    with torch.no_grad():
        saliences = [
            (grad * weight).view(-1).abs()
            for weight, grad in zip(model.parameters(), grads)
        ]
        score = torch.sum(torch.cat(saliences)).cpu().numpy()
    return score
