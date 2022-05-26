import numpy as np
import torch


def jacov(model, x, target, criterion):
    jacobs, labels = get_batch_jacobian(model, x, target)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    try:
        score = eval_score(jacobs, labels)
    except Exception as e:
        print(e)
        score = -10e8
    return score


def get_batch_jacobian(model, x, target):
    model.zero_grad()

    x.requires_grad_(True)

    y = model(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))
