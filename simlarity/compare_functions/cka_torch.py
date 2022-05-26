import torch

def cka_linear_torch(x1, x2):
    x1 = gram_linear(rearrange_activations(x1))
    x2 = gram_linear(rearrange_activations(x2))
    similarity = _cka(x1, x2)
    return similarity

def cka_rbf_torch(x1, x2, threshold=1.0):
    x1 = gram_rbf(rearrange_activations(x1), threshold=threshold)
    x2 = gram_rbf(rearrange_activations(x2), threshold=threshold)
    similarity = _cka(x1, x2)
    return similarity


def rearrange_activations(activations):
    batch_size = activations.shape[0]
    flat_activations = activations.view(batch_size, -1)
    return flat_activations


def gram_linear(x):
    return torch.mm(x, x.T)

def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
        x: A num_examples x num_features matrix of features.
        threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.mm(x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    if not torch.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')

    if unbiased:
        pass
        # TODO
    else:
        means = torch.mean(gram, dim=0, dtype=torch.float64) # , dtype=torch.float64
        means -= torch.mean(means) / 2
        gram -= torch.unsqueeze(means, len(means.shape))
        gram -= torch.unsqueeze(means, 0)

    return gram


def _cka(gram_x, gram_y, debiased=False):
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    scaled_hsic = (gram_x.view(-1) * gram_y.view(-1)).sum()
    # normalization_x = torch.linalg.norm(gram_x)
    # normalization_y = torch.linalg.norm(gram_y)
    normalization_x = torch.norm(gram_x)
    normalization_y = torch.norm(gram_y)

    return scaled_hsic / (normalization_x * normalization_y)