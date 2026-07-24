import numpy as np


def compare_sampling(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    losses = coeffs[:, 0] + coeffs[:, 1] + coeffs[:, 2]

    uniform_loss = float(np.mean(losses))

    z = -losses
    z = z - np.max(z)
    weights = np.exp(z)
    weights = weights / np.sum(weights)

    reweighted_loss = float(np.sum(weights * losses))
    reduction = float(uniform_loss - reweighted_loss)

    return uniform_loss, reweighted_loss, reduction
