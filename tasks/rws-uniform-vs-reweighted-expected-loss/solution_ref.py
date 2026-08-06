import math
import numpy as np


def compare_sampling(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n = coeffs.shape[0]
    
    losses = np.empty(n, dtype=np.float64)
    for i in range(n):
        losses[i] = coeffs[i, 0] + coeffs[i, 1] + coeffs[i, 2]

    loss_sum = 0.0
    for i in range(n):
        loss_sum += losses[i]
    uniform_loss = float(loss_sum / n)

    z = np.empty(n, dtype=np.float64)
    for i in range(n):
        z[i] = -losses[i]

    max_z = z[0]
    for i in range(1, n):
        if z[i] > max_z:
            max_z = z[i]

    weights = np.empty(n, dtype=np.float64)
    for i in range(n):
        weights[i] = math.exp(z[i] - max_z)

    weight_sum = 0.0
    for i in range(n):
        weight_sum += weights[i]

    for i in range(n):
        weights[i] = weights[i] / weight_sum

    reweighted_loss_acc = 0.0
    for i in range(n):
        reweighted_loss_acc += weights[i] * losses[i]
    reweighted_loss = float(reweighted_loss_acc)

    reduction = float(uniform_loss - reweighted_loss)

    return uniform_loss, reweighted_loss, reduction
