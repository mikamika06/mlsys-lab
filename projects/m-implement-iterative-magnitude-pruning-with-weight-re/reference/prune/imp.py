import numpy as np


def apply_mask(weights, mask):
    if isinstance(weights, list):
        return [w * m for w, m in zip(weights, mask)]
    return weights * mask


def magnitude_mask(weights, sparsity):
    if isinstance(weights, list):
        flat = np.concatenate([w.flatten() for w in weights])
        k = int(np.round(sparsity * flat.size))
        if k == 0:
            return [np.ones_like(w) for w in weights]
        if k >= flat.size:
            return [np.zeros_like(w) for w in weights]
        thresh = np.sort(np.abs(flat))[k - 1]
        return [np.where(np.abs(w) <= thresh, 0.0, 1.0) for w in weights]
    else:
        flat = weights.flatten()
        k = int(np.round(sparsity * flat.size))
        if k == 0:
            return np.ones_like(weights)
        if k >= flat.size:
            return np.zeros_like(weights)
        thresh = np.sort(np.abs(flat))[k - 1]
        return np.where(np.abs(weights) <= thresh, 0.0, 1.0)


def iterative_prune(model_fn, train_data, num_rounds, final_sparsity, init_weights):
    current_weights = [w.copy() for w in init_weights]
    masks = [np.ones_like(w) for w in current_weights]
    target_sparsities = [1.0 - (1.0 - final_sparsity) ** (r / num_rounds) for r in range(1, num_rounds + 1)]

    for r in range(num_rounds):
        sp = target_sparsities[r]
        trained_weights = model_fn(current_weights, masks, train_data)
        masks = magnitude_mask(trained_weights, sp)
        current_weights = [iw * m for iw, m in zip(init_weights, masks)]

    return current_weights, masks
