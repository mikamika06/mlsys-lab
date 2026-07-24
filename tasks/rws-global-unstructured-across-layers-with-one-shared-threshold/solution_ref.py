import numpy as np


def global_unstructured_masks(weights: list, amount: float) -> list:
    sizes = [w.size for w in weights]
    total = sum(sizes)
    flat_abs = np.concatenate([np.abs(np.asarray(w, dtype=np.float64)).ravel() for w in weights])
    k = int(round(amount * total))

    order = np.argsort(flat_abs, kind="stable")  # ascending magnitude
    prune_idx = order[:k]
    flat_mask = np.ones(total, dtype=bool)
    flat_mask[prune_idx] = False

    masks = []
    offset = 0
    for w in weights:
        n = w.size
        masks.append(flat_mask[offset:offset + n].reshape(w.shape))
        offset += n
    return masks
