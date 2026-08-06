import numpy as np


def regrow_masks(w, g, mask, k, seed):
    """Two dynamic-sparse-training regrowth strategies, same regrow count."""
    mask = np.asarray(mask, dtype=bool)
    g = np.asarray(g, dtype=np.float64)
    shape = mask.shape

    flat_mask = mask.reshape(-1)
    flat_g = g.reshape(-1)

    zero_idx = []
    flat_abs_g = []
    for i in range(flat_mask.shape[0]):
        val_g = flat_g[i]
        abs_g = val_g if val_g >= 0.0 else -val_g
        flat_abs_g.append(abs_g)
        if not flat_mask[i]:
            zero_idx.append(i)

    pairs = []
    for idx in zero_idx:
        pairs.append((idx, flat_abs_g[idx]))

    sorted_pairs = sorted(pairs, key=lambda x: (-x[1], x[0]))
    order_indices = [p[0] for p in sorted_pairs]
    rigl_pick = order_indices[:k]

    rigl_mask = flat_mask.copy()
    for idx in rigl_pick:
        rigl_mask[idx] = True

    rng = np.random.default_rng(seed)
    set_pick = rng.choice(zero_idx, size=k, replace=False)
    set_mask = flat_mask.copy()
    for idx in set_pick:
        set_mask[idx] = True

    return rigl_mask.reshape(shape), set_mask.reshape(shape)
