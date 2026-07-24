import numpy as np


def rigl_topology_update(w, grad, mask, update_fraction):
    out = np.asarray(mask, dtype=np.int64).copy()
    live = int(np.sum(out))
    k = int(np.floor(update_fraction * live))

    if k == 0:
        return out

    active = np.flatnonzero(out == 1)
    drop_order = sorted(active.tolist(), key=lambda i: (abs(float(w[i])), i))
    out[drop_order[:k]] = 0

    candidates = np.flatnonzero(out == 0)
    grow_order = sorted(
        candidates.tolist(),
        key=lambda i: (-abs(float(grad[i])), i),
    )
    out[grow_order[:k]] = 1

    return out
