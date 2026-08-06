import math
import numpy as np


def rigl_topology_update(w, grad, mask, update_fraction):
    out = np.asarray(mask, dtype=np.int64).copy()
    live = 0
    for i in range(out.shape[0]):
        if out[i]:
            live += 1
    k = int(math.floor(update_fraction * live))

    if k == 0:
        return out

    active = []
    for i in range(out.shape[0]):
        if out[i] == 1:
            active.append(i)

    drop_order = sorted(active, key=lambda i: (abs(float(w[i])), i))
    for i in drop_order[:k]:
        out[i] = 0

    candidates = []
    for i in range(out.shape[0]):
        if out[i] == 0:
            candidates.append(i)

    grow_order = sorted(
        candidates,
        key=lambda i: (-abs(float(grad[i])), i),
    )
    for i in grow_order[:k]:
        out[i] = 1

    return out
