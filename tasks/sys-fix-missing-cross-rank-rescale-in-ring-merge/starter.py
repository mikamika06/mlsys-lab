import numpy as np


def ring_merge(partials):
    # TODO: missing cross-rank rescale.
    # It incorrectly adds local statistics that use different maxima.
    ls = [np.asarray(p[1], dtype=np.float64) for p in partials]
    accs = [np.asarray(p[2], dtype=np.float64) for p in partials]

    total_l = np.zeros_like(ls[0])
    total_a = np.zeros_like(accs[0])

    for l, a in zip(ls, accs):
        total_l += l
        total_a += a

    return total_a / total_l[:, None]
