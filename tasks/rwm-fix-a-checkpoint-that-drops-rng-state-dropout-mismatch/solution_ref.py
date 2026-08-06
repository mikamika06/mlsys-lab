import numpy as np


def _block(x, W1, W2, mask, p):
    N = x.shape[0]
    D1 = x.shape[1]
    D2 = W1.shape[1]
    O = W2.shape[1]

    h = np.zeros((N, D2), dtype=np.float64)
    for i in range(N):
        for j in range(D2):
            acc = 0.0
            for k in range(D1):
                acc += x[i, k] * W1[k, j]
            h[i, j] = acc

    r = np.zeros((N, D2), dtype=np.float64)
    for i in range(N):
        for j in range(D2):
            r[i, j] = h[i, j] if h[i, j] > 0.0 else 0.0

    scale = 1.0 / (1.0 - p)
    d = np.zeros((N, D2), dtype=np.float64)
    for i in range(N):
        for j in range(D2):
            d[i, j] = r[i, j] * mask[i, j] * scale

    y = np.zeros((N, O), dtype=np.float64)
    for i in range(N):
        for j in range(O):
            acc = 0.0
            for k in range(D2):
                acc += d[i, k] * W2[k, j]
            y[i, j] = acc

    return y, h, r, d


def checkpointed_layer(x, W1, W2, p, seed, n_pre, n_post, dY):
    """Activation-checkpointed dropout block: forward once, discard the
    intermediates, then RECOMPUTE for backward under the identical dropout
    mask by saving/restoring the shared RNG's state around the block.

    Returns (Y, dX, dW1, dW2).
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    dY = np.asarray(dY, dtype=np.float64)

    rng = np.random.default_rng(seed)
    for _ in range(n_pre):
        rng.random(3)

    entry_state = rng.bit_generator.state

    rand_vals = rng.random((x.shape[0], W1.shape[1]))
    mask = np.zeros((x.shape[0], W1.shape[1]), dtype=np.float64)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i, j] = 1.0 if rand_vals[i, j] >= p else 0.0

    Y, _h, _r, _d = _block(x, W1, W2, mask, p)

    for _ in range(n_post):
        rng.random(3)

    rng.bit_generator.state = entry_state
    rand_vals_recompute = rng.random((x.shape[0], W1.shape[1]))
    mask_recompute = np.zeros((x.shape[0], W1.shape[1]), dtype=np.float64)
    for i in range(mask_recompute.shape[0]):
        for j in range(mask_recompute.shape[1]):
            mask_recompute[i, j] = 1.0 if rand_vals_recompute[i, j] >= p else 0.0

    _y2, h, r, d = _block(x, W1, W2, mask_recompute, p)

    N = d.shape[0]
    D2 = d.shape[1]
    O = dY.shape[1]
    D1 = x.shape[1]

    dW2 = np.zeros((D2, O), dtype=np.float64)
    for i in range(D2):
        for j in range(O):
            acc = 0.0
            for k in range(N):
                acc += d[k, i] * dY[k, j]
            dW2[i, j] = acc

    dd = np.zeros((N, D2), dtype=np.float64)
    for i in range(N):
        for j in range(D2):
            acc = 0.0
            for k in range(O):
                acc += dY[i, k] * W2[j, k]
            dd[i, j] = acc

    scale = 1.0 / (1.0 - p)
    dr = np.zeros((N, D2), dtype=np.float64)
    for i in range(N):
        for j in range(D2):
            dr[i, j] = dd[i, j] * mask_recompute[i, j] * scale

    dh = np.zeros((N, D2), dtype=np.float64)
    for i in range(N):
        for j in range(D2):
            cond = 1.0 if h[i, j] > 0.0 else 0.0
            dh[i, j] = dr[i, j] * cond

    dW1 = np.zeros((D1, D2), dtype=np.float64)
    for i in range(D1):
        for j in range(D2):
            acc = 0.0
            for k in range(N):
                acc += x[k, i] * dh[k, j]
            dW1[i, j] = acc

    dX = np.zeros((N, D1), dtype=np.float64)
    for i in range(N):
        for j in range(D1):
            acc = 0.0
            for k in range(D2):
                acc += dh[i, k] * W1[j, k]
            dX[i, j] = acc

    return Y, dX, dW1, dW2
