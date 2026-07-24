import numpy as np


def _block(x, W1, W2, mask, p):
    h = x @ W1
    r = np.maximum(h, 0.0)
    d = r * mask / (1.0 - p)
    y = d @ W2
    return y, h, r, d


def checkpointed_layer(x, W1, W2, p, seed, n_pre, n_post, dY):
    """Activation-checkpointed dropout block: forward once, discard the
    intermediates, then RECOMPUTE for backward under the identical dropout
    mask by saving/restoring the shared RNG's state around the block.

    Returns (Y, dX, dW1, dW2).

    BUG: this version never snapshots the RNG state at checkpoint entry, so
    the recompute below draws its mask from wherever the shared stream
    happens to be sitting after `n_post` unrelated draws -- a different mask
    than the one the original forward pass used. Fix it.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    dY = np.asarray(dY, dtype=np.float64)

    rng = np.random.default_rng(seed)
    for _ in range(n_pre):
        rng.random(3)

    mask = (rng.random((x.shape[0], W1.shape[1])) >= p).astype(np.float64)
    Y, _h, _r, _d = _block(x, W1, W2, mask, p)

    for _ in range(n_post):
        rng.random(3)

    # recompute WITHOUT restoring the RNG -- wrong mask.
    mask_recompute = (rng.random((x.shape[0], W1.shape[1])) >= p).astype(np.float64)
    _y2, h, r, d = _block(x, W1, W2, mask_recompute, p)

    dW2 = d.T @ dY
    dd = dY @ W2.T
    dr = dd * mask_recompute / (1.0 - p)
    dh = dr * (h > 0).astype(np.float64)
    dW1 = x.T @ dh
    dX = dh @ W1.T

    return Y, dX, dW1, dW2
