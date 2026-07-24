import numpy as np


def checkpointed_dropout_block(x, W, p, seed, n_pre, n_between):
    """Checkpointed linear -> relu -> dropout block sharing one RNG stream
    with the rest of the (simulated) network.

    Returns (y_forward, y_recomputed): the forward pass's output, and the
    output produced when the block is later recomputed (as activation
    checkpointing does during backward) after `n_between` unrelated draws
    from the same shared RNG have happened in between.

    BUG: this version never snapshots the RNG state at checkpoint entry, so
    the recompute below draws its mask from wherever the shared stream
    happens to be sitting after the `n_between` unrelated draws -- a
    different mask than the one the original forward pass used. Fix it.
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    rng = np.random.default_rng(seed)
    for _ in range(n_pre):
        rng.random(3)

    mask = (rng.random(x.shape[0] * W.shape[1]).reshape(x.shape[0], W.shape[1]) >= p).astype(np.float64)
    h = np.maximum(x @ W, 0.0)
    y_forward = h * mask / (1.0 - p)

    for _ in range(n_between):
        rng.random(3)

    # recompute WITHOUT restoring the RNG -- wrong mask.
    mask2 = (rng.random(x.shape[0] * W.shape[1]).reshape(x.shape[0], W.shape[1]) >= p).astype(np.float64)
    h2 = np.maximum(x @ W, 0.0)
    y_recomputed = h2 * mask2 / (1.0 - p)

    return y_forward, y_recomputed
