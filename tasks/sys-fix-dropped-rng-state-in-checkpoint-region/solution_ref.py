import numpy as np


def checkpointed_dropout_block(x, W, p, seed, n_pre, n_between):
    """Checkpointed linear -> relu -> dropout block sharing one RNG stream
    with the rest of the (simulated) network.

    Returns (y_forward, y_recomputed): the forward pass's output, and the
    output produced when the block is later recomputed (as activation
    checkpointing does during backward) after `n_between` unrelated draws
    from the same shared RNG have happened in between.
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    rng = np.random.default_rng(seed)
    for _ in range(n_pre):
        rng.random(3)

    # -- checkpoint entry: snapshot the RNG state before this block draws
    # any of its own randomness (mirrors torch.utils.checkpoint saving the
    # RNG state before invoking the wrapped function).
    entry_state = rng.bit_generator.state

    mask = (rng.random(x.shape[0] * W.shape[1]).reshape(x.shape[0], W.shape[1]) >= p).astype(np.float64)
    h = np.maximum(x @ W, 0.0)
    y_forward = h * mask / (1.0 - p)
    # mask/h are "discarded" here -- only y_forward survives forward.

    for _ in range(n_between):
        rng.random(3)  # rest of the network runs here, consuming the shared RNG

    # -- backward: recompute the block under the SAME dropout mask by
    # restoring the RNG to its state at checkpoint entry first.
    rng.bit_generator.state = entry_state
    mask2 = (rng.random(x.shape[0] * W.shape[1]).reshape(x.shape[0], W.shape[1]) >= p).astype(np.float64)
    h2 = np.maximum(x @ W, 0.0)
    y_recomputed = h2 * mask2 / (1.0 - p)

    return y_forward, y_recomputed
