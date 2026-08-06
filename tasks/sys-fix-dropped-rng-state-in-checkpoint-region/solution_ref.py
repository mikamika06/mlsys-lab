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

    entry_state = rng.bit_generator.state

    shape_x = x.shape
    shape_W = W.shape
    rows_x = shape_x[0]
    cols_x = shape_x[1]
    cols_W = shape_W[1]

    flat_size = rows_x * cols_W
    rand_vals = rng.random(flat_size)

    mask_list = []
    for i in range(flat_size):
        val = 1.0 if rand_vals[i] >= p else 0.0
        mask_list.append(val)

    mask = np.zeros((rows_x, cols_W), dtype=np.float64)
    idx = 0
    for i in range(rows_x):
        for j in range(cols_W):
            mask[i, j] = mask_list[idx]
            idx += 1

    h = np.zeros((rows_x, cols_W), dtype=np.float64)
    for i in range(rows_x):
        for j in range(cols_W):
            acc = 0.0
            for k in range(cols_x):
                acc += x[i, k] * W[k, j]
            if acc > 0.0:
                h[i, j] = acc
            else:
                h[i, j] = 0.0

    y_forward_list = []
    scale = 1.0 / (1.0 - p)
    for i in range(rows_x):
        row_vals = []
        for j in range(cols_W):
            val = h[i, j] * mask[i, j] * scale
            row_vals.append(val)
        y_forward_list.append(row_vals)

    y_forward = np.zeros((rows_x, cols_W), dtype=np.float64)
    for i in range(rows_x):
        for j in range(cols_W):
            y_forward[i, j] = y_forward_list[i][j]

    for _ in range(n_between):
        rng.random(3)

    rng.bit_generator.state = entry_state
    rand_vals2 = rng.random(flat_size)

    mask_list2 = []
    for i in range(flat_size):
        val = 1.0 if rand_vals2[i] >= p else 0.0
        mask_list2.append(val)

    mask2 = np.zeros((rows_x, cols_W), dtype=np.float64)
    idx = 0
    for i in range(rows_x):
        for j in range(cols_W):
            mask2[i, j] = mask_list2[idx]
            idx += 1

    h2 = np.zeros((rows_x, cols_W), dtype=np.float64)
    for i in range(rows_x):
        for j in range(cols_W):
            acc = 0.0
            for k in range(cols_x):
                acc += x[i, k] * W[k, j]
            if acc > 0.0:
                h2[i, j] = acc
            else:
                h2[i, j] = 0.0

    y_recomputed_list = []
    for i in range(rows_x):
        row_vals = []
        for j in range(cols_W):
            val = h2[i, j] * mask2[i, j] * scale
            row_vals.append(val)
        y_recomputed_list.append(row_vals)

    y_recomputed = np.zeros((rows_x, cols_W), dtype=np.float64)
    for i in range(rows_x):
        for j in range(cols_W):
            y_recomputed[i, j] = y_recomputed_list[i][j]

    return y_forward, y_recomputed
