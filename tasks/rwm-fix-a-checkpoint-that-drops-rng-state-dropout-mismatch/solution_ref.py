import random


def _matmul(A, B):
    rows_A = len(A)
    cols_A = len(A[0]) if rows_A > 0 else 0
    rows_B = len(B)
    cols_B = len(B[0]) if rows_B > 0 else 0

    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            aik = A[i][k]
            if aik == 0.0:
                continue
            B_k = B[k]
            res_i = result[i]
            for j in range(cols_B):
                res_i[j] += aik * B_k[j]
    return result


def _transpose(A):
    if not A:
        return []
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def _block(x, W1, W2, mask, p):
    h = _matmul(x, W1)
    r = [[max(val, 0.0) for val in row] for row in h]
    d = [[r_val * m_val / (1.0 - p) for r_val, m_val in zip(r_row, m_row)] for r_row, m_row in zip(r, mask)]
    y = _matmul(d, W2)
    return y, h, r, d


def checkpointed_layer(x: list[list[float]], W1: list[list[float]], W2: list[list[float]], p: float, seed: int, n_pre: int, n_post: int, dY: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    """Activation-checkpointed dropout block: forward once, discard the
    intermediates, then RECOMPUTE for backward under the identical dropout
    mask by saving/restoring the shared RNG's state around the block.

    Returns (Y, dX, dW1, dW2).
    """
    rng = random.Random(seed)
    for _ in range(n_pre):
        for _ in range(3):
            rng.random()

    # -- checkpoint entry: snapshot the RNG state before consuming any of
    # this block's own randomness, exactly like torch.utils.checkpoint
    # saving torch.get_rng_state() before calling the wrapped function.
    entry_state = rng.getstate()

    rows = len(x)
    cols = len(W1[0]) if W1 else 0
    mask = [[1.0 if rng.random() >= p else 0.0 for _ in range(cols)] for _ in range(rows)]

    Y, _h, _r, _d = _block(x, W1, W2, mask, p)
    # intermediates (_h, _r, _d, mask) are "discarded" here -- only Y survives
    # into the rest of the (simulated) network's forward pass.

    for _ in range(n_post):
        for _ in range(3):
            rng.random()  # later layers / rest of the network run here

    # -- backward: recompute the block under the SAME dropout mask by
    # restoring the RNG to its state at checkpoint entry.
    rng.setstate(entry_state)
    mask_recompute = [[1.0 if rng.random() >= p else 0.0 for _ in range(cols)] for _ in range(rows)]
    _y2, h, r, d = _block(x, W1, W2, mask_recompute, p)

    dW2 = _matmul(_transpose(d), dY)
    dd = _matmul(dY, _transpose(W2))
    dr = [[dd_val * m_val / (1.0 - p) for dd_val, m_val in zip(dd_row, m_row)] for dd_row, m_row in zip(dd, mask_recompute)]
    dh = [[dr_val if h_val > 0 else 0.0 for dr_val, h_val in zip(dr_row, h_row)] for dr_row, h_row in zip(dr, h)]
    dW1 = _matmul(_transpose(x), dh)
    dX = _matmul(dh, _transpose(W1))

    return Y, dX, dW1, dW2
