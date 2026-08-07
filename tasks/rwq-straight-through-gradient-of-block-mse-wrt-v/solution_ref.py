def ste_block_mse_grad_wrt_v(X: list[list[float]], W: list[list[float]], V: list[list[float]],
                              scale: list[float], bits: int) -> list[list[float]]:
    """Straight-through-estimator gradient of the block MSE loss wrt V.

    See task.md for the derivation. Treats round() as identity except
    where the clip actually saturates (mask == 0 there).
    """
    qmax = (1 << (bits - 1)) - 1

    B = len(X)
    I = len(X[0])
    O = len(W)

    mask_list = []
    Wq_list = []
    for o in range(O):
        s = scale[o]
        mask_row = []
        Wq_row = []
        for i in range(I):
            val = V[o][i]
            r_val = val / s
            if abs(r_val) <= qmax + 0.5:
                mask_row.append(1.0)
            else:
                mask_row.append(0.0)

            rounded = round(r_val)
            clipped = max(-qmax, min(qmax, rounded))
            Wq_row.append(s * clipped)
        mask_list.append(mask_row)
        Wq_list.append(Wq_row)

    pred = [[sum(X[b][i] * Wq_list[o][i] for i in range(I)) for o in range(O)] for b in range(B)]
    target = [[sum(X[b][i] * W[o][i] for i in range(I)) for o in range(O)] for b in range(B)]

    diff = [[pred[b][o] - target[b][o] for o in range(O)] for b in range(B)]

    term = [[sum(diff[b][o] * X[b][i] for b in range(B)) for i in range(I)] for o in range(O)]

    factor = 2.0 / (B * O)
    res = [[mask_list[o][i] * factor * term[o][i] for i in range(I)] for o in range(O)]

    return res
