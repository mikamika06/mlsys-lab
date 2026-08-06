import numpy as np


def ste_block_mse_grad_wrt_v(X: np.ndarray, W: np.ndarray, V: np.ndarray,
                              scale: np.ndarray, bits: int) -> np.ndarray:
    """Straight-through-estimator gradient of the block MSE loss wrt V.

    See task.md for the derivation. Treats round() as identity except
    where the clip actually saturates (mask == 0 there).
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1

    B, I = X.shape
    O, _ = W.shape

    mask_list = []
    Wq_list = []
    for o in range(O):
        s = scale[o]
        mask_row = []
        Wq_row = []
        for i in range(I):
            val = V[o, i]
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

    pred = [[sum(X[b, i] * Wq_list[o][i] for i in range(I)) for o in range(O)] for b in range(B)]
    target = [[sum(X[b, i] * W[o, i] for i in range(I)) for o in range(O)] for b in range(B)]

    diff = [[pred[b][o] - target[b][o] for o in range(O)] for b in range(B)]

    term = [[sum(diff[b][o] * X[b, i] for b in range(B)) for i in range(I)] for o in range(O)]

    factor = 2.0 / (B * O)
    res = [[mask_list[o][i] * factor * term[o][i] for i in range(I)] for o in range(O)]

    return np.array(res, dtype=np.float64)
