import numpy as np
from gptq.scales import compute_group_scales


def gptq_quantize_with_recompute(w: np.ndarray, h: np.ndarray, group_size: int, bits: int) -> np.ndarray:
    w_q = w.copy()
    rows, cols = w.shape
    max_val = float(2 ** (bits - 1) - 1)

    invh = np.linalg.inv(h)

    for i in range(cols):
        if i > 0 and i % group_size == 0:
            pass

        current_scales = compute_group_scales(w_q, group_size, bits)

        col_w = w_q[:, i]
        g_idx = i // group_size
        scale = current_scales[:, g_idx]

        q = np.round(col_w / scale)
        q = np.clip(q, -max_val, max_val)
        q_w = q * scale

        err = (col_w - q_w) / invh[i, i]
        w_q[:, i] = q_w
        if i + 1 < cols:
            w_q[:, i+1:] -= np.outer(err, invh[i, i+1:])

    return w_q
